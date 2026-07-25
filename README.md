# Index — local screenshot & downloads search

A background service that watches your `Screenshots` and `Downloads`
folders, classifies every new file into a category, embeds it for semantic
search, and gives you a fast local search UI over all of it — filenames,
OCR'd text, and meaning ("that terminal error from last week"), not just
exact keywords. Everything runs on your machine. Nothing is uploaded anywhere.

## Architecture

```
                    ┌─────────────┐
  Screenshots/  ──▶ │   watcher   │  watchdog, debounced, background threads
  Downloads         └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │   extract   │  OCR (images) / text extraction (pdf, docx)
                     └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │    embed    │  CLIP image/text tower → 512-d shared vector
                     └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │  classify   │  trained LogisticRegression (or zero-shot CLIP
                     └──────┬──────┘  fallback before you've trained one)
                            ▼
                     ┌─────────────┐
                     │  LanceDB    │  embedded vector store, on-disk, no server
                     └──────┬──────┘
                            ▼
                     ┌─────────────┐
                     │  FastAPI    │  /api/search, /api/correct, ...
                     │  + HTML UI  │  semantic + keyword hybrid search
                     └─────────────┘
```

### Why this design, not a model trained from scratch

You explicitly asked for a trained model, and there genuinely is one here —
but training a CNN/ViT from scratch on your screenshot folder would need
thousands of labeled examples per category to not badly overfit, and you
don't have those (nobody does, for a personal folder). The correct
architecture is **transfer learning**:

- **Embeddings** come from a frozen, pretrained CLIP model (`open_clip`,
  ViT-B-32). Zero training required, and images/text share one vector
  space, so a text query compares directly against image embeddings.
- **The classifier** — `train/train_classifier.py` — is a genuinely
  trained model: scikit-learn `LogisticRegression` fit on top of those
  frozen embeddings, using labels you provide. Small labeled sets (tens
  to low-thousands of examples) are exactly the regime where a linear
  model on strong frozen features beats a deep model trained end-to-end.
  It trains in under a second and can be retrained after every correction
  with zero friction — that's what `/api/correct` + re-running the
  training script gives you: an active-learning loop that gets more
  accurate over time, personalized to *your* files.
- **Cold start** is handled by zero-shot classification (comparing a
  file's embedding to CLIP-encoded category descriptions) so the app is
  useful before you've labeled anything. `classify.py` automatically
  prefers the trained model once one exists.

## Setup

### 1. System dependencies

- **Tesseract OCR** (for reading text inside screenshots):
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: [installer here](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

(macOS users can instead run `./scripts/setup_macos.sh` to do both steps.)

### 3. Configure

Edit `config.yaml`:
- `watch_folders` — point these at your real Screenshots/Downloads paths.
- `categories` — edit names/prompts to match how *you* think about your files.
- `embedding.device` — leave as `auto` (picks CUDA/MPS/CPU automatically).

### 4. Bootstrap + train the classifier

```bash
python -m train.bootstrap_labels        # zero-shot labels every existing file
python -m train.review_labels --n 50    # (recommended) fix the 50 least-confident ones
python -m train.train_classifier        # fits the actual model, saves to models/
```

Skipping this is fine too — the app runs on the zero-shot fallback until you
train a model, which you can do at any point later.

### 5. Run

```bash
python -m app.main
```

Open **http://127.0.0.1:8756**. The watcher starts backfilling your existing
files in the background (see `backfill_on_first_run` in config) and picks up
new files as they land.

### 6. Improve it over time

In the UI, click a file's category tag to correct it. Corrections are logged
to `data/labels.csv`. Periodically re-run:

```bash
python -m train.train_classifier
curl -X POST http://127.0.0.1:8756/api/reload-classifier   # hot-swap, no restart
```

and the classifier gets sharper without ever leaving your machine.

### 7. (Optional) Run automatically on login/boot

- macOS: `scripts/com.local.screenshotsearch.plist` (launchd)
- Linux: `scripts/screenshot-search.service` (systemd --user)

Both need the placeholder paths edited to your actual install location —
instructions are in the comments at the top of each file.

## Project layout

```
app/
  config.py       config.yaml loader
  extract.py      OCR + text extraction per file type
  embed.py        CLIP embedding wrapper (image + text, shared space)
  classify.py     trained classifier + zero-shot fallback
  store.py        LanceDB vector store wrapper
  pipeline.py      extract -> embed -> classify -> store, one file in, one record out
  watcher.py       watchdog folder watcher, background thread pool
  api.py            FastAPI endpoints + serves the frontend
  main.py            entrypoint: watcher + API together
train/
  bootstrap_labels.py   zero-shot label every existing file
  review_labels.py       CLI active-learning review loop
  train_classifier.py    fits the LogisticRegression classifier
frontend/
  index.html              the search UI (single file, no build step)
scripts/                   OS-level auto-start + setup helpers
config.yaml                 all tunables live here
```

## Extending it

- **New categories**: edit `config.yaml`, re-run bootstrap + train.
- **New file types**: add a branch in `app/extract.py`.
- **More accurate classifier**: swap `LogisticRegression` for
  `sklearn.neural_network.MLPClassifier` in `train/train_classifier.py` —
  drop-in, same surrounding pipeline, useful once you have a few thousand
  labeled examples.
- **Bigger/more accurate embeddings**: change `embedding.model_name` /
  `pretrained` in `config.yaml` to a larger open_clip checkpoint (e.g.
  `ViT-L-14` / `laion2b_s32b_b82k`) if you have a GPU and want more accuracy
  at the cost of speed.
