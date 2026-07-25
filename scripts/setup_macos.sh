#!/usr/bin/env bash
# One-time setup on macOS. Run from the project root: ./scripts/setup_macos.sh
set -e

if ! command -v brew &> /dev/null; then
  echo "Homebrew not found. Install it from https://brew.sh first."
  exit 1
fi

echo "Installing tesseract (OCR engine)..."
brew install tesseract

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python dependencies (this pulls torch, may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit config.yaml — set watch_folders to your real Screenshots/Downloads paths."
echo "  2. source .venv/bin/activate"
echo "  3. python -m train.bootstrap_labels"
echo "  4. python -m train.review_labels --n 50"
echo "  5. python -m train.train_classifier"
echo "  6. python -m app.main    # then open http://127.0.0.1:8756"
