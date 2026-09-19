"""
Step 2 of training (optional but recommended): a fast terminal review loop
over the labels bootstrap_labels.py produced, worst-confidence first. This
is the "human-in-the-loop" part — a few minutes of corrections here has an
outsized effect on classifier accuracy, since these are exactly the examples
the model is currently most confused about.

Usage:
    python -m train.review_labels            # review the 30 least-confident
    python -m train.review_labels --n 100     # review more

Controls: type a category number to relabel it, Enter to accept the
suggested label as-is, 's' to skip, 'q' to save and quit.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import platform

from app.config import CFG


def open_file(path: str):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            import os
            os.startfile(path)  # type: ignore
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of lowest-confidence rows to review")
    parser.add_argument("--open-files", action="store_true", help="open each file in your default viewer")
    args = parser.parse_args()

    if not CFG.labels_csv.exists():
        print("No labels.csv found — run `python -m train.bootstrap_labels` first.")
        return

    with open(CFG.labels_csv) as f:
        rows = list(csv.DictReader(f))

    # Rows are already sorted worst-first by bootstrap_labels.py, but
    # re-sort defensively in case the file's been edited/appended to.
    def conf_of(r):
        src = r.get("source", "")
        if src.startswith("zero-shot("):
            return float(src[len("zero-shot("):-1])
        return 1.0  # user-corrected rows are treated as fully confident

    rows.sort(key=conf_of)
    names = CFG.category_names

    print("Categories:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    print("\nFor each file: type a number to relabel, Enter to accept, 's' to skip, 'q' to quit.\n")

    changed = 0
    for i, row in enumerate(rows[: args.n]):
        print(f"({i + 1}/{min(args.n, len(rows))}) {row['filepath']}")
        print(f"    current label: {row['category']}   ({row['source']})")
        if args.open_files:
            open_file(row["filepath"])

        choice = input("    > ").strip().lower()
        if choice == "q":
            break
        if choice == "s" or choice == "":
            continue
        if choice.isdigit() and 0 <= int(choice) < len(names):
            row["category"] = names[int(choice)]
            row["source"] = "user-reviewed"
            changed += 1
        else:
            print("    (not understood, skipping)")

    with open(CFG.labels_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "category", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved. {changed} label(s) corrected.")
    print("Next: python -m train.train_classifier")


if __name__ == "__main__":
    main()
