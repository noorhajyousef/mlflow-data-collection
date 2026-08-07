"""
count_files.py — count files and folders under a path, broken down by
extension. Useful before running the pipeline on a large dataset, to
avoid accidentally kicking off a multi-hour run on thousands of files
when you only meant to test on a handful (this exact mistake happened
once during development — INPUT_DIR pointed at the full dataset
instead of a small test folder).

Usage:
  python utils/count_files.py --path dataset/original_code
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, type=Path)
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Path does not exist: {args.path}")
        return

    n_files = 0
    n_dirs = 0
    ext_counts: Counter[str] = Counter()
    top_level_dirs = set()

    for entry in args.path.rglob("*"):
        if entry.is_dir():
            n_dirs += 1
        elif entry.is_file():
            n_files += 1
            ext_counts[entry.suffix or "(no extension)"] += 1
            rel = entry.relative_to(args.path)
            if len(rel.parts) > 1:
                top_level_dirs.add(rel.parts[0])

    print(f"Path: {args.path}")
    print(f"Total files: {n_files}")
    print(f"Total subfolders: {n_dirs}")
    print(f"Top-level subfolders (repos): {len(top_level_dirs)}")
    print("\nFiles by extension:")
    for ext, count in ext_counts.most_common():
        print(f"  {ext}: {count}")

    if n_files > 50:
        print(f"\nNote: {n_files} files is enough to take a long time locally on a "
              f"small model, and much longer on a large one. Consider pointing "
              f"--input-dir at a small subset first if this is a smoke test.")


if __name__ == "__main__":
    main()
