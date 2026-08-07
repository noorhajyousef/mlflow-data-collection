"""
fetch_raw_files.py — reconstruct raw .py source files from a manifest
CSV (columns: repo, file_path) by re-downloading them from GitHub,
into the folder structure expected by run_pipeline.py:
    dataset/original_code/<repo>/<file_path>

Usage:
  python utils/fetch_raw_files.py --manifest ../data/mlflow_files.csv --output-dir dataset/original_code
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

BRANCHES_TO_TRY = ("main", "master")


def fetch_one(repo: str, file_path: str, output_dir: Path) -> bool:
    for branch in BRANCHES_TO_TRY:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            dest = output_dir / repo.replace("/", "__") / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(resp.text, encoding="utf-8", errors="replace")
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="Path to mlflow_files.csv")
    ap.add_argument("--output-dir", required=True, type=Path, help="Where to write files (e.g. dataset/original_code)")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)} files listed in manifest")
    ok, failed = 0, []
    for i, row in enumerate(rows, 1):
        repo, file_path = row["repo"], row["file_path"]
        print(f"[{i}/{len(rows)}] {repo}/{file_path}", end=" ")
        if fetch_one(repo, file_path, args.output_dir):
            print("OK")
            ok += 1
        else:
            print("FAILED (not on main or master)")
            failed.append(f"{repo}/{file_path}")
        time.sleep(0.1)  # be polite to GitHub's raw content servers

    print(f"\nDone: {ok}/{len(rows)} files fetched -> {args.output_dir}")
    if failed:
        print(f"\n{len(failed)} failed (repo may use a different default branch, or file since removed):")
        for f_ in failed:
            print(f"  - {f_}")


if __name__ == "__main__":
    main()
