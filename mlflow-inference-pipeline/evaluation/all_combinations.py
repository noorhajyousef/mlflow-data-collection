"""
all_combinations.py — combine many bleu3_scores_*.csv files into one dataset.

Expects filenames of the form:
    bleu3_scores_<desc_model>_to_<code_model>.csv
(model name components sanitized, e.g. slashes replaced by dashes/underscores
upstream — this script just splits on the literal "_to_").

Each row of the merged output gets: prompt_id, desc_model, code_model, bleu3.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(r"^bleu3_scores_(?P<desc_model>.+)_to_(?P<code_model>.+)\.csv$")


def parse_model_names(csv_path: Path) -> tuple[str, str]:
    m = FILENAME_RE.match(csv_path.name)
    if not m:
        raise ValueError(
            f"Filename does not match 'bleu3_scores_<desc_model>_to_<code_model>.csv': {csv_path.name}"
        )
    return m.group("desc_model"), m.group("code_model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-path", required=True, type=Path, help="Directory containing bleu3_scores_*.csv files")
    ap.add_argument("--output-path", required=True, type=Path, help="Directory to write the combined CSV into")
    args = ap.parse_args()

    csv_files = sorted(args.input_path.glob("bleu3_scores_*.csv"))
    if not csv_files:
        print(f"No bleu3_scores_*.csv files found in {args.input_path}")
        return

    frames = []
    for csv_file in csv_files:
        try:
            desc_model, code_model = parse_model_names(csv_file)
        except ValueError as exc:
            print(f"SKIP: {exc}")
            continue
        df = pd.read_csv(csv_file)
        df = df.rename(columns={"BLEU-3": "bleu3", "File": "file"})
        df["desc_model"] = desc_model
        df["code_model"] = code_model
        df["prompt_id"] = df["file"]
        frames.append(df[["prompt_id", "file", "desc_model", "code_model", "bleu3"]])

    if not frames:
        print("Nothing valid to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)
    args.output_path.mkdir(parents=True, exist_ok=True)
    out_file = args.output_path / "all_combinations_bleu3_data.csv"
    combined.to_csv(out_file, index=False)
    print(f"Combined {len(csv_files)} files, {len(combined)} rows -> {out_file}")
    print(combined.groupby(["desc_model", "code_model"])["bleu3"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
