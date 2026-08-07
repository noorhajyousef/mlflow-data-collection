"""
wilcoxon.py — paired Wilcoxon signed-rank tests for matched BLEU-3 pairs.

Expects --comparisons-dir to contain one CSV per comparison, each with
exactly two numeric columns: the matched BLEU-3 scores for two models
on the SAME set of files (same file, same order, one column per model).
The two column names are used as the model labels in the output.

Example comparisons-dir/qwen_vs_llama.csv:
    qwen,llama
    0.41,0.53
    0.30,0.28
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparisons-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    csv_files = sorted(args.comparisons_dir.glob("*.csv"))
    if not csv_files:
        print(f"No comparison CSV files found in {args.comparisons_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.shape[1] != 2:
            print(f"SKIP {csv_file.name}: expected exactly 2 columns, found {df.shape[1]}")
            continue
        col_a, col_b = df.columns[:2]
        try:
            stat, p_value = wilcoxon(df[col_a], df[col_b])
        except ValueError as exc:
            print(f"SKIP {csv_file.name}: {exc}")
            continue
        rows.append({
            "comparison": csv_file.stem,
            "model_a": col_a,
            "model_b": col_b,
            "n_pairs": len(df),
            "mean_a": df[col_a].mean(),
            "mean_b": df[col_b].mean(),
            "w_statistic": stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        })

    if not rows:
        print("No valid comparisons produced.")
        return

    result_df = pd.DataFrame(rows)
    out_file = args.output_dir / "wilcoxon_results.csv"
    result_df.to_csv(out_file, index=False)
    print(f"Wrote {len(rows)} comparisons -> {out_file}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
