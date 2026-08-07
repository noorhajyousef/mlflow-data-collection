"""
mann_whitney.py — unpaired Mann-Whitney U tests between code_model groups.

For every pair of distinct code_model values in --data, runs a two-sided
Mann-Whitney U test on their BLEU-3 distributions (pooled across all
desc_model / files) and writes one row per pair to --output.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    models = sorted(df["code_model"].unique())
    if len(models) < 2:
        print("Need at least 2 distinct code_model values to compare.")
        return

    rows = []
    for model_a, model_b in combinations(models, 2):
        scores_a = df.loc[df["code_model"] == model_a, "bleu3"]
        scores_b = df.loc[df["code_model"] == model_b, "bleu3"]
        stat, p_value = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "n_a": len(scores_a),
            "n_b": len(scores_b),
            "mean_a": scores_a.mean(),
            "mean_b": scores_b.mean(),
            "u_statistic": stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} comparisons -> {args.output}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
