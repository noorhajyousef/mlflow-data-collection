"""
diagnose.py — flag model combinations likely to cause GLM fitting issues.

Combinations with near-zero variance, or scores clustered at 0 or 1,
can cause "separation" in a binomial GLM (unstable/extreme
coefficients, convergence failures). Run this BEFORE glm.py if a fit
looks suspicious, to see which combinations are the likely cause.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Path to all_combinations_bleu3_data.csv")
    ap.add_argument("--output", type=Path, default=None, help="Optional CSV to save the full stats table")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    stats = df.groupby(["desc_model", "code_model"])["bleu3"].agg(
        n_obs="count", mean="mean", std="std", min="min", max="max",
        n_zeros=lambda s: (s == 0).sum(),
        n_ones=lambda s: (s == 1).sum(),
    ).round(4)

    print("Per-combination statistics:")
    print(stats.to_string())

    problems = stats[
        (stats["std"].fillna(0) < 0.05)
        | (stats["mean"] > 0.95)
        | (stats["mean"] < 0.05)
        | (stats["n_ones"] > stats["n_obs"] * 0.9)
    ]

    print("\nPotentially problematic combinations (low variance / clustered near 0 or 1):")
    if len(problems) > 0:
        print(problems.to_string())
        print(f"\n{len(problems)} combination(s) flagged — GLM fit may need glm_fixed.py or more data for these.")
    else:
        print("None found — GLM should fit without separation issues.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.output)
        print(f"\nFull stats table saved -> {args.output}")


if __name__ == "__main__":
    main()
