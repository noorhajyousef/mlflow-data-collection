"""
score_distribution.py — cumulative distribution of BLEU-3 scores from a single
scoring CSV (the output of bleu3_score.py, i.e. columns File, BLEU-3).

Shows, for each 5% threshold, how many/what fraction of files scored
BELOW that threshold — useful to see the overall shape of results
(e.g. "80% of files score below 0.5" tells a different story than the
mean alone).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def analyze(df: pd.DataFrame) -> None:
    total = len(df)
    print("BLEU-3 cumulative distribution")
    print("=" * 50)
    for i in range(1, 21):
        threshold = i * 0.05
        count = (df["BLEU-3"] < threshold).sum()
        pct = count / total * 100
        print(f"< {int(threshold * 100):3d}% (< {threshold:.2f}): {count:4d} files ({pct:6.2f}%)")

    print("=" * 50)
    print(f"Total files: {total}")
    print(f"Mean:   {df['BLEU-3'].mean():.4f}")
    print(f"Median: {df['BLEU-3'].median():.4f}")
    print(f"Stddev: {df['BLEU-3'].std():.4f}")
    print(f"Min:    {df['BLEU-3'].min():.4f}")
    print(f"Max:    {df['BLEU-3'].max():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Path to a bleu3_score.py output CSV")
    ap.add_argument("--report", type=Path, default=None, help="Optional path to save a text report")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}")
        return

    df = pd.read_csv(args.input)

    if args.report:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyze(df)
        output = buf.getvalue()
        print(output)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
        print(f"Report saved -> {args.report}")
    else:
        analyze(df)


if __name__ == "__main__":
    main()
