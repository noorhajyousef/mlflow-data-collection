"""
glm_fixed.py — more robust variant of glm.py.

Same formula and intent as glm.py, but:
  - clips scores more conservatively and warns if many values needed clipping
  - falls back to an OLS fit (with a note in the output) if the Binomial
    GLM fails to converge, rather than crashing
  - includes basic residual diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

EPS = 1e-3
FORMULA = "bleu3_clipped ~ C(desc_model) + C(code_model) + C(desc_model):C(code_model)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    n_clipped = ((df["bleu3"] <= EPS) | (df["bleu3"] >= 1 - EPS)).sum()
    if n_clipped:
        print(f"WARN: clipping {n_clipped}/{len(df)} extreme BLEU-3 values before fitting")
    df["bleu3_clipped"] = df["bleu3"].clip(EPS, 1 - EPS)

    args.output.mkdir(parents=True, exist_ok=True)
    fit_method = "binomial_glm"
    try:
        model = smf.glm(formula=FORMULA, data=df, family=sm.families.Binomial())
        result = model.fit()
    except Exception as exc:
        print(f"Binomial GLM failed to converge ({exc}); falling back to OLS on clipped scores.")
        fit_method = "ols_fallback"
        model = smf.ols(formula=FORMULA, data=df)
        result = model.fit()

    with open(args.output / "glm_fixed_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"fit_method: {fit_method}\n\n")
        f.write(str(result.summary()))

    odds_ratios = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "odds_ratio": np.exp(result.params.values) if fit_method == "binomial_glm" else np.nan,
        "p_value": result.pvalues.values,
    })
    odds_ratios.to_csv(args.output / "glm_fixed_odds_ratios.csv", index=False)

    combos = df[["desc_model", "code_model"]].drop_duplicates()
    combos["predicted_bleu3"] = result.predict(combos)
    combos.to_csv(args.output / "glm_fixed_predictions.csv", index=False)

    residuals = result.resid_response if hasattr(result, "resid_response") else result.resid

    plt.figure(figsize=(11, 4.5))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual distribution")

    plt.subplot(1, 2, 2)
    plt.scatter(result.fittedvalues, residuals, alpha=0.4)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residual")
    plt.title("Residuals vs. fitted")
    plt.tight_layout()
    plt.savefig(args.output / "glm_fixed_residuals.png", dpi=200)
    plt.close()

    print(f"Fit method used: {fit_method}")
    print(f"Residuals — mean: {residuals.mean():.4f}, std: {residuals.std():.4f}")
    print(f"Output -> {args.output}/ (glm_fixed_summary.txt, glm_fixed_odds_ratios.csv, glm_fixed_predictions.csv, glm_fixed_residuals.png)")


if __name__ == "__main__":
    main()
