"""
glm.py — binomial GLM comparing desc_model and code_model effects on BLEU-3.

Formula: bleu3 ~ C(desc_model) + C(code_model) + C(desc_model):C(code_model)

BLEU-3 is a bounded [0,1] continuous score, so we fit it as a proportion
under the Binomial family (a standard approach for bounded scores, though
not a true count model — this mirrors the original project's method).
Scores are clipped away from exact 0/1 to avoid fitting instabilities.

Outputs (to --output dir):
  - glm_summary.txt      full statsmodels summary (coefficients, p-values, etc.)
  - glm_odds_ratios.csv  exp(coef) per term
  - glm_predictions.csv  predicted BLEU-3 for every desc_model x code_model combo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed (works on headless cluster nodes too)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

EPS = 1e-4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["bleu3_clipped"] = df["bleu3"].clip(EPS, 1 - EPS)

    formula = "bleu3_clipped ~ C(desc_model) + C(code_model) + C(desc_model):C(code_model)"
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    result = model.fit()

    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.output / "glm_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(result.summary()))

    odds_ratios = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "odds_ratio": np.exp(result.params.values),
        "p_value": result.pvalues.values,
    })
    odds_ratios.to_csv(args.output / "glm_odds_ratios.csv", index=False)

    combos = df[["desc_model", "code_model"]].drop_duplicates()
    combos["predicted_bleu3"] = result.predict(combos)
    combos.to_csv(args.output / "glm_predictions.csv", index=False)

    # --- Plot 1: coefficients (excluding intercept), colored by significance ---
    coef_plot_df = odds_ratios[odds_ratios["term"] != "Intercept"].copy()
    if len(coef_plot_df) > 0:
        colors = ["tab:green" if p < 0.05 else "tab:gray" for p in coef_plot_df["p_value"]]
        plt.figure(figsize=(9, max(3, 0.4 * len(coef_plot_df))))
        plt.barh(coef_plot_df["term"], coef_plot_df["coef"], color=colors)
        plt.axvline(x=0, color="black", linewidth=1)
        plt.xlabel("Coefficient (log-odds scale)")
        plt.title("GLM coefficients (green = significant at p<0.05)")
        plt.tight_layout()
        plt.savefig(args.output / "glm_coefficients.png", dpi=200)
        plt.close()

    # --- Plot 2: predicted vs. actual (mean) BLEU-3 per combination ---
    actual_means = df.groupby(["desc_model", "code_model"])["bleu3"].mean().reset_index()
    merged = combos.merge(actual_means, on=["desc_model", "code_model"])
    plt.figure(figsize=(7, 7))
    plt.scatter(merged["bleu3"], merged["predicted_bleu3"], s=80, alpha=0.7)
    for _, row in merged.iterrows():
        plt.annotate(f"{row['desc_model']}→{row['code_model']}", (row["bleu3"], row["predicted_bleu3"]), fontsize=7)
    lo = min(merged["bleu3"].min(), merged["predicted_bleu3"].min())
    hi = max(merged["bleu3"].max(), merged["predicted_bleu3"].max())
    plt.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="Perfect prediction")
    plt.xlabel("Actual mean BLEU-3")
    plt.ylabel("Predicted BLEU-3")
    plt.title("GLM: predicted vs. actual BLEU-3 per combination")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "glm_predicted_vs_actual.png", dpi=200)
    plt.close()

    print(f"GLM fit complete -> {args.output}/")
    print(f"  glm_summary.txt, glm_odds_ratios.csv, glm_predictions.csv")
    print(f"  glm_coefficients.png, glm_predicted_vs_actual.png")
    print(odds_ratios.to_string(index=False))


if __name__ == "__main__":
    main()
