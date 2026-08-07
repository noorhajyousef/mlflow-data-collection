# Data Flow

Visual map of how data moves through this project, from raw input to
final statistical comparison.

```
dataset/original_code/<repo>/<file>.py
            │
            │  src/generate_descriptions.py
            │  (model: e.g. Qwen3-0.6B, later Qwen3-8B / Llama-3.1-8B-Instruct)
            ▼
outputs/generated_descriptions/<date>-<model>/<repo>--<file>.txt
            │
            │  src/generate_code.py
            │  (model: same family, can differ from the description model)
            ▼
outputs/generated_code/<date>-<model>/<repo>/<file>.py
            │
            │  evaluation/no_comments.py   (optional, both sides)
            │  strips comments/docstrings before scoring
            ▼
   ┌────────┴─────────┐
   ▼                   ▼
original_code      generated_code
(stripped)          (stripped)
   │                   │
   └─────────┬─────────┘
             │  evaluation/bleu3_score.py
             ▼
data/bleu3_results/bleu3_scores_<desc_model>_to_<code_model>.csv
             │
             │  evaluation/all_combinations.py
             │  (run once per desc_model x code_model combination first,
             │   then combine all resulting CSVs together)
             ▼
data/all_combinations_bleu3_data.csv
             │
   ┌─────────┼─────────────────┐
   ▼         ▼                 ▼
evaluation/  evaluation/        evaluation/
glm.py       mann_whitney.py    wilcoxon.py
(or          (unpaired,         (paired,
glm_fixed.py) any 2 code_models) matched files)
   │         │                 │
   ▼         ▼                 ▼
data/glm_outputs/   data/mann_whitney_   data/
  glm_summary.txt     results.csv         wilcoxon_results.csv
  glm_odds_ratios.csv
  glm_predictions.csv
```

## Key idea

Every model combination you want to compare goes through the SAME
left-hand path (descriptions → code → BLEU-3 CSV) independently. Only
at the `all_combinations.py` step do the separate CSVs get merged into
one dataset that the statistical tools can compare across models.

## Minimal example for 2 models

To compare Qwen3-8B and Llama-3.1-8B-Instruct as the code-generation
model (keeping the same description model, e.g. Qwen3-8B):

1. Generate descriptions once: `descriptions --model Qwen/Qwen3-8B`
2. Generate code twice, from those same descriptions:
   - `code --model Qwen/Qwen3-8B`
   - `code --model meta-llama/Llama-3.1-8B-Instruct`
3. Score each generated set separately (2 CSV files, one per code_model)
4. Combine both CSVs with `all_combinations.py`
5. Compare with `glm.py` and/or `mann_whitney.py`
