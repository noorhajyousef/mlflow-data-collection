# Reference description — example-repo/train_classifier.py

This is a hand-written reference, not model output. Use it to
sanity-check that a generated description (step 1) is reasonable — it
doesn't need to match this wording, but it SHOULD mention the same key
facts:

- **Purpose**: trains a `RandomForestClassifier` on a dataset split
  with `train_test_split`, tracked with MLflow.
- **MLflow integration**:
  - `mlflow.set_experiment("regression-test-experiment")`
  - a run named `"rf-baseline"` via `mlflow.start_run(run_name=...)`
  - two params logged: `n_estimators` (100), `max_depth` (5)
  - one metric logged: `accuracy` (from `model.score(...)`)
  - the trained model logged with `mlflow.sklearn.log_model(model, "model")`
- **Structure**: a single function `train(X, y)` that returns the
  fitted model.
- **Data split**: 80/20 train/test split, `random_state=42` for both
  the split and the classifier (reproducibility).

## How to use this for a quick regression check

1. Run the pipeline on `test_samples/original_code/`:
   ```bash
   python run_pipeline.py descriptions --model Qwen/Qwen3-0.6B \
       --input-dir test_samples/original_code
   ```
2. Open the generated `.txt` description and compare against the
   bullet points above — flag it if the model completely misses the
   MLflow-specific details (params/metrics/model logging), since
   that's the part the whole project cares about.
3. Run step 2 on that output and check the regenerated code still
   contains equivalent `mlflow.*` calls, even if variable names or
   exact structure differ.

This is a fast sanity check, not a substitute for the real BLEU-3 /
GLM evaluation on the full dataset.
