# MLflow Inference Pipeline — Description → Code

A two-step LLM inference pipeline for the *"Comparative Study of LLMs
for MLflow Code Generation"* project:

1. **Descriptions**: read `.py` files, generate a natural-language
   description of each.
2. **Code**: read those descriptions, regenerate Python code from them.
3. **Evaluation**: score regenerated code against the originals
   (BLEU-3), then statistically compare model combinations.

Built to run first locally with small models (fast iteration), then
unchanged on Compute Canada with the target models only the
`--model` argument changes.

See [`DATAFLOW.md`](DATAFLOW.md) for a visual map of how data moves
through every step below.

---

## 1. Setup

```bash
python -m venv venv

# activate the environment:
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows cmd
venv\Scripts\Activate.ps1         # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env              # then edit .env and add your HF_TOKEN
```

`HF_TOKEN` is required even for public models (get one at
https://huggingface.co/settings/tokens). For any Llama model, also
request "gated" access on that model's Hugging Face page first 
Qwen models have no such requirement.

If `evaluation/bleu3_score.py` complains about missing NLTK data on
first run:
```bash
python -c "import nltk; nltk.download('punkt')"
```

**Windows PowerShell users:** environment variables need
`$env:VAR = "value"` — plain `set VAR=value` (cmd syntax) silently
does nothing in PowerShell. This project avoids relying on env vars
for paths (everything is a `--flag` instead) specifically to sidestep
that trap; `HF_TOKEN` in `.env` is the only exception.

---

## 2. Put input files in place

**The real dataset for the
first time**, the raw `.py` files aren't stored anywhere yet — only
`data/mlflow_files.csv` (repo + file path) exists. Reconstruct them
first:

```bash
python utils/fetch_raw_files.py \
  --manifest ../data/mlflow_files.csv \
  --output-dir dataset/original_code
```

(Adjust the `--manifest` path to wherever `mlflow_files.csv` actually
sits relative to this folder in the repo.) This re-downloads each
file from its GitHub repo (tries `main` then `master`) into
`dataset/original_code/<repo>/<file_path>` — the exact structure the
pipeline expects. It prints a summary of any files it couldn't fetch
(e.g. a repo using a non-standard default branch) — check that list
before assuming the full dataset made it in.

**Otherwise**, place `.py` files under `dataset/original_code/`
yourself, one subfolder per repo (the subfolder name becomes the
"repo name" used in output filenames):

```
dataset/original_code/
├── some-repo-1/
│   ├── file_a.py
│   └── file_b.py
└── some-repo-2/
    └── file_c.py
```

A working example file is already included at
`dataset/original_code/example-repo/example_train.py` so you can test
the whole pipeline immediately, before plugging in real data.

Before running on a large dataset, check its size first to avoid an
accidental multi-hour run:
```bash
python utils/count_files.py --path dataset/original_code
```

There's also `test_samples/` — a small fixture with a hand-written
reference description (`test_samples/reference_description.md`) for a
quick regression check after any code change. See that file for how
to use it.

---

## 3. Generate — smoke test with a small model

```bash
python run_pipeline.py descriptions --model Qwen/Qwen3-0.6B
```
Prints the exact output folder at the end, e.g.
`outputs/generated_descriptions/2026-08-06-Qwen-Qwen3-0.6B`. Use that
folder in the next command:

```bash
python run_pipeline.py code --model Qwen/Qwen3-0.6B \
    --desc-input-dir outputs/generated_descriptions/2026-08-06-Qwen-Qwen3-0.6B
```

Check the results:
- `outputs/generated_descriptions/<date>-<model>/*.txt`
- `outputs/generated_code/<date>-<model>/<repo>/*.py`
- `status.json` in each output folder — one entry per file, with
  `status: completed / failed / skipped`.

**Resuming after an interruption:** just re-run the exact same
command. Already-finished files are skipped automatically (tracked in
`status.json`) — nothing is redone, nothing already-written is lost.

---

## 4. Switch to the full target models

No code changes — only `--model` changes:

```bash
python run_pipeline.py descriptions --model Qwen/Qwen3-8B
python run_pipeline.py descriptions --model meta-llama/Llama-3.1-8B-Instruct
```

These need a real GPU with enough VRAM — run them on Compute Canada
(step 5), not locally.

---

## 5. Running on Compute Canada (cluster: Fir)

```bash
sbatch slurm/submit_pipeline.sh descriptions Qwen/Qwen3-8B dataset/original_code
# after it finishes, check the printed output folder, then:
sbatch slurm/submit_pipeline.sh code Qwen/Qwen3-8B outputs/generated_descriptions/<date>-<model>
```

Edit `#SBATCH --account=bmj-842-02` in `slurm/submit_pipeline.sh` if
your allocation code differs. Logs land in `logs/`.

---

## 6. Evaluate and compare models

Once you have generated code from **at least two different model
combinations**, score and compare them. Full explanation of every
step in [`DATAFLOW.md`](DATAFLOW.md); command sequence below.

### 6.1 (Optional but recommended) strip comments/docstrings first

Comments and docstrings are free-form text a model can phrase very
differently while producing functionally identical code — stripping
them keeps BLEU-3 focused on actual code structure:

```bash
python evaluation/no_comments.py \
  --input-dir dataset/original_code \
  --output-dir dataset/original_code_stripped

python evaluation/no_comments.py \
  --input-dir outputs/generated_code/<date>-<model> \
  --output-dir outputs/generated_code_stripped/<date>-<model>
```
Use the `_stripped` folders in step 6.2 instead of the raw ones if you
do this.

### 6.2 Score one generated set against the originals

```bash
python evaluation/bleu3_score.py \
  --generated-dir outputs/generated_code/<date>-<model> \
  --original-dir dataset/original_code \
  --output-csv data/bleu3_results/bleu3_scores_<desc_model>_to_<code_model>.csv
```

Repeat for every model combination you want to compare. The
`--output-csv` filename **must** follow the pattern
`bleu3_scores_<desc_model>_to_<code_model>.csv` exactly — the next
step parses model names out of the filename.

To see the overall distribution of a single scored set (useful before
comparing models — e.g. "80% of files scored below 0.5"):
```bash
python evaluation/score_distribution.py --input data/bleu3_results/bleu3_scores_<desc_model>_to_<code_model>.csv
```

### 6.3 Combine all BLEU-3 results into one dataset

```bash
python evaluation/all_combinations.py \
  --input-path data/bleu3_results \
  --output-path data
```
→ `data/all_combinations_bleu3_data.csv`

### 6.4 (Optional) Diagnose before fitting

If a GLM fit looks unstable or fails to converge, check which model
combinations might be causing it first:

```bash
python evaluation/diagnose.py \
  --data data/all_combinations_bleu3_data.csv \
  --output data/diagnose_stats.csv
```

Flags combinations with near-zero variance or scores clustered at 0/1
— common causes of GLM "separation" issues.

### 6.5 Fit the GLM comparing model effects

```bash
python evaluation/glm.py \
  --data data/all_combinations_bleu3_data.csv \
  --output data/glm_outputs
```
Produces `glm_summary.txt`, `glm_odds_ratios.csv`, `glm_predictions.csv`,
plus two plots: `glm_coefficients.png` (bar chart, significant terms in
green) and `glm_predicted_vs_actual.png` (scatter, closer to the
diagonal = better fit).

If it fails to converge, use the more robust variant instead (falls
back to OLS automatically, and adds residual diagnostic plots):
```bash
python evaluation/glm_fixed.py \
  --data data/all_combinations_bleu3_data.csv \
  --output data/glm_outputs
```

### 6.6 Significance tests (optional)

```bash
# unpaired, any two code_model groups:
python evaluation/mann_whitney.py \
  --data data/all_combinations_bleu3_data.csv \
  --output data/mann_whitney_results.csv

# paired, matched file-by-file scores — you prepare the comparison
# CSVs yourself under data/wilcoxon_comparisons/ first (see the
# docstring in evaluation/wilcoxon.py for the exact 2-column format):
python evaluation/wilcoxon.py \
  --comparisons-dir data/wilcoxon_comparisons \
  --output-dir data
```

---

## 7. Project structure

```
mlflow-inference-pipeline/
├── README.md                      <- this file
├── DATAFLOW.md                    <- visual map of the full pipeline
├── run_pipeline.py                <- single entry point (descriptions / code)
├── requirements.txt
├── .env.example
├── src/
│   ├── model_loader.py            <- loads model+tokenizer, handles generation + truncation
│   ├── progress_tracker.py        <- completed/failed/skipped tracking, resumable
│   ├── generate_descriptions.py   <- step 1: code -> description
│   ├── generate_code.py           <- step 2: description -> code
│   └── utils.py
├── prompts/                       <- editable prompt templates (system/user, per step)
├── evaluation/
│   ├── no_comments.py             <- strip comments/docstrings (optional pre-scoring step)
│   ├── bleu3_score.py             <- per-file BLEU-3 vs. originals
│   ├── score_distribution.py      <- cumulative score distribution for one CSV
│   ├── all_combinations.py        <- merges bleu3_scores_*.csv into one dataset
│   ├── diagnose.py                <- flags combinations likely to break the GLM fit
│   ├── glm.py                     <- binomial GLM comparing model effects (+ plots)
│   ├── glm_fixed.py               <- more robust GLM with OLS fallback (+ residual plots)
│   ├── mann_whitney.py            <- unpaired significance tests
│   └── wilcoxon.py                <- paired significance tests
├── dataset/original_code/         <- put input .py files here (example included)
├── test_samples/                  <- regression-check fixture with a reference description
├── utils/
│   ├── count_files.py             <- count files/folders before a run (avoid huge accidental runs)
│   └── fetch_raw_files.py         <- reconstruct raw .py files from mlflow_files.csv 
├── outputs/                       <- generated descriptions + code land here
├── data/                          <- evaluation outputs land here
├── slurm/submit_pipeline.sh       <- Compute Canada job script (account bmj-842-02)
```

---

## 8. Known limitations / notes

- **Truncation strategy**: if a file (or description) is too long for
  the model's context window, the **tail** (most recent/relevant
  part) is kept and the beginning is cut. Documented here per the
  context-limit handling requirement; expected to affect only a small
  subset of files.
- **Failed items** are recorded in `status.json` with the error
  message, but are **not** automatically retried on the next run
  (they count as "done" so the run doesn't loop on a permanently
  broken file). To force a retry, delete that item's entry from
  `status.json`.
- **Decoding**: generation uses greedy decoding (`do_sample=False`)
  by default in `src/model_loader.py` — the same input always
  produces the same output, which matters for a comparative study
  (with sampling on, re-scoring the same file twice gives different
  BLEU-3 numbers, which is noise unrelated to which model is
  actually better). Set `do_sample=True` in `GenerationParams` only
  for exploratory/creative use, not for the final comparison.
- **GLM on bounded scores**: BLEU-3 is a bounded `[0, 1]` continuous
  score, fit here as a proportion under the Binomial family (a common
  approximation, not a true count model) — scores are clipped away
  from exact 0/1 to avoid fitting instabilities.
- **GLM needs enough data per combination**: with only 1 observation
  per `desc_model`/`code_model` pair, statsmodels can't estimate
  variance (you'll see a `divide by zero` warning) and coefficient
  plots for non-intercept terms won't be generated, since there's
  nothing to compare against. Run `evaluation/diagnose.py` first if a
  fit looks off.
- **`bleu3_score.py` also reports a "Valid Python" column** — whether
  the generated file parses via `ast.parse()`. This is a companion
  metric, not a replacement for BLEU-3: a file can have low textual
  similarity to the original while still being valid, semantically
  equivalent code (or vice versa).
