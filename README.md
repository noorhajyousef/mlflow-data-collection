# MLflow Data Collection — Handoff Package

Dataset and collection pipeline for "A Comparative Study of LLMs for MLflow Code Generation" (NSERC USRA, supervised by Dr. Ahmad Abdellatif, mentored by Corey Yang-Smith).

Maintainer: Noor Haj Yousef (GitHub: noorhajyousef)
Handoff to: Houaida Mangour (inference pipeline development)

## 1. Dataset summary

Final verified funnel (as of July 2026):

| Stage | Count |
|---|---|
| SEART GHS export (candidate repos) | 15,739 |
| After license filter | 9,154 |
| After contamination check (cutoff: created on/after 2025-04-29) | 9,154 (0 removed; satisfied by construction) |
| Pre-filter hits (repos likely containing MLflow) | 65 (52 root-only scan + 13 widened subfolder scan) |
| Final dataset | 43 repos / 147 files / 715 evaluation instances |

The contamination cutoff (April 29, 2025) is the Qwen3 release date.

## 2. How the data was collected

1. **Candidate repositories** were pulled from SEART GitHub Search (https://seart-ghs.si.usi.ch/): Python as main language, created after 2025-08-31. Export: 15,739 repos. Note SEART only indexes repos with 10+ stars.
   <!-- TODO(Noor): confirm any additional SEART criteria (min stars/commits, non-fork, archived) per the methodology draft's open bracket -->
2. **License filter** kept MIT, Apache-2.0, BSD-2-Clause, and BSD-3-Clause only (15,739 -> 9,154; mostly MIT and Apache-2.0).
3. **Contamination check**: cutoff is 2025-04-29 (Qwen3 release; later than Llama 3.1's July 2024 release). Satisfied by construction since the earliest creation date in the set is 2025-08-31. Creation date (not last commit) is the agreed conservative field.
4. **Manifest pre-filter** (`local_scan.py` / `widened_scan.py`): scans each candidate's dependency manifests (requirements.txt, pyproject.toml, setup.py, setup.cfg, environment.yml, Pipfile, conda.yaml, etc.) for an mlflow declaration via the GitHub REST API. Root-level manifests first (52 repos), then widened to manifests in ALL subfolders via the Git Trees API (13 more; 65 total).
5. **MLflow file detection** (`local_detector.py`): for each pre-filtered repo, GitHub code search retrieves Python files mentioning mlflow; each is parsed with Python's ast module. A file is recorded if it imports mlflow (or a submodule) or contains direct call expressions on the bare `mlflow` module name. Stored per file: import flag and direct-call count. AST-parse failures fall back to a textual check. Duplicates from resumed runs are removed.
6. **Instance building** (`instance_builder.py`) creates one evaluation instance per qualifying call site: a context window of 40 lines before and 10 lines after the call, with the call line masked. Target = the original call line(s).
7. **Manual validation**: 10-sample review, 10/10 accuracy. Four edge-case patterns documented in the validation spreadsheet (lazy imports, module-as-attribute, name-shadowing, mock-shadowing).

## 3. Repository structure

```
mlflow-data-collection/
├── README.md                  <- this file
├── data/
│   ├── seart_export.csv       <- 15,739 candidate repos from SEART
│   ├── mlflow_files.csv       <- manifest of the 147 final MLflow files
│   └── instances.jsonl        <- 715 evaluation instances (if included; else on Fir)
├── scripts/
│   ├── local_scan.py          <- root-level import scanner
│   ├── widened_scan.py        <- subfolder manifest scanner
│   ├── local_detector.py      <- AST-based direct-call detector
│   └── instance_builder.py    <- builds instances.jsonl (has --dry-run)
├── inference/
│   ├── run_inference.py       <- DRAFT inference script
│   └── submit_inference.sh    <- DRAFT SLURM array job for Compute Canada (Fir)
├── docs/
│   ├── Task_Design_Proposal.pdf
│   └── validation_spreadsheet.xlsx
└── notebooks/
    └── MLflow_v3.ipynb        <- crash-proof resumable collection notebook
```


## 4. How to run the collection scripts
