"""
instance_builder.py

Builds masked-call completion instances from the collected MLflow dataset,
per Task_Design_Proposal.md (Sections 2 and 3).

Input:  mlflow_files.csv  (columns: repo, file_path, has_import, n_calls)
Output: instances.jsonl   (one JSON object per instance)

Design decisions implemented here (see Task_Design_Proposal.md for rationale):
  - Only DIRECT calls on the bare `mlflow` name are turned into instances
    (mlflow.log_param(...), not self._mlflow.log_param(...) or aliased/shadowed
    names). This matches the AST detector's own counting logic, so instance
    counts should line up with the n_calls column.
  - Test files are excluded entirely (path contains "/tests/" or "test_"
    filename), removing the MagicMock-shadowing pattern found in validation.
  - Context window: 40 lines before the call, 10 lines after.
  - One instance per qualifying call site (a file with 8 calls -> up to 8
    instances, fewer if some sites are too close to file start/end to be
    useful, which is logged, not silently dropped).
  - Both a "completion" prompt and an "instruction" prompt are generated per
    instance so the two prompting modes in the proposal can both be run
    without rebuilding instances.

Resumable: progress is checkpointed by (repo) so a killed run can restart
without re-fetching files already processed. Safe to re-run.

Usage:
    export GITHUB_TOKEN=...
    python instance_builder.py --input mlflow_files.csv --output instances.jsonl
"""

import argparse
import ast
import base64
import json
import os
import sys
import time

import requests

MASK_TOKEN = "<MLFLOW_CALL_MASKED>"
CONTEXT_BEFORE = 40
CONTEXT_AFTER = 10

INSTRUCTION_TEMPLATE = (
    "Complete the missing MLflow API call at the marker {mask}.\n"
    "Respond with only the replacement code for that line (or lines).\n\n"
    "{context}"
)


def gh_session(token):
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return s


def gh_get(session, url):
    while True:
        r = session.get(url)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = max(int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()) + 1, 1)
            print(f"  rate limit, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        return r


def fetch_file(session, repo, path):
    r = gh_get(session, f"https://api.github.com/repos/{repo}/contents/{path}")
    if r.status_code != 200:
        return None
    d = r.json()
    if d.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(d["content"]).decode("utf-8", "ignore")
    except Exception:
        return None


def is_test_file(path):
    lower = path.lower()
    fname = lower.split("/")[-1]
    return "/tests/" in f"/{lower}" or fname.startswith("test_") or fname.endswith("_test.py")


def find_direct_call_sites(source):
    """
    Returns a list of (start_line, end_line, call_name) for every direct call
    expression on the bare `mlflow` module name, e.g. mlflow.log_param(...).
    Lines are 1-indexed and inclusive, matching ast lineno/end_lineno.
    Mirrors the AST logic used by the collection pipeline's detector, so
    instance counts should align with the n_calls column in mlflow_files.csv.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # walk the attribute chain looking for a base Name == "mlflow"
        chain = []
        cur = f
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id == "mlflow" and chain:
            call_name = "mlflow." + ".".join(reversed(chain))
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is not None:
                sites.append((start, end, call_name))
    return sites


def build_instance(lines, start_line, end_line, call_name, repo, path):
    """
    lines: 0-indexed list of source lines (no trailing newlines).
    start_line/end_line: 1-indexed inclusive range of the call to mask.
    """
    n = len(lines)
    s0 = start_line - 1  # 0-indexed start of masked region
    e0 = end_line - 1    # 0-indexed end of masked region (inclusive)

    ctx_start = max(0, s0 - CONTEXT_BEFORE)
    ctx_end = min(n, e0 + 1 + CONTEXT_AFTER)

    target_lines = lines[s0:e0 + 1]
    target = "\n".join(target_lines).strip()
    if not target:
        return None

    before_ctx = lines[ctx_start:s0]
    after_ctx = lines[e0 + 1:ctx_end]
    context = "\n".join(before_ctx + [MASK_TOKEN] + after_ctx)

    return {
        "repo": repo,
        "file_path": path,
        "call_name": call_name,
        "start_line": start_line,
        "end_line": end_line,
        "mask_token": MASK_TOKEN,
        "target": target,
        "context_completion": context,
        "context_instruction": INSTRUCTION_TEMPLATE.format(mask=MASK_TOKEN, context=context),
    }


# ---------------------------------------------------------------------------
# Dry-run mode: exercises the full instance-building logic (AST call
# detection, test-file exclusion, masking, context window, both prompt
# modes) against a handful of small synthetic files held in memory. No
# network access and no GITHUB_TOKEN required. Use this to sanity-check the
# script before pointing it at the real dataset and GitHub API.
# ---------------------------------------------------------------------------

DRY_RUN_FILES = {
    ("demo/plain-import", "train.py"): '''\
import mlflow

def run():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("demo")
    with mlflow.start_run():
        mlflow.log_param("lr", 0.01)
        mlflow.log_metric("accuracy", 0.93)
        mlflow.log_artifact("model.pkl")
''',
    ("demo/lazy-import", "observability.py"): '''\
import logging

class Sink:
    def __post_init__(self):
        try:
            import mlflow
        except ImportError:
            self._available = False
            return
        self._mlflow = mlflow
        self._mlflow.set_tracking_uri("http://localhost:5000")  # attribute call: NOT counted
        self._available = True
''',
    ("demo/shadowed-name", "adapter.py"): '''\
from harness.mlflow_client import MLflowTraceClient

def build(tracking_uri):
    mlflow = MLflowTraceClient(tracking_uri)  # shadows the module name
    if mlflow.verify_connection():
        mlflow.log_run_metadata({"ok": True})  # counted per the documented limitation
    return mlflow
''',
    ("demo/test-file", "tests/test_tracker.py"): '''\
from unittest.mock import MagicMock

def test_logs_metric():
    mlflow = MagicMock()
    mlflow.log_metric("acc", 0.9)
    mlflow.log_metric.assert_called_once()
''',
    ("demo/no-mlflow", "utils.py"): '''\
def helper(x):
    return x * 2
''',
}


def run_dry_run(output_path):
    print("Running in DRY-RUN mode: no network access, no GITHUB_TOKEN needed.")
    print()

    total_instances = 0
    out_f = open(output_path, "w")

    for (repo, path), source in DRY_RUN_FILES.items():
        print(f"--- {repo} / {path} ---")

        if is_test_file(path):
            print("  SKIPPED (test file exclusion)")
            print()
            continue

        sites = find_direct_call_sites(source)
        if not sites:
            print("  0 direct mlflow.* call sites found")
            print()
            continue

        lines = source.splitlines()
        made = 0
        for start, end, call_name in sites:
            inst = build_instance(lines, start, end, call_name, repo, path)
            if inst is None:
                continue
            out_f.write(json.dumps(inst) + "\n")
            made += 1
            print(f"  instance: {call_name}  (lines {start}-{end})  target={inst['target']!r}")
        print(f"  -> {made} instance(s) from this file")
        print()
        total_instances += made

    out_f.close()

    print("=" * 60)
    print(f"DRY RUN COMPLETE: {total_instances} instance(s) written to {output_path}")
    print("Expected behavior to check by eye:")
    print("  - demo/plain-import:   6 instances (all direct mlflow.* calls)")
    print("  - demo/lazy-import:    0 instances (usage is via self._mlflow, an attribute)")
    print("  - demo/shadowed-name:  2 instances (documented limitation: local var named mlflow)")
    print("  - demo/test-file:      SKIPPED entirely (test file exclusion)")
    print("  - demo/no-mlflow:      0 instances (no mlflow usage)")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="mlflow_files.csv")
    ap.add_argument("--output", default="instances.jsonl")
    ap.add_argument("--checkpoint", default="instance_builder_processed.txt")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Test the pipeline logic on small synthetic files. No network, no GITHUB_TOKEN needed.",
    )
    args = ap.parse_args()

    if args.dry_run:
        run_dry_run(args.output if args.output != "instances.jsonl" else "instances_dryrun.jsonl")
        return

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("ERROR: set GITHUB_TOKEN before running (export GITHUB_TOKEN=...), or use --dry-run to test without it.")

    import pandas as pd  # local import so --help works without pandas installed

    df = pd.read_csv(args.input)
    df = df[df["file_path"].notna()]

    processed = set()
    if os.path.exists(args.checkpoint):
        processed = {l.strip() for l in open(args.checkpoint) if l.strip()}

    if not os.path.exists(args.output):
        open(args.output, "w").close()

    session = gh_session(token)

    # group by (repo, file_path) so each file is fetched once even if it
    # appears multiple times in the csv (shouldn't happen post-dedup, but safe)
    files = df[["repo", "file_path"]].drop_duplicates()
    key = lambda r: f"{r.repo}::{r.file_path}"
    todo = [row for row in files.itertuples() if key(row) not in processed]

    print(f"Total files: {len(files)} | already done: {len(processed)} | remaining: {len(todo)}", flush=True)

    total_instances = 0
    skipped_test_files = 0
    skipped_no_calls = 0

    out_f = open(args.output, "a")
    ckpt_f = open(args.checkpoint, "a")

    for i, row in enumerate(todo, 1):
        repo, path = row.repo, row.file_path

        if is_test_file(path):
            skipped_test_files += 1
            ckpt_f.write(key(row) + "\n")
            ckpt_f.flush()
            continue

        source = fetch_file(session, repo, path)
        if source is None:
            ckpt_f.write(key(row) + "\n")
            ckpt_f.flush()
            continue

        sites = find_direct_call_sites(source)
        if not sites:
            skipped_no_calls += 1
            ckpt_f.write(key(row) + "\n")
            ckpt_f.flush()
            continue

        lines = source.splitlines()
        made = 0
        for start, end, call_name in sites:
            inst = build_instance(lines, start, end, call_name, repo, path)
            if inst is None:
                continue
            out_f.write(json.dumps(inst) + "\n")
            made += 1
        out_f.flush()
        total_instances += made

        ckpt_f.write(key(row) + "\n")
        ckpt_f.flush()

        if i % 25 == 0:
            print(f"[{i}/{len(todo)}] instances so far: {total_instances}", flush=True)

        time.sleep(0.2)

    out_f.close()
    ckpt_f.close()

    print(
        f"\nFinished. New instances written: {total_instances} "
        f"(test files skipped: {skipped_test_files}, files with no direct calls: {skipped_no_calls})",
        flush=True,
    )


if __name__ == "__main__":
    main()