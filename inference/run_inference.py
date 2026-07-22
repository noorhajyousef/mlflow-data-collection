#!/usr/bin/env python3
"""
run_inference.py

Generate masked single-call completions for all evaluation instances,
for one model at a time, in one or both prompting modes.

Design (locked):
  - Task: masked single-call completion
  - Context window: 40 lines before the masked region, 10 lines after
  - Two prompting modes (see build_prompt below)
  - Models: Qwen/Qwen3-8B and meta-llama/Llama-3.1-8B-Instruct
  - Deterministic decoding (greedy) so runs are reproducible

Resumability:
  Output is JSONL, one record per (instance_id, mode). On startup the
  script reads the existing output file (if any) and skips pairs that
  are already complete, so a killed SLURM job can simply be resubmitted.

Usage (on Fir, inside the venv, on a GPU node):
  python run_inference.py --model qwen  --instances instances.jsonl --output outputs/qwen.jsonl
  python run_inference.py --model llama --instances instances.jsonl --output outputs/llama.jsonl

Smoke test (5 instances, one mode):
  python run_inference.py --model qwen --instances instances.jsonl \
      --output outputs/smoke.jsonl --limit 5 --modes instruction
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "qwen": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

# The two prompting modes from the Task Design Proposal (Section 3.4):
#   completion  - raw code prefix provided, model continues it (base-model
#                 usage; no chat template, no instruction, prefix only)
#   instruction - short instruction wraps the full context window with the
#                 mask marker (instruct-model usage; chat template)
MODES = ["completion", "instruction"]

GENERATION_CONFIG = {
    "max_new_tokens": 256,   # masked regions are short; adjust if needed
    "do_sample": False,      # greedy = deterministic = reproducible
    "temperature": None,     # unused when do_sample=False
}

MASK_TOKEN = "<MASKED>"      # must match what instance_builder.py emits


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(instance: dict, mode: str):
    """
    Build the model input for one instance in one mode.

    Expected instance fields (from instance_builder.py -- confirm names):
      instance_id     unique ID
      context_before  str, up to 40 lines of code before the mask
      context_after   str, up to 10 lines of code after the mask
      target          str, the masked call line(s) -- NEVER goes in the prompt

    Returns:
      ("raw", prompt_string)      for completion mode (no chat template)
      ("chat", list_of_messages)  for instruction mode (chat template)

    Task Design Proposal, Section 3.4:
      Completion mode: "the raw code prefix is provided and the model
      continues it, matching base-model usage."
      NOTE: as written this uses only the 40-line prefix; the 10-line
      suffix cannot be used in plain left-to-right continuation.
      [Flagged to Corey as an information-asymmetry question.]

      Instruction mode: "a short instruction wraps the context
      ('Complete the missing MLflow call at the marker'), matching
      instruct-model usage."
    """
    before = instance["context_before"]
    after = instance["context_after"]

    if mode == "completion":
        # Raw prefix continuation. The prompt ends exactly where the masked
        # call should begin, so the model's next tokens are the completion.
        return ("raw", before.rstrip("\n") + "\n")

    elif mode == "instruction":
        code_block = f"{before}\n{MASK_TOKEN}\n{after}"
        # Instruction wording verbatim from the Task Design Proposal, plus
        # a minimal output-format constraint so completions are scoreable.
        user = (
            f"Complete the missing MLflow call at the marker {MASK_TOKEN}.\n"
            "Reply with only the code that replaces the marker, with no "
            "explanation and no markdown fences.\n\n"
            f"{code_block}"
        )
        return ("chat", [{"role": "user", "content": user}])

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_instances(path: Path) -> list:
    instances = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                instances.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"Bad JSON on line {line_no} of {path}: {e}")
    return instances


def load_completed(path: Path) -> set:
    """Return the set of (instance_id, mode) pairs already in the output."""
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["instance_id"], rec["mode"]))
            except (json.JSONDecodeError, KeyError):
                # A partial last line from a killed job is expected; skip it.
                continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=MODELS.keys())
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES,
                    help="Which prompting modes to run (default: both)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N instances (smoke testing)")
    ap.add_argument("--max-new-tokens", type=int,
                    default=GENERATION_CONFIG["max_new_tokens"])
    args = ap.parse_args()

    model_name = MODELS[args.model]
    instances = load_instances(args.instances)
    if args.limit:
        instances = instances[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed(args.output)

    todo = [
        (inst, mode)
        for inst in instances
        for mode in args.modes
        if (inst["instance_id"], mode) not in done
    ]
    total = len(instances) * len(args.modes)
    print(f"[run_inference] model={model_name}")
    print(f"[run_inference] instances={len(instances)} modes={args.modes}")
    print(f"[run_inference] already done={len(done)}  remaining={len(todo)}/{total}")
    if not todo:
        print("[run_inference] Nothing to do. Exiting.")
        return

    # Import here so --help works without GPU/torch installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[run_inference] Loading tokenizer and model (bf16)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[run_inference] Model loaded in {time.time() - t0:.1f}s")

    chat_kwargs = {}
    if args.model == "qwen":
        # Qwen3 emits <think> reasoning by default; disable it so the raw
        # completion is just code.
        chat_kwargs["enable_thinking"] = False

    with args.output.open("a", encoding="utf-8") as out:
        for i, (inst, mode) in enumerate(todo, 1):
            kind, prompt = build_prompt(inst, mode)
            if kind == "chat":
                text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_kwargs,
                )
            else:
                # Completion mode: raw text, no chat template, matching
                # base-model usage per the Task Design Proposal.
                text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            t_gen = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_seconds = time.time() - t_gen

            completion = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            record = {
                "instance_id": inst["instance_id"],
                "model": args.model,
                "model_name": model_name,
                "mode": mode,
                "completion": completion,
                "max_new_tokens": args.max_new_tokens,
                "gen_seconds": round(gen_seconds, 2),
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()  # flush per record so resume never loses work

            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed * 3600
                print(f"[run_inference] {i}/{len(todo)} done "
                      f"({rate:.0f}/hr, last gen {gen_seconds:.1f}s)")

    print("[run_inference] Complete.")


if __name__ == "__main__":
    main()
