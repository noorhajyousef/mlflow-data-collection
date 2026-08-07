"""
generate_descriptions.py — Step 1: code -> natural-language description.

Reads .py files recursively from --input-dir, sends each to the model,
and saves one .txt description per file. Tracks progress so an
interrupted run can be resumed without repeating finished files.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from model_loader import ModelRunner, GenerationParams
from progress_tracker import ProgressTracker
from utils import load_prompt, sanitize_for_path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE.parent / "prompts"


def truncate_for_prompt(runner: ModelRunner, system_prompt: str, user_template: str, code: str) -> str:
    """Rough char-based pre-truncation before the model-level token truncation
    in ModelRunner.generate(), to avoid tokenizing huge files unnecessarily."""
    # ~4 chars/token heuristic; keep a generous margin, exact truncation
    # still happens (token-accurate) inside ModelRunner.generate().
    max_chars = runner.input_tokens * 4
    overhead = len(system_prompt) + len(user_template)
    budget = max(1000, max_chars - overhead)
    if len(code) > budget:
        return code[-budget:]  # keep the tail: most relevant for context
    return code


def generate_description_for_file(runner: ModelRunner, system_prompt: str, user_template: str,
                                   input_file: Path, output_dir: Path, repo_name: str) -> bool:
    code = input_file.read_text(encoding="utf-8", errors="replace")
    if not code.strip():
        logger.warning(f"Skipping empty file: {input_file}")
        return False

    code = truncate_for_prompt(runner, system_prompt, user_template, code)
    prompt = user_template.format(code=code)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    completion = runner.generate(messages, params=GenerationParams(max_new_tokens=2048))

    output_file = output_dir / f"{sanitize_for_path(repo_name)}--{sanitize_for_path(input_file.stem)}.txt"
    output_file.write_text(completion, encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--input-dir", default="dataset/original_code", type=Path)
    ap.add_argument("--output-base-dir", default="outputs/generated_descriptions", type=Path)
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = args.output_base_dir / f"{date_str}-{sanitize_for_path(args.model)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    py_files = []
    folder_mapping = {}
    for py_file in args.input_dir.rglob("*.py"):
        if py_file.is_file():
            rel = py_file.relative_to(args.input_dir)
            repo_name = rel.parts[0] if len(rel.parts) > 1 else args.input_dir.name
            py_files.append(py_file)
            folder_mapping[py_file] = repo_name

    if not py_files:
        print(f"No .py files found under {args.input_dir}")
        return

    system_prompt = load_prompt(PROMPTS_DIR / "description_system.txt")
    user_template = load_prompt(PROMPTS_DIR / "description_user.txt")

    print(f"Total: {len(py_files)} Python files to process")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")

    runner = ModelRunner(args.model)
    tracker = ProgressTracker(output_dir / "status.json")

    for i, py_file in enumerate(py_files, 1):
        repo_name = folder_mapping[py_file]
        key = f"{repo_name}/{py_file.name}"
        if tracker.is_done(key):
            print(f"Skip (already done): {key}")
            continue
        print(f"[{i}/{len(py_files)}] {key}")
        try:
            done = generate_description_for_file(runner, system_prompt, user_template, py_file, output_dir, repo_name)
            tracker.mark(key, "completed" if done else "skipped")
        except Exception as exc:
            logger.error(f"Failed on {py_file}: {exc}")
            tracker.mark(key, "failed", error=str(exc))

    print(f"Done. Summary: {tracker.summary()}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
