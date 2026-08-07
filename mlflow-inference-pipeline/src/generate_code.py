"""
generate_code.py — Step 2: description -> regenerated code.

Reads .txt description files from --desc-input-dir (non-recursive),
sends each to the model, and saves one regenerated .py file per
description, under <output-dir>/<repo>/<filename>.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from model_loader import ModelRunner, GenerationParams
from progress_tracker import ProgressTracker
from utils import load_prompt, parse_repo_and_filename

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE.parent / "prompts"


def strip_markdown_fences(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(l for l in lines if not l.strip().startswith("```")).strip()


def generate_code_for_description(runner: ModelRunner, system_prompt: str, user_template: str,
                                   description_file: Path, base_output_dir: Path) -> None:
    description = description_file.read_text(encoding="utf-8", errors="replace")
    if not description.strip():
        logger.warning(f"Skipping empty description: {description_file}")
        return

    prompt = user_template.format(description=description)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    completion = runner.generate(messages, params=GenerationParams(max_new_tokens=3000))
    completion = strip_markdown_fences(completion)

    repo_name, file_name = parse_repo_and_filename(description_file)
    repo_output_dir = base_output_dir / repo_name
    repo_output_dir.mkdir(parents=True, exist_ok=True)
    # description filenames come from generate_descriptions.py as
    # '<repo>--<stem>.txt'; drop the .txt and restore a .py extension.
    out_name = file_name if file_name.endswith(".py") else f"{file_name}.py"
    (repo_output_dir / out_name).write_text(completion, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--desc-input-dir", required=True, type=Path)
    ap.add_argument("--output-base-dir", default="outputs/generated_code", type=Path)
    args = ap.parse_args()

    desc_files = sorted(args.desc_input_dir.glob("*.txt"))
    if not desc_files:
        print(f"No .txt description files found in {args.desc_input_dir}")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    model_token = args.model.replace("/", "-")
    output_dir = args.output_base_dir / f"{date_str}-{model_token}"
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_prompt(PROMPTS_DIR / "code_system.txt")
    user_template = load_prompt(PROMPTS_DIR / "code_user.txt")

    print(f"Found {len(desc_files)} description files")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")

    runner = ModelRunner(args.model)
    tracker = ProgressTracker(output_dir / "status.json")

    for i, desc_file in enumerate(desc_files, 1):
        repo_name, file_name = parse_repo_and_filename(desc_file)
        key = f"{repo_name}/{file_name}"
        if tracker.is_done(key):
            print(f"Skip (already done): {key}")
            continue
        print(f"[{i}/{len(desc_files)}] {key}")
        try:
            generate_code_for_description(runner, system_prompt, user_template, desc_file, output_dir)
            tracker.mark(key, "completed")
        except Exception as exc:
            logger.error(f"Failed on {desc_file}: {exc}")
            tracker.mark(key, "failed", error=str(exc))

    print(f"Done. Summary: {tracker.summary()}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
