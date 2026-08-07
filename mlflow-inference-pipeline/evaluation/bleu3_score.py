"""
bleu3_score.py — per-file BLEU-3 between generated code and original code.

Matches files by relative path (<repo>/<filename>) between
--generated-dir and --original-dir, tokenizes on whitespace/punctuation,
and computes a BLEU-3 score (uniform 1/3 weights on 1-, 2-, 3-grams)
per file. Writes a per-file CSV plus prints summary statistics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(code: str) -> list[str]:
    return TOKEN_RE.findall(code)


def is_valid_python(code: str) -> bool:
    """
    Companion metric to BLEU-3: does the generated file even parse as
    valid Python? BLEU-3 measures textual similarity to the original,
    but a file can score low on BLEU-3 while still being perfectly
    valid, semantically-equivalent code (different variable names,
    reordered statements, etc.) -- and vice versa, a high-BLEU-3 file
    can still be syntactically broken. This flags that separately.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def bleu3(reference: str, candidate: str) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not cand_tokens or not ref_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu(
        [ref_tokens], cand_tokens,
        weights=(1 / 3, 1 / 3, 1 / 3),
        smoothing_function=smoothing,
    )


def find_matching_original(generated_file: Path, generated_dir: Path, original_dir: Path) -> Path | None:
    rel = generated_file.relative_to(generated_dir)
    candidate = original_dir / rel
    if candidate.exists():
        return candidate
    # Fallback: match by filename only, in case folder structure differs slightly
    matches = list(original_dir.rglob(generated_file.name))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", required=True, type=Path)
    ap.add_argument("--original-dir", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for gen_file in sorted(args.generated_dir.rglob("*.py")):
        orig_file = find_matching_original(gen_file, args.generated_dir, args.original_dir)
        if orig_file is None:
            print(f"WARN: no original match for {gen_file}")
            continue
        generated_code = gen_file.read_text(encoding="utf-8", errors="replace")
        original_code = orig_file.read_text(encoding="utf-8", errors="replace")
        score = bleu3(original_code, generated_code)
        rows.append({
            "File": str(gen_file.relative_to(args.generated_dir)),
            "BLEU-3": round(score, 4),
            "Valid Python": is_valid_python(generated_code),
        })

    if not rows:
        print("No matched file pairs found. Nothing to score.")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["File", "BLEU-3", "Valid Python"])
        writer.writeheader()
        writer.writerows(rows)

    scores = [r["BLEU-3"] for r in rows]
    n_valid = sum(1 for r in rows if r["Valid Python"])
    print(f"Scored {len(scores)} files -> {args.output_csv}")
    print(f"  average: {statistics.mean(scores):.4f}")
    print(f"  median:  {statistics.median(scores):.4f}")
    print(f"  min:     {min(scores):.4f}")
    print(f"  max:     {max(scores):.4f}")
    print(f"  stddev:  {statistics.pstdev(scores):.4f}" if len(scores) > 1 else "  stddev:  n/a (only 1 file)")
    print(f"  valid Python: {n_valid}/{len(scores)} ({n_valid / len(scores) * 100:.1f}%)")


if __name__ == "__main__":
    main()
