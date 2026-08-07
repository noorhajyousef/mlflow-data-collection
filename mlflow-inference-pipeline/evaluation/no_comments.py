"""
no_comments.py — strip comments and docstrings from Python files.

Used as an optional preprocessing step before BLEU-3 scoring: comments
and docstrings are free-form natural language that a model can phrase
very differently from the original while producing functionally
identical code. Scoring on stripped code isolates BLEU-3 to the actual
code structure/logic rather than penalizing harmless comment wording.

Usage:
  python no_comments.py --input-dir dataset/original_code --output-dir dataset/original_code_stripped
  python no_comments.py --input-dir outputs/generated_code/<date>-<model> --output-dir outputs/generated_code_stripped/<date>-<model>
"""

from __future__ import annotations

import argparse
import ast
import io
import tokenize
from pathlib import Path


def strip_comments_and_docstrings(source: str) -> str:
    # Pass 1: remove comment tokens (tokenize is whitespace/structure-safe).
    result = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_string, start, end, line in tokens:
            if tok_type == tokenize.COMMENT:
                continue
            result.append((tok_type, tok_string, start, end, line))
        source_no_comments = tokenize.untokenize(
            [(t[0], t[1]) for t in result]
        )
    except tokenize.TokenizeError:
        # Malformed/truncated source (common with model output); fall back
        # to returning it unchanged rather than crashing the whole run.
        return source

    # Pass 2: remove docstrings via the AST (first statement of a module,
    # class, or function body if it's a bare string expression).
    try:
        tree = ast.parse(source_no_comments)
    except SyntaxError:
        return source_no_comments

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(getattr(node.body[0], "value", None), (ast.Constant,))
                    and isinstance(node.body[0].value.value, str)):
                node.body[0] = ast.Pass()
                ast.fix_missing_locations(node.body[0])

    try:
        return ast.unparse(tree)
    except Exception:
        # ast.unparse reformats code; if anything goes wrong, prefer the
        # comment-stripped-only version over crashing.
        return source_no_comments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    py_files = list(args.input_dir.rglob("*.py"))
    if not py_files:
        print(f"No .py files found under {args.input_dir}")
        return

    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        stripped = strip_comments_and_docstrings(source)
        rel = py_file.relative_to(args.input_dir)
        out_path = args.output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(stripped, encoding="utf-8")

    print(f"Stripped {len(py_files)} files -> {args.output_dir}")


if __name__ == "__main__":
    main()
