from __future__ import annotations

import re
from pathlib import Path


def load_prompt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def sanitize_for_path(name: str) -> str:
    """Make a string safe to use as (part of) a filename."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)


def parse_repo_and_filename(description_path: Path) -> tuple[str, str]:
    """
    Split a description filename of the form '<repo>--<filename>.txt'
    back into (repo_name, original_filename). Uses the LAST '--' as the
    separator, in case the repo or file name itself contains '--'.
    """
    stem = description_path.stem  # drop the trailing .txt
    if "--" in stem:
        repo_name, file_name = stem.rsplit("--", 1)
    else:
        repo_name, file_name = "unknown_repo", stem
    return repo_name, file_name
