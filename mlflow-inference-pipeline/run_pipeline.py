#!/usr/bin/env python3
"""
run_pipeline.py — single entry point for the description->code pipeline.

Usage:
  python run_pipeline.py descriptions --model Qwen/Qwen3-0.6B --input-dir dataset/original_code
  python run_pipeline.py code --model Qwen/Qwen3-0.6B --desc-input-dir outputs/generated_descriptions/<date>-<model>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["descriptions", "code"])
    args, remaining = ap.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.task == "descriptions":
        import generate_descriptions
        generate_descriptions.main()
    else:
        import generate_code
        generate_code.main()


if __name__ == "__main__":
    main()
