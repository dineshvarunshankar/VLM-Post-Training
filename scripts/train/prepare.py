#!/usr/bin/env python3
"""Command entry point for prepare."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    if sys.argv[1:2] == ['splits']:
        from scripts.train.splits.cli import main as split_main
        return split_main(sys.argv[2:])
    from scripts.train.shared.prepare import main as gt_main
    return gt_main()


if __name__ == "__main__":
    raise SystemExit(main())
