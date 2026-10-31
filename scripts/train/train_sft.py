#!/usr/bin/env python3
"""Command entry point for train sft."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train.sft.launch import main


if __name__ == "__main__":
    raise SystemExit(main())
