"""Repository resource paths, independent of the working directory."""

from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAIN_ROOT.parents[1]
CONFIG_DIR = TRAIN_ROOT / "configs"
PROMPT_DIR = TRAIN_ROOT / "prompts"

ANSWER_ONLY_PROMPT_DIR = PROMPT_DIR / "answer_only"
