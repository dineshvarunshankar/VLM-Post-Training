#!/usr/bin/env python3
"""Download base weights from AirLab or publish a completed fine-tuning run."""


import argparse
import os
from scripts.train.shared.storage_settings import local_weights_directory, local_dataset_directory
from pathlib import Path

from scripts.train.shared.paths import REPO_ROOT, CONFIG_DIR
from scripts.train.storage.cloud_models import download_model
from scripts.train.shared.data import read_yaml
from dotenv import load_dotenv


def main():
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--model", required=True)
    download.add_argument(
        "--models", type=Path, default=CONFIG_DIR / "models.yaml"
    )
    download.add_argument(
        "--weights-root", type=Path, default=local_weights_directory("base_weights")
    )
    download.add_argument("--inspect", action="store_true")
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--inspect", action="store_true")
    export = subparsers.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument(
        "--weights-root", type=Path, default=local_weights_directory("base_weights")
    )
    args = parser.parse_args()
    if args.command == "download":
        if args.weights_root is None:
            parser.error("Set LOCAL_WEIGHTS_ROOT or pass --weights-root")
        model = read_yaml(args.models)["models"][args.model]
        download_model(model, args.weights_root, inspect=args.inspect)
    elif args.command == "export":
        from scripts.train.storage.export_model import export_saved_run

        export_saved_run(args.run_dir, args.weights_root)
    else:
        from scripts.train.storage.publish_model import publish_run

        publish_run(args.run_dir, inspect=args.inspect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
