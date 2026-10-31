"""Storage defaults shared across machines; credentials stay in the environment."""

import os
from pathlib import Path

import yaml
from scripts.train.shared.paths import CONFIG_DIR


def storage_settings():
    return yaml.safe_load((CONFIG_DIR / 'storage.yaml').read_text())


def cloud_setting(name):
    return os.environ.get('AIRLAB_' + name.upper()) or storage_settings()['cloud'][name]


def local_weights_directory(kind):
    root = os.environ.get('LOCAL_WEIGHTS_ROOT')
    if root:
        return Path(root).expanduser() / storage_settings()['local'][kind]
    legacy = 'AIRLAB_WEIGHTS_DIR' if kind == 'base_weights' else 'TRAIN_OUTPUT_ROOT'
    value = os.environ.get(legacy)
    return Path(value).expanduser() if value else None


def local_dataset_directory():
    value = os.environ.get('LOCAL_DATASET_ROOT') or os.environ.get('TRAIN_MEDIA_ROOT')
    return Path(value).expanduser() if value else None


def annotation_dataset_directory():
    value = os.environ.get('AIRLAB_DATASET_ROOT') or os.environ.get('TRIAGE_DATASET_ROOT')
    return Path(value).expanduser() if value else None
