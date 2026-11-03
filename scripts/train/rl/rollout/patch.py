#!/usr/bin/env python3
"""Check or apply the pinned vLLM patches in the active UV environment."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess


def patch_state(root, patch, strip):
    command = ['patch', '--batch', '--fuzz=0', '--dry-run', f'-p{strip}', '-d', str(root), '-i', str(patch)]
    if subprocess.run(command + ['--forward'], capture_output=True).returncode == 0:
        return 'missing'
    if subprocess.run(command + ['--reverse'], capture_output=True).returncode == 0:
        return 'applied'
    raise ValueError(f'{patch.name} does not match installed sources; no files changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply missing patches to the active environment')
    args = parser.parse_args()
    if importlib.metadata.version('vllm') != '0.30.0':
        raise ValueError('These patches require vLLM 0.30.0')
    package = Path(importlib.util.find_spec('vllm').origin).parent
    directory = Path(__file__).parent / 'patches'
    jobs = [(package.parent, directory / 'vllm-video-placeholder-mask.patch', 1),
            (package.parent, directory / 'vllm-task-thinking.patch', 1),
            (package, directory / 'gemma-video-budget.patch', 1),
            (package, directory / 'qwen-video-processor-kwargs.patch', 1)]
    states = [(root, patch, strip, patch_state(root, patch, strip)) for root, patch, strip in jobs]
    for root, patch, strip, state in states:
        if state == 'missing' and args.apply:
            subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', f'-p{strip}',
                            '-d', str(root), '-i', str(patch)], check=True)
            state = patch_state(root, patch, strip)
        print(f'{patch.name}: {state}')
    if not args.apply and any(state == 'missing' for _, _, _, state in states):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
