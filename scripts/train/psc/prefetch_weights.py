"""Read immutable safetensor shards sequentially once per node into page cache."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time


def prefetch(root):
    root = Path(root).resolve(strict=True)
    index = root / 'model.safetensors.index.json'
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else [p.name for p in root.glob('*.safetensors')]
    paths = []
    for name in names:
        path = (root / name).resolve(strict=True)
        if path.parent != root or path.suffix != '.safetensors':
            raise ValueError('Expected direct safetensors shard inside checkpoint directory')
        paths.append(path)
    if not paths:
        raise ValueError('No weight shards found')

    def read(path):
        before = path.stat()
        count = 0
        with path.open('rb') as stream:
            while block := stream.read(8 * 1024**2):
                count += len(block)
        after = path.stat()
        if count != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint changed during prefetch')
        return count

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        size = sum(pool.map(read, paths))
    result = {'shards': len(paths), 'bytes': size, 'seconds': round(time.monotonic() - started, 2)}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    prefetch(parser.parse_args().checkpoint)
