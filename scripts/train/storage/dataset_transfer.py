"""Publish and retrieve verified prepared-dataset snapshots through Swift."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import tempfile
from urllib.parse import urlsplit

import requests
from scripts.train.shared.data import atomic_json
from scripts.train.storage.cloud_models import SwiftSource, local_file, safe_relative, digest_file

MANIFEST = 'dataset-transfer.json'
ROOTS = {'06-native_media_dataset', '07-stabilized_media_dataset', '08-train_val_test'}


def location(value):
    parsed = urlsplit(value)
    if parsed.scheme != 'swift' or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('Use swift://container/snapshot-prefix')
    container = safe_relative(parsed.netloc)
    prefix = safe_relative(parsed.path.lstrip('/').rstrip('/'))
    if '/' in container or '@' in container or ':' in container:
        raise ValueError('Invalid Swift container')
    return container, prefix


def validate_manifest(value):
    if value.get('schema_version') not in (1, 2) or not isinstance(value.get('files'), dict) or not value['files']:
        raise ValueError('Invalid dataset transfer manifest')
    for name, record in value['files'].items():
        safe_relative(name)
        if name.startswith('/') or name.split('/')[0] not in ROOTS:
            raise ValueError(f'Unexpected dataset path: {name}')
        if not isinstance(record, dict) or type(record.get('bytes')) is not int or record['bytes'] < 0:
            raise ValueError(f'Invalid file size: {name}')
        if not isinstance(record.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', record['sha256']):
            raise ValueError(f'Invalid checksum: {name}')
        if 'object' in record:
            safe_relative(record['object'])
            if record['object'] != 'objects/' + record['sha256']:
                raise ValueError(f'Invalid content object: {name}')
    names = set(value['files'])
    for name in names:
        if any(str(p) in names for p in Path(name).parents):
            raise ValueError('A file cannot also be a parent directory')
    return value


def parallel(items, operation, workers, label):
    if not 1 <= workers <= 64:
        raise ValueError('workers must be between 1 and 64')
    result = {}
    total = len(items)
    started = last = time.monotonic()
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(operation, name): name for name in items}
        try:
            for number, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                record = future.result()
                result[name] = record
                completed_bytes += record['bytes']
                now = time.monotonic()
                if number == total or now - last >= 5:
                    print(f'{label}: {number}/{total} files; {completed_bytes / 1024**3:.2f} GiB; '
                          f'checked in {now-started:.1f}s', flush=True)
                    last = now
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return result


def clients(source):
    """Give each worker a session while sharing the thread-safe token provider."""
    state = threading.local()

    def get():
        if not hasattr(state, 'client'):
            state.client = SwiftSource(source.url)
            state.client.session.auth = source.session.auth
        return state.client
    return get


def object_matches(response, record):
    return (response.status_code == 200
            and response.headers.get('Content-Length') == str(record['bytes'])
            and response.headers.get('X-Object-Meta-Source-Sha256') == record['sha256'])


def put(client, url, path, record, md5, *, rewrite=False):
    head = client.session.head(url, timeout=(30, 60), allow_redirects=False)
    if object_matches(head, record) and not rewrite:
        return
    if head.status_code != 404 and not (rewrite and object_matches(head, record)):
        head.raise_for_status()
        raise ValueError('Cloud object conflicts with this snapshot; choose a new prefix')
    for attempt in range(3):
        try:
            with path.open('rb') as stream:
                response = client.session.put(url, data=stream, headers={
                    'ETag': md5, 'X-Object-Meta-Source-Sha256': record['sha256'],
                    'Content-Type': 'application/octet-stream',
                    **({} if rewrite else {'If-None-Match': '*'}),
                }, timeout=(30, 120), allow_redirects=False)
            if response.status_code in (409, 412):
                check = client.session.head(url, timeout=(30, 60), allow_redirects=False)
                if object_matches(check, record):
                    return
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.ConnectionError(f'Upload HTTP {response.status_code}')
            if response.status_code not in (200, 201, 202, 204):
                raise ValueError(f'Upload failed (HTTP {response.status_code})')
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def upload_dataset(plan, destination, *, workers=16, source=None, inspect=False, rewrite=False):
    container, prefix = location(destination)
    source = source or SwiftSource()
    if not 1 <= workers <= 64:
        raise ValueError('workers must be between 1 and 64')
    plan.verify_unchanged()
    for name in plan.files:
        validate_manifest({'schema_version': 1, 'files': {name: {'bytes': 0, 'sha256': '0'*64}}})
    if inspect:
        print(f'Cloud preview: {len(plan.files)} files to {destination}', flush=True)
        return
    url = source.object_url(container, prefix+'/'+MANIFEST)
    previous = source.session.get(url, timeout=(30, 60), allow_redirects=False)
    if previous.status_code not in (200, 404):
        raise ValueError(f'Manifest lookup failed (HTTP {previous.status_code})')
    old = validate_manifest(previous.json()) if previous.status_code == 200 else {'files': {}}
    previous_etag = previous.headers.get('ETag')
    if previous.status_code == 200 and not previous_etag:
        previous_etag = hashlib.md5(previous.content).hexdigest()
    get_client = clients(source)

    def send(name):
        path = local_file(plan.root, name)
        before = path.stat()
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                sha.update(block)
                md5.update(block)
        record = {'bytes': before.st_size, 'sha256': sha.hexdigest()}
        client = get_client()
        existing = old['files'].get(name)
        if existing and all(existing.get(k) == record[k] for k in ('bytes', 'sha256')):
            record = dict(existing)
        elif previous.status_code == 200:
            record['object'] = 'objects/' + record['sha256']
        object_name = record.get('object', 'files/' + name)
        put(client, client.object_url(container, prefix+'/'+object_name), path, record,
            md5.hexdigest(), rewrite=rewrite)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f'Source changed during upload: {name}')
        return record

    records = parallel(plan.files, send, workers, 'Uploaded')
    plan.verify_unchanged()
    manifest = validate_manifest({'schema_version': 2, 'files': records})
    body = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    if manifest != old or previous.status_code == 404:
        response = source.session.put(url, data=body, headers={
            'Content-Type': 'application/json', 'ETag': hashlib.md5(body).hexdigest(),
            **({'If-Match': previous_etag} if previous.status_code == 200 else {'If-None-Match': '*'}),
        }, timeout=(30, 60), allow_redirects=False)
        if response.status_code not in (200, 201, 202, 204):
            raise ValueError(f'Manifest publication failed (HTTP {response.status_code}); rerun to reconcile')
    print(f'Dataset published: {destination}', flush=True)
    return manifest


def download_dataset(origin, destination, *, workers=16, source=None, inspect=False, rewrite=False):
    container, prefix = location(origin)
    source = source or SwiftSource()
    if not 1 <= workers <= 64:
        raise ValueError('workers must be between 1 and 64')
    root = Path(destination).expanduser().absolute()
    if str(root) == '/' or ':' in str(destination):
        raise ValueError('Cloud download destination must be a local dataset directory')
    response = source.session.get(source.object_url(container, prefix+'/'+MANIFEST), timeout=(30, 60))
    response.raise_for_status()
    manifest = validate_manifest(response.json())
    for name in [MANIFEST, *manifest['files']]:
        local_file(root, name)
    if inspect:
        print(f'Cloud preview: {len(manifest["files"])} files to {root}', flush=True)
        return manifest
    get_client = clients(source)

    def fetch(name):
        record = manifest['files'][name]
        path = local_file(root, name)
        item = {**record, 'key': prefix+'/'+record.get('object', 'files/'+name)}
        if not rewrite and path.exists() and path.stat().st_size == record['bytes'] and digest_file(path) == record['sha256']:
            return record
        if not path.exists() and not rewrite:
            get_client().download(container, item, path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the old file until the replacement passes size and SHA-256 checks.
            with tempfile.TemporaryDirectory(prefix='.dataset-transfer-', dir=path.parent) as staging:
                replacement = Path(staging) / 'replacement'
                get_client().download(container, item, replacement)
                replacement.replace(path)
        return record

    parallel(manifest['files'], fetch, workers, 'Verified')
    atomic_json(local_file(root, MANIFEST), manifest)
    print(f'Dataset verified: {root}', flush=True)
    return manifest
