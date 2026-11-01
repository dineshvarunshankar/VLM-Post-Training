"""Verified AirLab model downloads and dated run destinations."""

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests
from scripts.train.shared.data import atomic_json, read_json
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from scripts.train.shared.storage_settings import cloud_setting

DEFAULT_SWIFT_URL = "https://airlab-s3.andrew.cmu.edu/swift/v1/AUTH_0601b6ebfa1a43c18f2d16a9c3364004"


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a nonempty relative cloud path")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value}")
    return value


def run_destination(prefix, repository, run_date, run_name, *, answer_format=None, recipe=None):
    date.fromisoformat(run_date)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(
            "Run name must contain only letters, digits, dots, underscores, or hyphens"
        )
    parts = [prefix, run_date, repository]
    if answer_format is not None or recipe is not None:
        if answer_format not in {'answer_only', 'cot'} or recipe not in {'sft_only', 'rl_only', 'sft_then_rl'}:
            raise ValueError('Unsupported answer format or training recipe')
        parts.extend([answer_format, recipe])
    parts.append(run_name)
    return "/".join(safe_relative(x) for x in parts)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_file(root, relative):
    path = root / safe_relative(relative)
    if root.is_symlink() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Model paths must not contain symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Model path escapes its directory")
    return path


def verify_local_source(root, model, *, check_hashes=True):
    root = Path(root)
    manifest = read_json(local_file(root, "cloud_source.json"))
    for key in ("repository", "revision"):
        if manifest.get(key) != model[key]:
            raise ValueError(f"Cloud source {key} does not match the selected model")
    if not manifest.get("files") or "config.json" not in manifest["files"]:
        raise ValueError("Cloud source manifest is incomplete")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"cloud_source.json"}:
        raise ValueError("Cloud source contains missing or unexpected files")
    for name, record in manifest["files"].items():
        path = local_file(root, name)
        stat = path.stat()
        if stat.st_size != record["bytes"]:
            raise ValueError(f"Cloud source size changed: {name}")
        if check_hashes or stat.st_mtime_ns != record.get("mtime_ns"):
            if digest_file(path) != record["sha256"]:
                raise ValueError(f"Cloud source checksum changed: {name}")
    # A local adapter config would cause Transformers to resolve a separate base model.
    if (root / "adapter_config.json").exists():
        raise ValueError("Expected complete base weights, not an adapter directory")
    return manifest


class SwiftSource:
    def __init__(self, url=None):
        self.url = (
            url or os.environ.get("AIRLAB_SWIFT_URL") or DEFAULT_SWIFT_URL
        ).rstrip("/")
        self.session = requests.Session()
        if os.environ.get("AIRLAB_APPLICATION_CREDENTIAL_FILE"):
            from scripts.train.storage.openstack_auth import ApplicationCredentialAuth

            self.session.auth = ApplicationCredentialAuth(self.url)
        retry = Retry(total=3, backoff_factor=1, allowed_methods=["HEAD", "GET"])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def object_url(self, container, key):
        if "/" in container:
            raise ValueError("Container must be one path component")
        return f"{self.url}/{quote(safe_relative(container), safe='')}/{quote(safe_relative(key), safe='/')}"

    def objects(self, container, prefix):
        prefix = safe_relative(prefix.rstrip("/")) + "/"
        marker = ""
        result = []
        while True:
            response = self.session.get(
                f"{self.url}/{quote(safe_relative(container), safe='')}/",
                params={
                    "prefix": prefix,
                    "format": "json",
                    "limit": 1000,
                    "marker": marker,
                },
                timeout=(30, 60),
            )
            response.raise_for_status()
            page = response.json()
            if not page:
                break
            for item in page:
                key = item["name"]
                if not key.startswith(prefix):
                    raise ValueError("Cloud listing escaped the requested prefix")
                if key.endswith("/"):
                    continue
                relative = safe_relative(key[len(prefix) :])
                head = self.session.head(
                    self.object_url(container, key), timeout=(30, 60)
                )
                head.raise_for_status()
                checksum = head.headers.get("X-Object-Meta-Source-Sha256", "")
                if not re.fullmatch("[0-9a-f]{64}", checksum):
                    raise ValueError(f"Missing source SHA256 metadata: {relative}")
                result.append(
                    {
                        "name": relative,
                        "key": key,
                        "bytes": int(head.headers["Content-Length"]),
                        "sha256": checksum,
                        "revision": head.headers.get("X-Object-Meta-Source-Revision"),
                    }
                )
            next_marker = page[-1]["name"]
            if next_marker <= marker:
                raise ValueError("Cloud listing did not advance")
            marker = next_marker
            if len(page) < 1000:
                break
        return result

    def download(self, container, item, path):
        if path.exists():
            if (
                path.stat().st_size == item["bytes"]
                and digest_file(path) == item["sha256"]
            ):
                return
            raise ValueError(
                f"Existing model file differs; choose a new cache directory: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        if partial.is_symlink():
            raise ValueError("Partial download must not be a symlink")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError(f"Oversized partial download: {partial}")
        if item["bytes"] == 0 and not partial.exists():
            partial.touch(exist_ok=False)
        if offset < item["bytes"]:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with self.session.get(
                self.object_url(container, item["key"]),
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if offset and (
                    response.status_code != 206
                    or not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    )
                ):
                    raise ValueError("Server did not honor the partial-download range")
                with partial.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(8 * 1024 * 1024):
                        stream.write(block)
        if (
            partial.stat().st_size != item["bytes"]
            or digest_file(partial) != item["sha256"]
        ):
            raise ValueError(f"Download verification failed: {partial}")
        partial.rename(path)


def download_model(model, weights_root, *, inspect=False, source=None):
    source = source or SwiftSource()
    container = cloud_setting("source_container")
    prefix = "/".join(
        [cloud_setting("source_prefix"), model["repository"]]
    )
    destination = (
        Path(weights_root).expanduser()
        / safe_relative(model["repository"])
        / safe_relative(model["revision"])
    )
    legacy_prefix = prefix
    prefix = legacy_prefix + "/" + safe_relative(model["revision"])
    items = source.objects(container, prefix)
    if not items:
        prefix = legacy_prefix
        items = source.objects(container, prefix)
    if not items or any(x["revision"] != model["revision"] for x in items):
        raise ValueError(
            "Cloud files are missing or their revision differs from the selected model"
        )
    if len({x["name"] for x in items}) != len(items):
        raise ValueError("Duplicate cloud object names")
    if any(x["name"] in ("cloud_source.json", "adapter_config.json") for x in items):
        raise ValueError(
            "Cloud source contains a reserved manifest or adapter configuration"
        )
    print(
        json.dumps(
            {
                "source": f"{source.url}/{container}/{prefix}/",
                "local": str(destination),
                "files": len(items),
                "bytes": sum(x["bytes"] for x in items),
            }
        ),
        flush=True,
    )
    if inspect:
        return destination
    for number, item in enumerate(items, 1):
        source.download(container, item, local_file(destination, item["name"]))
        print(f"Download verified {number}/{len(items)}: {item['name']}", flush=True)
    manifest = {
        "repository": model["repository"],
        "revision": model["revision"],
        "source": {"url": source.url, "container": container, "prefix": prefix},
        "files": {
            x["name"]: {
                "bytes": x["bytes"],
                "sha256": x["sha256"],
                "mtime_ns": (destination / x["name"]).stat().st_mtime_ns,
            }
            for x in items
        },
    }
    atomic_json(destination / "cloud_source.json", manifest)
    verify_local_source(destination, model, check_hashes=False)
    return destination
