"""Publish a verified run without replacing existing cloud objects."""

import hashlib
import io
import json
import os
from pathlib import Path

from scripts.train.storage.cloud_models import SwiftSource, digest_file, run_destination
from scripts.train.shared.data import read_json
from scripts.train.storage.export_model import verify_export


class CloudWriter:
    def __init__(self):
        endpoint = os.environ.get("AIRLAB_S3_ENDPOINT_URL")
        if endpoint and os.environ.get("AIRLAB_APPLICATION_CREDENTIAL_FILE"):
            raise ValueError(
                "Choose OpenStack application credentials or an S3 endpoint, not both"
            )
        self.s3 = None
        self.swift = None
        if endpoint:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            access = os.environ.get("AWS_ACCESS_KEY_ID")
            secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if bool(access) != bool(secret):
                raise ValueError("Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            self.s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                region_name=os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
                config=Config(
                    signature_version="s3v4" if access else UNSIGNED,
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3},
                ),
            )
        else:
            self.swift = SwiftSource()

    def head(self, container, key):
        if self.s3:
            from botocore.exceptions import ClientError

            try:
                result = self.s3.head_object(Bucket=container, Key=key)
            except ClientError as error:
                if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    return None
                raise
            return {
                "bytes": result["ContentLength"],
                "sha256": result.get("Metadata", {}).get("sha256"),
            }
        response = self.swift.session.head(
            self.swift.object_url(container, key), timeout=(30, 60)
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return {
            "bytes": int(response.headers["Content-Length"]),
            "sha256": response.headers.get("X-Object-Meta-Sha256"),
        }

    def put_new(self, container, key, stream, record):
        existing = self.head(container, key)
        if existing is not None:
            if existing != record:
                raise ValueError(
                    f"Cloud object already exists with different content: {key}"
                )
            return "unchanged"
        if record["bytes"] > 5_000_000_000:
            raise ValueError(
                "Object exceeds the single-upload limit; export weights with smaller shards"
            )
        if self.s3:
            self.s3.put_object(
                Bucket=container,
                Key=key,
                Body=stream,
                ContentLength=record["bytes"],
                Metadata={"sha256": record["sha256"]},
                IfNoneMatch="*",
            )
        else:
            response = self.swift.session.put(
                self.swift.object_url(container, key),
                data=stream,
                headers={
                    "If-None-Match": "*",
                    "Content-Length": str(record["bytes"]),
                    "X-Object-Meta-Sha256": record["sha256"],
                },
                timeout=(30, 600),
            )
            if response.status_code in (401, 403):
                raise ValueError(
                    "Upload is not permitted. Check the configured OpenStack application credential or S3 upload permissions."
                )
            response.raise_for_status()
        if self.head(container, key) != record:
            raise ValueError(f"Cloud upload verification failed: {key}")
        return "uploaded"


from scripts.train.shared.storage_settings import cloud_setting


def publish_run(run, *, inspect=False, writer=None):
    run = Path(run).resolve()
    verify_export(run)
    metadata = read_json(run / "run.json")
    location = metadata["run_location"]
    recorded_cloud = metadata.get('launch_config', {}).get('cloud', {})
    def setting(key):
        return recorded_cloud[key] if key in recorded_cloud else cloud_setting(key)
    container = setting("output_container")
    prefix = run_destination(
        setting("output_prefix"),
        metadata["model"]["repository"],
        location["date"],
        location["name"],
        answer_format=location.get("answer_format"), recipe=location.get("recipe"),
    )
    source_container = setting("source_container")
    source_prefix = (setting("source_prefix")).rstrip("/")
    if container == source_container and (
        prefix == source_prefix or prefix.startswith(source_prefix + "/")
    ):
        raise ValueError(
            "Output prefix must be separate from the base-weight source prefix"
        )
    directories = {"merged", "adapter", "checkpoints", "task_prompts"}
    files = {
        "run.json",
        "train_metrics.json",
        "test_results.json",
        "checkpoint_selection.json",
        "label_catalog.yaml",
        "prompts.yaml",
        "generation_policy.json",
    }
    records = {}
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run)
        if relative.parts[0] == "tracking":
            continue  # W&B sync manages its own logs and symlinks.
        if path.is_symlink():
            raise ValueError("Run must not contain symlinks")
        if path.is_file():
            if relative.parts[0] not in directories and str(relative) not in files:
                raise ValueError(f"Unexpected run file; refusing to upload: {relative}")
            records[str(relative)] = {
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            }
    manifest = {"container": container, "prefix": prefix, "files": records}
    print(
        json.dumps(
            {
                "container": container,
                "prefix": prefix,
                "files": len(records),
                "bytes": sum(x["bytes"] for x in records.values()),
            }
        ),
        flush=True,
    )
    if inspect:
        return manifest
    writer = writer or CloudWriter()
    payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    summary = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    writer.put_new(
        container, prefix + "/upload_manifest.json", io.BytesIO(payload), summary
    )
    for number, (relative, record) in enumerate(records.items(), 1):
        path = run / relative
        if (
            path.stat().st_size != record["bytes"]
            or digest_file(path) != record["sha256"]
        ):
            raise ValueError(f"Run changed during publication: {relative}")
        with path.open("rb") as stream:
            status = writer.put_new(container, prefix + "/" + relative, stream, record)
        if digest_file(path) != record["sha256"]:
            raise ValueError(f"Run changed during publication: {relative}")
        print(f"Publish {number}/{len(records)}: {status} {relative}", flush=True)
    # Consumers must require this marker; a partial upload is not a deployable run.
    writer.put_new(container, prefix + "/complete.json", io.BytesIO(payload), summary)
    return manifest
