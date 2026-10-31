"""Authenticate Swift requests with an OpenStack application credential file."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import requests


class ApplicationCredentialAuth(requests.auth.AuthBase):
    def __init__(self, storage_url):
        self.storage_url = storage_url.rstrip("/")
        self.auth_url = os.environ.get("OS_AUTH_URL", "").rstrip("/")
        if urlsplit(self.auth_url).scheme != "https":
            raise ValueError("Set OS_AUTH_URL to the HTTPS OpenStack identity endpoint")
        if not self.auth_url.endswith("/v3"):
            self.auth_url += "/v3"
        path = Path(os.environ["AIRLAB_APPLICATION_CREDENTIAL_FILE"]).expanduser()
        self.credential = json.loads(path.read_text())
        if not all(self.credential.get(k) for k in ("id", "secret", "project_id")):
            raise ValueError(
                "Application credential file needs id, secret, and project_id"
            )
        if urlsplit(
            self.storage_url
        ).scheme != "https" or not self.storage_url.endswith(
            "/AUTH_" + self.credential["project_id"]
        ):
            raise ValueError(
                "Swift endpoint must match the application credential project"
            )
        self.token = None
        self.expires_at = datetime.min.replace(tzinfo=timezone.utc)
        self.lock = Lock()

    def __call__(self, request):
        if not request.url.startswith(self.storage_url + "/"):
            raise ValueError(
                "Refusing to send an OpenStack token to a different endpoint"
            )
        with self.lock:
            if datetime.now(timezone.utc) + timedelta(minutes=2) >= self.expires_at:
                self._refresh()
            request.headers["X-Auth-Token"] = self.token
        return request

    def _refresh(self):
        response = requests.post(
            self.auth_url + "/auth/tokens",
            json={
                "auth": {
                    "identity": {
                        "methods": ["application_credential"],
                        "application_credential": {
                            "id": self.credential["id"],
                            "secret": self.credential["secret"],
                        },
                    }
                }
            },
            timeout=(15, 30),
            allow_redirects=False,
        )
        if response.status_code != 201:
            raise RuntimeError(
                f"OpenStack authentication failed (HTTP {response.status_code})"
            )
        token = response.json()["token"]
        if token["project"]["id"] != self.credential["project_id"]:
            raise RuntimeError(
                "Authenticated project does not match the credential file"
            )
        expires_at = datetime.fromisoformat(token["expires_at"].replace("Z", "+00:00"))
        if expires_at <= datetime.now(timezone.utc) + timedelta(minutes=2):
            raise RuntimeError("OpenStack returned an expired or nearly expired token")
        self.token = response.headers["X-Subject-Token"]
        self.expires_at = expires_at
