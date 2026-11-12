"""Shared fixtures: an isolated atmoz home and a fake HTTP layer (no network in tests)."""

from __future__ import annotations

import json as _json
import threading
from typing import Callable, List, Optional

import pytest
import requests


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.atmoz, cache and working directory, and no ambient secrets."""
    home = tmp_path / "atmoz_home"
    monkeypatch.setenv("ATMOZ_HOME", str(home))
    monkeypatch.delenv("ATMOZ_CACHE_DIR", raising=False)
    monkeypatch.delenv("ATMOZ_ENV_FILE", raising=False)
    monkeypatch.delenv("ATMOZ_AIRNOW_API_KEY", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return home


class FakeResponse:
    def __init__(self, status_code=200, body: bytes = b"", headers=None, json_data=None, text: Optional[str] = None):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._body = body
        self._json = json_data
        self.text = text if text is not None else (_json.dumps(json_data) if json_data is not None else body.decode("latin-1"))

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error: for url: fake")

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Records calls; ``handler(url, **kwargs)`` returns a FakeResponse or raises."""

    def __init__(self, handler: Callable[..., FakeResponse]):
        self.handler = handler
        self.calls: List[tuple] = []
        self._lock = threading.Lock()

    def get(self, url, **kwargs):
        with self._lock:
            self.calls.append((url, kwargs))
        return self.handler(url, **kwargs)


class FakeHandler:
    """Drop-in for SessionHandler."""

    def __init__(self, handler: Callable[..., FakeResponse]):
        self._session = FakeSession(handler)

    @property
    def calls(self):
        return self._session.calls

    def session(self):
        return self._session


@pytest.fixture
def fake_http():
    return FakeHandler
