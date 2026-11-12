"""On-disk file cache shared by every atmoz data source.

Layout::

    ~/.atmoz/                  <- atmoz_home()   (override: ATMOZ_HOME)
    ~/.atmoz/cache/            <- cache_dir()    (override: ATMOZ_CACHE_DIR)
    ~/.atmoz/cache/<namespace>/...

Everything is plain files you can inspect, copy or delete by hand.
Downloads are streamed to ``<name>.part`` and atomically renamed, so an
interrupted download can never leave a corrupt file that looks complete.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import requests


def atmoz_home() -> Path:
    """Per-user atmoz folder (``$ATMOZ_HOME`` or ``~/.atmoz``). Not created here."""
    override = os.environ.get("ATMOZ_HOME")
    return Path(override).expanduser() if override else Path.home() / ".atmoz"


def cache_dir(namespace: Optional[str] = None, *, create: bool = True) -> Path:
    """Cache folder, optionally for one *namespace* (e.g. ``"aqs"``)."""
    override = os.environ.get("ATMOZ_CACHE_DIR")
    root = Path(override).expanduser() if override else atmoz_home() / "cache"
    path = root / namespace if namespace else root
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def clear_cache(namespace: Optional[str] = None) -> int:
    """Delete cached files (all, or one namespace). Returns the bytes freed."""
    root = cache_dir(namespace, create=False)
    if not root.exists():
        return 0
    freed = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root, ignore_errors=True)
    return freed


def cache_info() -> Dict[str, Dict[str, int]]:
    """Return ``{namespace: {"files": n, "bytes": size}}`` for what is cached."""
    root = cache_dir(create=False)
    info: Dict[str, Dict[str, int]] = {}
    if not root.exists():
        return info
    for ns in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in ns.rglob("*") if p.is_file() and not p.name.endswith((".meta.json", ".part"))]
        info[ns.name] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    return info


# ---------------------------------------------------------------------------
# Streaming download with validators
# ---------------------------------------------------------------------------

def _meta_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".meta.json")


def _read_meta(dest: Path) -> Dict[str, Any]:
    try:
        return json.loads(_meta_path(dest).read_text())
    except (OSError, ValueError):
        return {}


def _write_meta(dest: Path, meta: Mapping[str, Any]) -> None:
    try:
        _meta_path(dest).write_text(json.dumps(dict(meta)))
    except OSError:
        pass


def download_to_cache(
    url: str,
    dest: Union[str, os.PathLike],
    session: requests.Session,
    *,
    max_age: Optional[timedelta] = None,
    refresh: bool = False,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream *url* to *dest* on disk (never into memory) and return the path.

    Parameters
    ----------
    max_age:
        ``None`` treats an existing file as valid forever (right for archives
        that never change). Otherwise an older file is *revalidated* with the
        server's ``ETag`` / ``Last-Modified`` and only re-downloaded if it
        changed.
    refresh:
        Ignore the cache and download again.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(dest)
    headers: Dict[str, str] = {}

    if dest.exists() and not refresh:
        fetched = meta.get("fetched_at", dest.stat().st_mtime)
        if max_age is None or (time.time() - fetched) < max_age.total_seconds():
            return dest
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]

    part = dest.with_name(f"{dest.name}.{os.getpid()}-{threading.get_ident()}.part")
    try:
        with session.get(url, stream=True, headers=headers or None) as response:
            if response.status_code == 304 and dest.exists():
                meta["fetched_at"] = time.time()
                _write_meta(dest, meta)
                return dest
            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            written = 0
            with open(part, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    written += len(chunk)
            # iter_content may decode gzip, so only check identity-encoded bodies.
            if expected and not response.headers.get("Content-Encoding") and written != int(expected):
                raise OSError(f"Incomplete download of {dest.name}: got {written} of {expected} bytes.")
            os.replace(part, dest)
            _write_meta(dest, {
                "url": url.split("?", 1)[0],
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "fetched_at": time.time(),
                "bytes": written,
            })
    finally:
        part.unlink(missing_ok=True)
    return dest
