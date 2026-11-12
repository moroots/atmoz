from datetime import timedelta

import pytest

from conftest import FakeResponse
from atmoz.resources import cache


def test_download_streams_to_disk_and_caches(tmp_path, fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, b"x" * 5000, {"Content-Length": "5000", "ETag": '"e1"'}))
    dest = tmp_path / "f.zip"
    cache.download_to_cache("https://h/f.zip", dest, h.session(), chunk_size=1024)
    assert dest.read_bytes() == b"x" * 5000
    assert not list(tmp_path.glob("*.part"))
    cache.download_to_cache("https://h/f.zip", dest, h.session())  # cached forever (max_age=None)
    assert len(h.calls) == 1


def test_refresh_forces_download(tmp_path, fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, b"abc", {"Content-Length": "3"}))
    dest = tmp_path / "f.bin"
    cache.download_to_cache("https://h/f", dest, h.session())
    cache.download_to_cache("https://h/f", dest, h.session(), refresh=True)
    assert len(h.calls) == 2


def test_stale_file_is_revalidated_with_etag(tmp_path, fake_http):
    def handler(url, headers=None, **kw):
        if headers and headers.get("If-None-Match") == '"e1"':
            return FakeResponse(304)
        return FakeResponse(200, b"abc", {"Content-Length": "3", "ETag": '"e1"'})

    h = fake_http(handler)
    dest = tmp_path / "f.bin"
    cache.download_to_cache("https://h/f", dest, h.session())
    meta = cache._read_meta(dest)
    meta["fetched_at"] -= 3 * 86400
    cache._write_meta(dest, meta)
    cache.download_to_cache("https://h/f", dest, h.session(), max_age=timedelta(days=1))
    assert h.calls[-1][1]["headers"] == {"If-None-Match": '"e1"'}
    assert dest.read_bytes() == b"abc"
    # 304 refreshed the timestamp, so the next call does not even hit the network
    n = len(h.calls)
    cache.download_to_cache("https://h/f", dest, h.session(), max_age=timedelta(days=1))
    assert len(h.calls) == n


def test_truncated_download_never_replaces_good_file(tmp_path, fake_http):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"good")
    h = fake_http(lambda url, **kw: FakeResponse(200, b"ab", {"Content-Length": "10"}))
    with pytest.raises(OSError, match="Incomplete"):
        cache.download_to_cache("https://h/f", dest, h.session(), refresh=True)
    assert dest.read_bytes() == b"good"
    assert not list(tmp_path.glob("*.part"))


def test_http_error_leaves_nothing_behind(tmp_path, fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(404))
    dest = tmp_path / "missing.zip"
    with pytest.raises(Exception):
        cache.download_to_cache("https://h/f", dest, h.session())
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_cache_info_and_clear(isolated_home):
    d = cache.cache_dir("aqs")
    (d / "a.zip").write_bytes(b"1234")
    (d / "a.zip.meta.json").write_text("{}")
    assert cache.cache_info() == {"aqs": {"files": 1, "bytes": 4}}
    assert cache.clear_cache("aqs") >= 4
    assert cache.cache_info() == {}


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ATMOZ_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert cache.cache_dir("x") == tmp_path / "elsewhere" / "x"
