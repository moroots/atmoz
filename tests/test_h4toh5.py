"""The HDF4 -> HDF5 converter helper: format detection, finding/installing the binary (offline, fake HTTP), conversion."""

import hashlib
import io
import stat
import sys
import zipfile
from pathlib import Path

import pytest

from atmoz.resources import h4toh5

HDF5 = b"\x89HDF\r\n\x1a\n" + b"\0" * 64
HDF4 = b"\x0e\x03\x13\x01" + b"\0" * 64
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as the fake converter")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATMOZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ATMOZ_H4TOH5", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))      # no real converter on PATH
    return tmp_path / "home"


class _Response(io.BytesIO):
    status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        while chunk := self.read(chunk_size):
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, payload):
        self.payload, self.urls = payload, []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return _Response(self.payload)

    def close(self):
        pass


@pytest.fixture
def fake_binary(monkeypatch):
    payload = b"#!/bin/sh\necho fake converter\n" * 50
    monkeypatch.setattr(h4toh5, "platform_tag", lambda: "linux-x86_64")
    monkeypatch.setitem(h4toh5._BINARIES, "linux-x86_64", {
        "path": "extras/h4toh5/linux-x86_64/h4toh5convert", "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(), "name": "h4toh5convert"})
    return payload


# --- format detection -----------------------------------------------------------------------------------------------
def test_detect_format(tmp_path):
    cases = {"a": (HDF5, "hdf5"), "b": (HDF4, "hdf4"), "c": (b"PK\x03\x04rest", "zip"),
             "d": (b"CDF\x01rest", "netcdf3"), "e": (b"hello world", "unknown")}
    for name, (data, expected) in cases.items():
        (tmp_path / name).write_bytes(data)
        assert h4toh5.detect_format(tmp_path / name) == expected, name


def test_detect_hdf5_with_user_block(tmp_path):
    (tmp_path / "ub.h5").write_bytes(b"\0" * 512 + HDF5)
    assert h4toh5.detect_format(tmp_path / "ub.h5") == "hdf5"


# --- finding --------------------------------------------------------------------------------------------------------
def test_find_raises_with_the_fix_and_downloads_nothing():
    with pytest.raises(FileNotFoundError, match="install_h4toh5"):
        h4toh5.find_h4toh5()


def test_find_unsupported_platform_points_to_hdf_group(monkeypatch):
    monkeypatch.setattr(h4toh5, "platform_tag", lambda: "darwin-arm64")
    with pytest.raises(FileNotFoundError, match="hdfgroup.org"):
        h4toh5.find_h4toh5()


@posix_only
def test_find_order_argument_env_installed(tmp_path, monkeypatch, fake_binary):
    exe_a, exe_b = tmp_path / "a", tmp_path / "b"
    for exe in (exe_a, exe_b):
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    installed = h4toh5.install_h4toh5(session=FakeSession(fake_binary))
    assert h4toh5.find_h4toh5() == installed
    monkeypatch.setenv("ATMOZ_H4TOH5", str(exe_b))
    assert h4toh5.find_h4toh5() == exe_b
    assert h4toh5.find_h4toh5(exe_a) == exe_a


# --- installing -----------------------------------------------------------------------------------------------------
@posix_only
def test_install_downloads_verifies_and_is_executable(home, fake_binary):
    session = FakeSession(fake_binary)
    path = h4toh5.install_h4toh5("v0.1.1", session=session)
    assert path == home / "bin" / "h4toh5convert"
    assert path.read_bytes() == fake_binary
    assert path.stat().st_mode & stat.S_IXUSR
    assert session.urls == ["https://raw.githubusercontent.com/moroots/atmoz/v0.1.1/extras/h4toh5/linux-x86_64/h4toh5convert"]
    assert not list(path.parent.glob("*.part"))


@pytest.mark.skipif(sys.platform == "win32", reason="posix permissions")
def test_install_is_idempotent_unless_forced(fake_binary):
    session = FakeSession(fake_binary)
    h4toh5.install_h4toh5(session=session)
    h4toh5.install_h4toh5(session=session)
    assert len(session.urls) == 1
    h4toh5.install_h4toh5(session=session, force=True)
    assert len(session.urls) == 2


def test_install_refuses_a_file_that_does_not_match_the_pinned_hash(home, fake_binary):
    with pytest.raises(RuntimeError, match="Refusing to install"):
        h4toh5.install_h4toh5(session=FakeSession(b"something else entirely"))
    assert not (home / "bin" / "h4toh5convert").exists()
    assert not list((home / "bin").glob("*"))                      # no leftover .part either


@pytest.mark.parametrize("ref", ["", "../evil", "a/../b", "-x", "main/", "a b", "x?y"])
def test_install_rejects_odd_refs(ref, fake_binary):
    session = FakeSession(fake_binary)
    with pytest.raises(ValueError, match="git ref"):
        h4toh5.install_h4toh5(ref, session=session)
    assert session.urls == []


def test_install_on_unsupported_platform_does_not_download(monkeypatch):
    monkeypatch.setattr(h4toh5, "platform_tag", lambda: "windows-amd64")
    session = FakeSession(b"x")
    with pytest.raises(RuntimeError, match="no packaged build"):
        h4toh5.install_h4toh5(session=session)
    assert session.urls == []


def test_the_binary_in_the_repo_matches_the_pinned_hash():
    repo_file = Path(__file__).resolve().parents[1] / h4toh5._BINARIES["linux-x86_64"]["path"]
    if not repo_file.is_file():
        pytest.skip("not running from a repo checkout")
    assert repo_file.stat().st_size == h4toh5._BINARIES["linux-x86_64"]["size"]
    assert h4toh5._sha256(repo_file) == h4toh5._BINARIES["linux-x86_64"]["sha256"]


# --- conversion (a fake converter script stands in for the HDF Group binary) -----------------------------------------
@pytest.fixture
def fake_exe(tmp_path):
    exe = tmp_path / "fake_h4toh5"
    exe.write_text("#!/bin/sh\nprintf '\\211HDF\\r\\n\\032\\n\\0\\0\\0\\0' > \"$2\"\n")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def failing_exe(tmp_path):
    exe = tmp_path / "bad_h4toh5"
    exe.write_text("#!/bin/sh\necho 'cannot read file' >&2\nexit 3\n")
    exe.chmod(0o755)
    return exe


@posix_only
def test_h4_to_h5_converts_and_reuses(tmp_path, fake_exe):
    src = tmp_path / "in.hdf"
    src.write_bytes(HDF4)
    out = h4toh5.h4_to_h5(src, exe=fake_exe)
    assert out == tmp_path / "in.h5" and h4toh5.detect_format(out) == "hdf5"
    out.write_bytes(b"sentinel")
    assert h4toh5.h4_to_h5(src, exe=fake_exe).read_bytes() == b"sentinel"          # not redone without overwrite
    assert h4toh5.detect_format(h4toh5.h4_to_h5(src, exe=fake_exe, overwrite=True)) == "hdf5"


@posix_only
def test_h4_to_h5_failure_leaves_nothing(tmp_path, failing_exe):
    src = tmp_path / "in.hdf"
    src.write_bytes(HDF4)
    with pytest.raises(RuntimeError, match="cannot read file"):
        h4toh5.h4_to_h5(src, exe=failing_exe)
    assert not (tmp_path / "in.h5").exists()


def test_h4_to_h5_rejects_non_hdf4(tmp_path):
    src = tmp_path / "x.h5"
    src.write_bytes(HDF5)
    with pytest.raises(ValueError, match="not an HDF4"):
        h4toh5.h4_to_h5(src)


@posix_only
def test_prepare_h5_handles_hdf5_hdf4_and_zip(tmp_path, fake_exe):
    h5 = tmp_path / "a.h5"
    h5.write_bytes(HDF5)
    assert h4toh5.prepare_h5(h5) == h5                                              # untouched, no converter needed
    h4 = tmp_path / "b.hdf"
    h4.write_bytes(HDF4)
    assert h4toh5.detect_format(h4toh5.prepare_h5(h4, tmp_path / "out", exe=fake_exe)) == "hdf5"
    zipped = tmp_path / "c.hdf"                                                      # zipped HDF4 served under .hdf
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("../../evil/name.bin", HDF4)                                      # member path is never trusted
    out = h4toh5.prepare_h5(zipped, tmp_path / "out", exe=fake_exe)
    assert out == tmp_path / "out" / "c.h5" and h4toh5.detect_format(out) == "hdf5"
    assert not (tmp_path / "evil").exists()


def test_prepare_h5_rejects_zip_without_hdf_and_unknown(tmp_path):
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", b"hello")
    with pytest.raises(ValueError, match="no HDF4/HDF5"):
        h4toh5.prepare_h5(z)
    junk = tmp_path / "e.bin"
    junk.write_bytes(b"not hdf at all")
    with pytest.raises(ValueError, match="unsupported format"):
        h4toh5.prepare_h5(junk)


@posix_only
def test_convert_hdf4_to_hdf5_batch(tmp_path, fake_exe):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.hdf").write_bytes(HDF4)
    (tmp_path / "in" / "b.hdf").write_bytes(HDF4)
    (tmp_path / "in" / "zipped.hdf").write_bytes(b"PK\x03\x04junk")                # not HDF4: skipped
    out = h4toh5.convert_hdf4_to_hdf5(tmp_path / "in", tmp_path / "out", exe=fake_exe)
    assert [p.name for p in out] == ["a.h5", "b.h5"]


def test_cli_where_and_usage(capsys):
    assert h4toh5.main(["where"]) == 1
    assert "install_h4toh5" in capsys.readouterr().err
    assert h4toh5.main([]) == 2
