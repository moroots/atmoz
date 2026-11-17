"""HDF4 -> HDF5 conversion with the HDF Group's ``h4toh5convert``, installed only when you ask for it.

Many NASA products (older TOLNet files, some satellite and aircraft archives) are HDF4, which ``h5py`` cannot read.
The converter is *not* bundled in the wheel. It lives in the GitHub repo (``extras/h4toh5/``) and is downloaded
on request, checked against a SHA-256 pinned in this module, and installed under ``~/.atmoz/bin`` (``$ATMOZ_HOME``).

>>> from atmoz.resources import h4toh5
>>> h4toh5.install_h4toh5()                         # once; explicit, never automatic
>>> h5_path = h4toh5.prepare_h5("tolnet_file.hdf")  # HDF5 in -> same file; zip/HDF4 in -> converted .h5

Command line: ``python -m atmoz.resources.h4toh5 install`` (or ``where``).
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]

ENV_VAR = "ATMOZ_H4TOH5"
REPO = "moroots/atmoz"
HDF_GROUP_URL = "https://support.hdfgroup.org/ftp/HDF5/tools/h4toh5/bin/"

#: What the repo carries. The hash is pinned here on purpose: a download that does not match is refused, whatever ``ref`` it came from.
_BINARIES: Dict[str, Dict[str, object]] = {
    "linux-x86_64": {
        "path": "extras/h4toh5/linux-x86_64/h4toh5convert",
        "size": 4427848,
        "sha256": "10cc85f7cc8a648150fc1b6bdfaa6406edc6ba2d6b2933b63a23e79e0befce4d",
        "name": "h4toh5convert",
    },
}

_MAGIC_HDF4 = b"\x0e\x03\x13\x01"
_MAGIC_ZIP = b"PK\x03\x04"
_MAGIC_NETCDF3 = (b"CDF\x01", b"CDF\x02")
_MAGIC_HDF5 = b"\x89HDF\r\n\x1a\n"
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def platform_tag() -> str:
    """``"linux-x86_64"``-style tag for this machine."""
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return f"{platform.system().lower()}-{machine}"


def bin_dir() -> Path:
    """Where :func:`install_h4toh5` puts the converter (``~/.atmoz/bin`` or ``$ATMOZ_HOME/bin``)."""
    from atmoz.resources.cache import atmoz_home
    return atmoz_home() / "bin"


def _installed_path(tag: Optional[str] = None) -> Path:
    info = _BINARIES.get(tag or platform_tag())
    name = info["name"] if info else ("h4toh5convert.exe" if platform.system() == "Windows" else "h4toh5convert")
    return bin_dir() / str(name)


# ---------------------------------------------------------------------------
# File format detection (no h5py needed)
# ---------------------------------------------------------------------------
def _is_hdf5(path: Path) -> bool:
    """HDF5 signature at 0 or at a power-of-two offset >= 512 (files with a user block)."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        offset = 0
        while offset + 8 <= size:
            fh.seek(offset)
            if fh.read(8) == _MAGIC_HDF5:
                return True
            offset = 512 if offset == 0 else offset * 2
    return False


def detect_format(filepath: PathLike) -> str:
    """Identify a file by its leading bytes (extensions are not trustworthy: TOLNet serves zipped HDF4 as ``.hdf``).

    Returns ``"hdf5"``, ``"hdf4"``, ``"zip"``, ``"netcdf3"`` or ``"unknown"``.
    """
    path = Path(filepath)
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head.startswith(_MAGIC_HDF4):
        return "hdf4"
    if head.startswith(_MAGIC_ZIP):
        return "zip"
    if head[:4] in _MAGIC_NETCDF3:
        return "netcdf3"
    if _is_hdf5(path):
        return "hdf5"
    return "unknown"


# ---------------------------------------------------------------------------
# Finding and installing the converter
# ---------------------------------------------------------------------------
def _install_hint() -> str:
    tag = platform_tag()
    if tag in _BINARIES:
        return ("Install it once with atmoz.resources.h4toh5.install_h4toh5() "
                "(or `python -m atmoz.resources.h4toh5 install`); it is downloaded from the atmoz GitHub repo, "
                "verified, and never fetched automatically.")
    return (f"There is no packaged build for {tag}. Get the h4h5tools binaries for your platform from {HDF_GROUP_URL} "
            f"and either set {ENV_VAR} to the executable or put it on PATH.")


def find_h4toh5(exe: Optional[PathLike] = None) -> Path:
    """Locate ``h4toh5convert``: the ``exe`` argument, ``$ATMOZ_H4TOH5``, the copy from :func:`install_h4toh5`, then ``PATH``.

    Never downloads anything. Raises ``FileNotFoundError`` with the one-line fix if it is missing.
    """
    candidates: List[Path] = []
    if exe is not None:
        candidates.append(Path(exe))
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    candidates.append(_installed_path())
    for name in ("h4toh5convert", "h4toh5"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"Could not find the h4toh5convert executable. {_install_hint()}")


def _check_ref(ref: str) -> str:
    if not _REF.fullmatch(ref) or ".." in ref or ref.endswith("/"):
        raise ValueError(f"Not a valid git ref: {ref!r} (use a branch or tag such as 'main' or 'v0.1.1').")
    return ref


def install_h4toh5(ref: str = "main", *, force: bool = False, session=None) -> Path:
    """Download the HDF Group converter from the atmoz GitHub repo and install it. Explicit only; returns its path.

    The file is fetched from ``extras/h4toh5/`` at git ``ref`` over HTTPS, checked against the SHA-256 pinned in this
    package, and only then made executable and moved into :func:`bin_dir`. A mismatch raises and installs nothing.
    Nothing is run before it is verified. Already installed and valid -> returned without a download (``force`` redoes it).
    """
    tag = platform_tag()
    info = _BINARIES.get(tag)
    if info is None:
        raise RuntimeError(_install_hint())
    dest = _installed_path(tag)
    expected = str(info["sha256"])
    if dest.is_file() and not force and _sha256(dest) == expected:
        return dest

    import requests
    url = f"https://raw.githubusercontent.com/{REPO}/{_check_ref(ref)}/{info['path']}"
    own_session = session is None
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="h4toh5.", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as fh, session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
                fh.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(
                f"Refusing to install: the file at {url} has SHA-256 {digest.hexdigest()[:16]}..., expected {expected[:16]}... "
                "(wrong ref, or the repo copy was changed). Nothing was installed.")
        tmp.chmod(0o755)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
        if own_session:
            session.close()
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def h4_to_h5(src: PathLike, dest: Optional[PathLike] = None, exe: Optional[PathLike] = None,
             overwrite: bool = False) -> Path:
    """Convert one HDF4 file to HDF5 and return the output path (default: next to ``src`` with ``.h5``)."""
    src = Path(src)
    dest = Path(dest) if dest is not None else src.with_suffix(".h5")
    if detect_format(src) != "hdf4":
        raise ValueError(f"{src.name} is not an HDF4 file (detected {detect_format(src)!r}).")
    if dest.exists() and not overwrite:
        return dest
    tool = find_h4toh5(exe)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(tool), str(src), str(dest)], capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file() or not _is_hdf5(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"h4toh5convert failed for {src.name} (exit {result.returncode}): "
                           f"{(result.stderr or result.stdout).strip()}")
    return dest


def convert_hdf4_to_hdf5(h4_dir: PathLike, h5_dir: PathLike, exe: Optional[PathLike] = None) -> List[Path]:
    """Convert every HDF4 file (``*.hdf``) in ``h4_dir`` into ``h5_dir``. Returns the new paths."""
    h4_dir, h5_dir = Path(h4_dir), Path(h5_dir)
    return [h4_to_h5(f, h5_dir / (f.stem + ".h5"), exe=exe)
            for f in sorted(h4_dir.glob("*.hdf")) if detect_format(f) == "hdf4"]


def prepare_h5(filepath: PathLike, dest_dir: Optional[PathLike] = None, exe: Optional[PathLike] = None,
               overwrite: bool = False) -> Path:
    """Turn a downloaded file into a readable HDF5 file and return its path (the input is never modified).

    HDF5 in -> returned unchanged. HDF4 in -> converted to ``<dest_dir>/<stem>.h5``. Zip in -> the HDF4/HDF5 member
    is extracted (to a name chosen here, never the archive's own path) and handled the same way.
    """
    filepath = Path(filepath)
    dest_dir = Path(dest_dir) if dest_dir is not None else filepath.parent
    out = dest_dir / (filepath.stem + ".h5")
    fmt = detect_format(filepath)
    if fmt == "hdf5":
        return filepath
    if fmt == "hdf4":
        return h4_to_h5(filepath, out, exe=exe, overwrite=overwrite)
    if fmt == "zip":
        with zipfile.ZipFile(filepath) as zf, tempfile.TemporaryDirectory() as tmp:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                inner = Path(tmp) / "inner.bin"
                inner.write_bytes(zf.read(member))
                inner_fmt = detect_format(inner)
                if inner_fmt == "hdf5":
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if overwrite or not out.exists():
                        shutil.copyfile(inner, out)
                    return out
                if inner_fmt == "hdf4":
                    return h4_to_h5(inner, out, exe=exe, overwrite=overwrite)
        raise ValueError(f"{filepath.name} is a zip archive with no HDF4/HDF5 file inside.")
    raise ValueError(f"{filepath.name}: unsupported format {fmt!r} (expected HDF5, HDF4 or a zip of one).")


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["install"]:
        print(install_h4toh5(*args[1:2]))
        return 0
    if args[:1] == ["where"]:
        try:
            print(find_h4toh5())
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    print("usage: python -m atmoz.resources.h4toh5 install [ref] | where", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
