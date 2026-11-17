"""The lazy HDF5 reader (atmoz.resources.hdf) and its HDF4 hook, on tiny synthetic files."""

import copy
import io
import sys
import zipfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from atmoz.lidar.tolnet import TOLNET  # noqa: E402
from atmoz.resources import hdf  # noqa: E402

HDF4 = b"\x0e\x03\x13\x01" + b"\0" * 64
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as the fake converter")


@pytest.fixture
def h5file(tmp_path):
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.attrs["DATA_LOCATION"] = np.bytes_("Hampton, VA")
        f.attrs["PI"] = "someone"
        g = f.create_group("DATA")
        d = g.create_dataset("O3", data=np.array([[1, 2, -999], [4, -999, 6]], dtype="i4"))
        d.attrs["VAR_FILL_VALUE"] = np.array([-999], dtype="i4")
        d.attrs["VAR_UNITS"] = np.bytes_("ppbv")
        g.create_dataset("scalar", data=np.float64(2.5))
        g.create_dataset("alt", data=np.arange(3.0))
    return path


def test_open_h5_is_lazy_and_a_context_manager(h5file):
    with hdf.open_h5(h5file) as f:
        assert isinstance(f, hdf.H5File) and f.filepath.name == "t.h5"
        assert "DATA" in f and "nope" not in f
        node = f.DATA
        assert isinstance(node, hdf.H5Node) and node.keys() == ["O3", "alt", "scalar"]
        assert isinstance(node["O3"], hdf.H5Dataset)
        assert f.attr("DATA_LOCATION") == "Hampton, VA"                  # bytes decoded
        assert f.attrs_dict() == {"DATA_LOCATION": "Hampton, VA", "PI": "someone"}
        assert f.attr("missing", "dflt") == "dflt"
    with pytest.raises(Exception):                                       # closed on exit
        f.DATA["O3"][()]


def test_dataset_array_masks_the_files_own_fill_value(h5file):
    with hdf.open_h5(h5file) as f:
        o3 = f.DATA["O3"]
        assert o3.shape == (2, 3) and o3.attr("VAR_UNITS") == "ppbv"
        out = o3.array(fill_value="auto")
        assert out.dtype.kind == "f" and np.isnan(out).sum() == 2 and out[0, 1] == 2
        assert np.array_equal(o3.array(), [[1, 2, -999], [4, -999, 6]])              # untouched without a fill value
        assert np.isnan(o3.array(fill_value=4)).sum() == 1
        assert o3.array(key=(0, slice(0, 2))).tolist() == [1, 2]                    # a window, not everything


def test_dataset_behaves_like_numpy_and_float(h5file):
    with hdf.open_h5(h5file) as f:
        alt = f.DATA["alt"]
        assert np.allclose(alt * 2, [0, 2, 4]) and np.allclose(1 + alt, [1, 2, 3]) and np.allclose(-alt, [0, -1, -2])
        assert np.allclose(np.asarray(alt), [0, 1, 2]) and len(alt) == 3
        assert float(f.DATA["scalar"]) == 2.5
        with pytest.raises(TypeError, match="size-1"):
            float(alt)


def test_missing_child_and_copying_do_not_recurse(h5file):
    with hdf.open_h5(h5file) as f:
        with pytest.raises(AttributeError, match="not found"):
            f.nothing_here
        with pytest.raises(AttributeError):
            hdf.H5Node.__new__(hdf.H5Node).anything                        # no _obj yet: must not loop forever
        node = f.DATA
        assert copy.copy(node)._obj is node._obj


def test_open_h5_accepts_a_file_like_stream(h5file):
    with open(h5file, "rb") as fh:
        with hdf.open_h5(fh) as f:
            assert f.DATA["alt"][()].tolist() == [0, 1, 2]
    with hdf.open_h5(io.BytesIO(h5file.read_bytes())) as f:
        assert "DATA" in f


def test_open_h5_points_hdf4_and_zip_to_the_converter(tmp_path):
    (tmp_path / "old.hdf").write_bytes(HDF4)
    with pytest.raises(ValueError, match="prepare_h5"):
        hdf.open_h5(tmp_path / "old.hdf")
    (tmp_path / "junk.h5").write_bytes(b"not hdf")
    with pytest.raises(ValueError, match="right file"):
        hdf.open_h5(tmp_path / "junk.h5")


def test_read_h5_and_read_h5_lazy_keep_their_shapes(h5file):
    data, name = hdf.read_h5(h5file)
    assert name == "t.h5" and data["_attrs"]["DATA_LOCATION"] == "Hampton, VA"
    assert data["DATA"]["alt"].tolist() == [0, 1, 2]
    node, name = hdf.read_h5_lazy(h5file)
    assert name == "t.h5" and "DATA" in node
    node.close()


def test_open_hdf_returns_name_to_node(h5file):
    nodes = hdf.open_hdf(h5file)
    assert list(nodes) == ["t.h5"]
    nodes["t.h5"].close()


@pytest.fixture
def fake_exe(tmp_path, monkeypatch):
    exe = tmp_path / "fake_h4toh5"
    exe.write_text("#!/bin/sh\ncp \"$FAKE_H5\" \"$2\"\n")
    exe.chmod(0o755)
    monkeypatch.setenv("ATMOZ_H4TOH5", str(exe))
    return exe


@posix_only
def test_convert_turns_hdf4_and_zipped_files_into_openable_hdf5(tmp_path, h5file, fake_exe, monkeypatch):
    monkeypatch.setenv("FAKE_H5", str(h5file))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.hdf").write_bytes(HDF4)
    with zipfile.ZipFile(raw / "b.hdf", "w") as zf:                        # TOLNet serves zipped HDF4 as .hdf
        zf.writestr("inner.hdf", HDF4)
    nodes = hdf.open_hdf([raw / "a.hdf", raw / "b.hdf"], convert=True)
    assert sorted(nodes) == ["a.h5", "b.h5"]
    assert (raw / "a.hdf").read_bytes() == HDF4                              # the raw download is kept
    for node in nodes.values():
        assert "DATA" in node
        node.close()
    with pytest.raises(ValueError, match="prepare_h5"):                      # without convert it still refuses, helpfully
        hdf.open_hdf(raw / "a.hdf")


def test_convert_without_the_converter_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("ATMOZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ATMOZ_H4TOH5", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    (tmp_path / "a.hdf").write_bytes(HDF4)
    with pytest.raises(FileNotFoundError, match="install_h4toh5"):
        hdf.open_hdf(tmp_path / "a.hdf", convert=True)


def test_hdf5_passes_through_convert_untouched(h5file):
    nodes = hdf.open_hdf(h5file, convert=True)                               # no converter needed for HDF5
    assert list(nodes) == ["t.h5"]
    nodes["t.h5"].close()


@posix_only
def test_tolnet_to_hdf5_and_open_hdf_convert(tmp_path, h5file, fake_exe, monkeypatch):
    monkeypatch.setenv("FAKE_H5", str(h5file))
    (tmp_path / "x.hdf").write_bytes(HDF4)
    out = TOLNET._to_hdf5({"x.hdf": tmp_path / "x.hdf", "t.h5": h5file})
    assert out["x.hdf"].name == "x.h5" and out["t.h5"] == h5file             # HDF5 is returned unchanged
    nodes = TOLNET.open_hdf([tmp_path / "x.hdf"], convert=True)
    assert list(nodes) == ["x.h5"]
    nodes["x.h5"].close()
