"""GEOS-CF uses the shared file cache (OPeNDAP is faked; no network)."""
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from atmoz.models import geos_cf as g  # noqa: E402


@pytest.fixture
def fake_opendap(monkeypatch, tmp_path):
    monkeypatch.setenv("ATMOZ_HOME", str(tmp_path / "home"))
    calls = []
    real_open = xr.open_dataset
    t = pd.date_range("2024-06-01", periods=4, freq="h")

    def fake(path, engine=None):
        if str(path).startswith("http"):
            calls.append(str(path))
            return xr.Dataset({"o3": (("time", "lat", "lon"), np.ones((4, 3, 3)))},
                              coords={"time": t, "lat": [38.0, 38.25, 38.5], "lon": [-77.0, -76.75, -76.5]})
        return real_open(path)

    monkeypatch.setattr(g.xr, "open_dataset", fake)
    return calls


def _fetch(**kw):
    geos = g.GEOS_CF()
    geos.fetch(38.2, -76.8, "2024-06-01", "2024-06-01", variables=["o3"], **kw)
    return geos


def test_second_fetch_hits_shared_cache(fake_opendap, tmp_path):
    _fetch()
    n = len(fake_opendap)
    assert n > 0
    _fetch()
    assert len(fake_opendap) == n  # served from disk
    assert list((tmp_path / "home" / "cache" / "geos_cf").glob("geos_cf_*.nc"))
    assert not list((tmp_path / "home" / "cache" / "geos_cf").glob("*.part"))


def test_force_download_and_opt_out(fake_opendap):
    _fetch()
    n = len(fake_opendap)
    _fetch(force_download=True)
    assert len(fake_opendap) > n
    n = len(fake_opendap)
    _fetch(use_cache=False)
    assert len(fake_opendap) > n
