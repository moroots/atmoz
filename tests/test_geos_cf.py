"""GEOS-CF: conventions (download -> atmoz_dataset, fetch -> paths), units, caching, errors. OPeNDAP is faked; no network."""
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from atmoz.exceptions import DownloadError  # noqa: E402
from atmoz.models import geos_cf as g  # noqa: E402
from atmoz.models.geos_cf import GEOS_CF  # noqa: E402
from atmoz.resources.atmoz_dataclasses import atmoz_dataset  # noqa: E402

TIMES = pd.date_range("2024-06-01", "2024-07-31 23:00", freq="h")
LATS = [38.0, 38.25, 38.5]
LONS = [-77.0, -76.75, -76.5]


@pytest.fixture
def opendap(monkeypatch):
    """Fake the server: ``o3`` 1e-8 mol/mol everywhere (10 ppbv). Like the real one it serves a GrADS time axis
    (``days since 1-1-1`` as raw floats plus a ``minimum`` attribute) and ``lev`` of size 1, or 3 for the 36-level products."""
    calls = []

    def fake(url, engine=None, decode_times=True):
        calls.append(url)
        lev = [70.0, 71.0, 72.0] if "_v36" in url else [72.0]
        shape = (len(TIMES), len(lev), len(LATS), len(LONS))
        data = np.full(shape, 1e-8)
        data[:, :, :, :] += np.arange(len(lev))[None, :, None, None] * 1e-9   # level k adds k ppbv
        if decode_times:
            time = TIMES
        else:
            time = xr.DataArray(739_000.0 + np.arange(len(TIMES)) / 24.0, dims="time",
                                attrs={"units": "days since 1-1-1 00:00:0.0", "minimum": "00z01jun2024"})
        return xr.Dataset({"o3": (("time", "lev", "lat", "lon"), data), "zl": (("time", "lev", "lat", "lon"), data * 1e12)},
                          coords={"time": time, "lev": lev, "lat": LATS, "lon": LONS})

    monkeypatch.setattr(g.xr, "open_dataset", fake)
    return calls


def test_flaky_server_is_retried_then_succeeds(opendap, monkeypatch):
    monkeypatch.setattr(g, "_RETRY_DELAY", 0.0)
    real, state = g.xr.open_dataset, {"n": 0}

    def flaky(url, **kw):
        state["n"] += 1
        if state["n"] <= 2:
            raise OSError(-70, "NetCDF: DAP server error")
        return real(url, **kw)

    monkeypatch.setattr(g.xr, "open_dataset", flaky)
    assert len(dl().data["aqc_assim"]) == 24 and state["n"] == 3


def test_persistent_server_errors_raise_download_error_after_the_retries(opendap, monkeypatch):
    monkeypatch.setattr(g, "_RETRY_DELAY", 0.0)
    state = {"n": 0}

    def down(url, **kw):
        state["n"] += 1
        raise OSError(-70, "NetCDF: DAP server error")

    monkeypatch.setattr(g.xr, "open_dataset", down)
    with pytest.raises(DownloadError, match="DAP server error"):
        dl()
    assert state["n"] == g._RETRIES


def test_linear_grads_axis_is_rebuilt_from_its_attributes_not_the_damaged_values():
    values = 739_000.0 + np.arange(5) / 24.0
    values[2] = 146_465.3                                   # damaged in transit, as seen live on met3d
    t = xr.DataArray(values, dims="time", attrs={"minimum": "00:30z01jan2020", "grads_mapping": "linear",
                                                 "grads_step": "60mn", "grads_size": "5"})
    assert g._decode_time(t).tolist() == list(pd.date_range("2020-01-01 00:30", periods=5, freq="h"))
    daily = xr.DataArray(values, dims="time", attrs={"minimum": "00z05feb2021", "grads_step": "1dy", "grads_size": "5"})
    assert g._decode_time(daily)[-1] == pd.Timestamp("2021-02-09")


def test_damaged_time_axis_is_rejected():
    bad = xr.DataArray([739_000.0, 1e20], dims="time", attrs={"minimum": "00z01jan2024"})
    with pytest.raises(OSError, match="damaged"):
        g._decode_time(bad)


def dl(**kw):
    kw = {"silent": True, **kw}
    return GEOS_CF.download(38.2, -76.8, kw.pop("start", "2024-06-01"), kw.pop("end", "2024-06-01"),
                            variables=kw.pop("variables", ["o3"]), **kw)


def test_surface_download_is_one_dataframe_in_ppbv(opendap):
    ds = dl()
    assert isinstance(ds, atmoz_dataset) and ds.datatype == "model"
    assert list(ds.data) == ["aqc_assim"]
    df = ds.data["aqc_assim"]
    assert list(df.columns) == ["o3"] and len(df) == 24 and str(df.index.tz) == "UTC"
    assert df["o3"].iloc[0] == pytest.approx(10.0)                    # 1e-8 mol/mol -> 10 ppbv
    assert df.attrs["units"] == {"o3": "ppbv"}


def test_raw_units_and_grid_cell_metadata(opendap):
    ds = dl(units="mol/mol")
    assert ds.data["aqc_assim"]["o3"].iloc[0] == pytest.approx(1e-8)
    m = ds.metadata
    assert (m["lat_requested"], m["lon_requested"]) == (38.2, -76.8) and (m["lat"], m["lon"]) == (38.25, -76.75)
    assert m["endpoint"] == "geos_cf" and m["units"] == {"o3": "mol/mol"}


def test_profiles_are_time_by_level_per_variable(opendap):
    ds = dl(collection="chm")
    prof = ds.data["chm_assim_o3"]
    assert list(prof.columns) == [70.0, 71.0, 72.0] and len(prof) == 24
    assert prof.iloc[0].tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert prof.attrs["units"] == "ppbv"


def test_server_time_axis_is_decoded_from_the_grads_minimum(opendap):
    df = dl(start="2024-06-02", end="2024-06-02").data["aqc_assim"]
    assert df.index[0] == pd.Timestamp("2024-06-02 00:00", tz="UTC") and df.index[-1] == pd.Timestamp("2024-06-02 23:00", tz="UTC")


def test_decode_time_handles_minutes_and_rejects_unknown_axes():
    t = xr.DataArray(739_000.0 + np.array([0.0, 1.0, 2.0]) / 24.0, dims="time", attrs={"minimum": "00:30z01jan2018"})
    assert g._decode_time(t).tolist() == list(pd.date_range("2018-01-01 00:30", periods=3, freq="h"))
    with pytest.raises(ValueError, match="time axis"):
        g._decode_time(xr.DataArray([1.0], dims="time", attrs={"units": "days since 1-1-1"}))


def test_forecast_uses_the_latest_dataset_and_profile_sets_are_distinct(opendap):
    dl(mode="fcast")
    assert opendap[-1].endswith("aqc_tavg_1hr_g1440x721_v1.latest")
    dl(collection="chm")
    assert opendap[-1].endswith("chm_tavg_1hr_g1440x721_v36")
    ds = dl(collection="chm_sfc")
    assert opendap[-1].endswith("chm_tavg_1hr_g1440x721_v1") and list(ds.data) == ["chm_sfc_assim"]


def test_layer_heights_come_back_as_a_time_by_level_frame_in_native_units(opendap):
    zl = dl(collection="met3d", variables=["zl"]).data["met3d_assim_zl"]
    assert zl.attrs["units"] == "m" and list(zl.columns) == [70.0, 71.0, 72.0]


def test_request_spanning_months_is_one_request_per_month_and_sorted(opendap):
    ds = dl(start="2024-06-30", end="2024-07-02")
    assert len(opendap) == 2
    idx = ds.data["aqc_assim"].index
    assert idx.is_monotonic_increasing and idx[0].day == 30 and idx[-1].day == 2 and len(idx) == 72


def test_second_download_uses_the_cache(opendap, tmp_path):
    dl()
    assert len(opendap) == 1
    dl()
    assert len(opendap) == 1                                          # served from disk
    cache = tmp_path / "atmoz_home" / "cache" / "geos_cf"
    assert list(cache.glob("*.parquet")) and not list(cache.glob("*.part"))


def test_refresh_and_use_cache_false_go_back_to_the_server(opendap):
    dl()
    dl(refresh=True)
    assert len(opendap) == 2
    dl(use_cache=False)
    assert len(opendap) == 3


def test_sites_in_the_same_grid_cell_share_the_cache(opendap):
    GEOS_CF.download(38.20, -76.80, "2024-06-01", "2024-06-01", variables=["o3"], silent=True)
    GEOS_CF.download(38.30, -76.70, "2024-06-01", "2024-06-01", variables=["o3"], silent=True)
    assert len(opendap) == 1


def test_fetch_returns_cached_parquet_paths(opendap):
    out = GEOS_CF.fetch(38.2, -76.8, "2024-06-01", "2024-06-01", variables=["o3"], silent=True)
    (name, path), = out.items()
    assert name.endswith(".parquet") and path.exists()
    assert {"time", "lat", "lon", "o3"} <= set(pd.read_parquet(path).columns)


def test_unknown_variable_fails_all_or_nothing(opendap):
    with pytest.raises(DownloadError, match="nope"):
        dl(variables=["o3", "nope"])


def test_bad_requests_are_rejected_before_any_request(opendap):
    with pytest.raises(ValueError, match="not available"):
        dl(collection="chm", mode="fcast")
    with pytest.raises(ValueError, match="start_date"):
        dl(start="2024-06-02", end="2024-06-01")
    with pytest.raises(ValueError, match="units"):
        dl(units="ppm")
    assert opendap == []


def test_catalog_lists_collections():
    assert set(GEOS_CF.catalog) == {"aqc", "chm", "chm_sfc", "met", "met3d"} and "o3" in GEOS_CF.catalog["aqc"]["variables"]
