"""UCN (Unified Ceilometer Network): offline, HTTP is faked and the NetCDF files are built in memory."""
import json
import time

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

import atmoz
from conftest import FakeResponse
from atmoz.dataclasses import AtmozDataset, AtmozFrame
from atmoz.exceptions import DownloadError, NoDataError, RequestTooLargeError
from atmoz.lidar import UCN
from atmoz.lidar import ucn as ucn_mod

ALT = np.array([10.0, 20.0, 30.0, 40.0])


def l2_nc(tmp_path, day="2026-11-01"):
    t = pd.date_range(day, periods=6, freq="10min")
    ds = xr.Dataset(
        {
            "attenuated_backscatter": (("altitude", "time"), np.arange(24, dtype="float64").reshape(4, 6), {"units": "km^-1.sr^-1"}),
            "nsl_ml_height": ("time", np.array([100.0, 120.0, np.nan, 140.0, 150.0, 160.0]), {"units": "m"}),
            "cloud_layers": (("cloud_vars", "time"), np.arange(18, dtype="float64").reshape(3, 6)),
            "jump_primary": (("retrieval_layer", "time"), np.arange(12, dtype="float64").reshape(2, 6)),
            "latitude": ((), 38.92), "longitude": ((), -77.01),
        },
        coords={"time": t, "altitude": ALT, "retrieval_layer": ["NSL", "RL"]},
        attrs={"algorithm_version": "V1", "history": "made in a test"},
    )
    path = tmp_path / f"l2_{day}.nc"
    ds.to_netcdf(path)
    return path.read_bytes()


def l3_nc(tmp_path):
    t = pd.date_range("2026-11-01", periods=6, freq="10min")
    h = pd.date_range("2026-11-01 00:30", periods=3, freq="1h")
    ds = xr.Dataset(
        {
            "attenuated_backscatter": (("altitude", "time"), np.zeros((4, 6)), {"units": "km^-1.sr^-1"}),
            "nsl_ml_hourly_mean": ("hourly_time", np.array([300.0, 310.0, 320.0]), {"units": "m"}),
            "nsl_ml_hourly_count": ("hourly_time", np.array([6, 5, 6])),
            "latitude": ((), 38.92), "longitude": ((), -77.01),
        },
        coords={"time": t, "altitude": ALT, "hourly_time": h},
    )
    path = tmp_path / "l3.nc"
    ds.to_netcdf(path)
    return path.read_bytes()


def server(tmp_path, *, missing=(), listed=None):
    """Fake UCN. *missing* are 'YYYY-MM-DD' days with no file; *listed* is the viewer's availability list."""
    files = {"3": l3_nc(tmp_path)}
    l2 = {}

    def l2_for(day):
        with ucn_mod._netcdf_lock:      # the HDF5 library is not thread-safe: build files and read them one at a time
            if day not in l2:
                l2[day] = l2_nc(tmp_path, day)
            return l2[day]

    def handler(url, **kwargs):
        if "action=netcdf_download" in url:
            day = url.split("date=")[1].split("&")[0]
            level = url.split("level=")[1].split("&")[0]
            if day in missing:
                return FakeResponse(404, b"")
            return FakeResponse(200, l2_for(day) if level == "2" else files[level])
        if "/viewer/?ceilometer=" in url:
            days = listed if listed is not None else ["2026-11-01", "2026-11-02", "2026-11-03"]
            return FakeResponse(200, text=f"<script>window.availableDatesJson = {json.dumps(days)};</script>")
        raise AssertionError(url)

    return handler


def dl(h, *a, **kw):
    kw.setdefault("silent", True)
    kw.setdefault("pause", 0)
    return UCN.download(*a, session_handler=h, **kw)


def cache_files():
    return sorted(p.name for p in atmoz.cache_dir("ucn").iterdir())


# ---- _validate / _url -------------------------------------------------------

def test_validate_expands_days_ceilometers_levels_without_duplicates_and_defaults_to_l2():
    jobs = UCN._validate([9, "09", 4], "2026-11-01", "2026-11-03", ["L2", "l3", "L3"])
    assert len(jobs) == 2 * 3 * 2
    assert {j["level"] for j in jobs} == {"L2", "L3"}
    assert UCN._name(jobs[0]) == "CEILO-9_20261101_L2"
    assert {j["level"] for j in UCN._validate(9, "2026-11-14")} == {"L2"}


def test_validate_rejects_bad_input_and_l0b():
    with pytest.raises(ValueError, match="before"):
        UCN._validate(9, "2026-11-14", "2026-11-13")
    with pytest.raises(ValueError, match="Unknown level"):
        UCN._validate(9, "2026-11-14", level="L1")
    with pytest.raises(ValueError, match="L0b.*planned"):
        UCN._validate(9, "2026-11-14", level="L0b")
    assert "L0b" in UCN.planned_levels and "L0b" not in UCN.levels


def test_url():
    j = UCN._validate(9, "2026-11-14", level="L3")[0]
    assert UCN._url(j).endswith("/viewer/pbl_plots.php?action=netcdf_download&ceilometer=9&date=2026-11-14&level=3")


# ---- download ----------------------------------------------------------------

def test_download_l2_returns_long_frames(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    ds = dl(h, 9, "2026-11-01")
    assert isinstance(ds, AtmozDataset) and ds.name == "UCN" and ds.source == "UCN"
    assert ds.metadata == {"endpoint": "ucn", "files": ["CEILO-9_20261101_L2"], "missing": []}
    f = ds["CEILO-9_20261101_L2"]
    assert isinstance(f, AtmozFrame) and len(f) == 6 * 4                      # 6 times x 4 altitudes
    df = f.df
    assert df.index.name == "time" and str(df.index.tz) == "UTC"
    assert f.t[0] == pd.Timestamp("2026-11-01", tz="UTC") and f.t[4] - f.t[0] == pd.Timedelta("10min")
    assert f.axes == {"t": "index", "lat": "lat", "lon": "lon", "alt": "alt"}
    assert df["alt"].iloc[:4].tolist() == [0.01, 0.02, 0.03, 0.04]            # km
    assert df["value"].iloc[4:8].tolist() == [1.0, 7.0, 13.0, 19.0]           # time step 1 of the (altitude, time) array
    assert df["value"].dtype == np.float32
    assert df["nsl_ml_height"].dtype == np.float64 and df["nsl_ml_height"].iloc[8:12].isna().all()
    assert df["nsl_ml_height"].iloc[4:8].tolist() == [120.0] * 4
    assert "cloud_layers" not in df and "jump_primary" not in df
    assert (df["lat"] == 38.92).all() and (df["lon"] == -77.01).all()
    assert f.units["value"] == "km^-1.sr^-1" and f.units["alt"] == "km" and f.units["nsl_ml_height"] == "m"
    m = f.metadata
    assert m["ceilometer"] == 9 and m["level"] == "L2" and m["date"] == "2026-11-01" and m["file"] == "CEILO-9_20261101_L2"
    assert m["altitude_m"] == ALT.tolist() and m["list_labels"]["jump_primary"] == ["NSL", "RL"]
    assert sorted(m["not_in_frame"]) == ["cloud_layers", "jump_primary"] and m["attrs"]["algorithm_version"] == "V1"


def test_the_cache_holds_only_parquet(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    dl(h, 9, "2026-11-01")
    assert cache_files() == ["CEILO-9_20261101_L2.parquet", "CEILO-9_20261101_L2.parquet.meta.json"]


def test_second_call_is_served_from_the_cache(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    dl(h, 9, "2026-10-01")  # an old (final) day
    n = len(h.calls)
    dl(h, 9, "2026-10-01")
    assert len(h.calls) == n
    dl(h, 9, "2026-10-01", refresh=True)
    assert len(h.calls) == n + 1


def test_l3_keeps_only_the_hourly_table(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    ds = dl(h, 9, "2026-11-01", level="L3")
    f = ds["CEILO-9_20261101_L3"]
    assert list(f.df.columns) == ["alt", "nsl_ml_hourly_mean", "nsl_ml_hourly_count", "lat", "lon"] and len(f) == 3
    assert f.df["alt"].isna().all() and str(f.t.tz) == "UTC" and f.units["nsl_ml_hourly_mean"] == "m"
    assert "value" not in f.df and "backscatter" in f.metadata["note"]


def test_a_netcdf_left_by_an_earlier_version_is_converted_without_network(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    legacy = atmoz.cache_dir("ucn") / "CEILO-9_20261101_L2.nc"
    legacy.write_bytes(l2_nc(tmp_path))
    ds = dl(h, 9, "2026-11-01")
    assert len(ds["CEILO-9_20261101_L2"]) == 24 and h.calls == []
    assert not legacy.exists()


# ---- missing days ------------------------------------------------------------

def test_missing_day_raises_by_default_and_is_never_cached(tmp_path, fake_http):
    h = fake_http(server(tmp_path, missing={"2026-11-02"}))
    with pytest.raises(DownloadError, match="CEILO-9_20261102_L2"):
        dl(h, 9, "2026-11-01", "2026-11-02")
    assert not [n for n in cache_files() if "20261102" in n]


def test_an_html_page_instead_of_a_file_counts_as_no_data(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, b"No matching files found.\n"))
    with pytest.raises(DownloadError, match="No matching files"):
        dl(h, 9, "2026-11-01")
    assert cache_files() == []


def test_skip_missing_leaves_gaps_and_lists_them(tmp_path, fake_http):
    h = fake_http(server(tmp_path, missing={"2026-11-02"}))
    ds = dl(h, 9, "2026-11-01", "2026-11-03", skip_missing=True)
    assert ds.metadata["missing"] == ["CEILO-9_20261102_L2"] and len(ds) == 2


def test_skip_missing_does_not_ask_for_days_the_viewer_does_not_list(tmp_path, fake_http):
    h = fake_http(server(tmp_path, listed=["2026-11-01", "2026-11-03"]))
    ds = dl(h, 9, "2026-11-01", "2026-11-03", skip_missing=True)
    assert ds.metadata["missing"] == ["CEILO-9_20261102_L2"]
    assert not any("date=2026-11-02" in url and "netcdf_download" in url for url, _ in h.calls)
    assert sum("netcdf_download" in url for url, _ in h.calls) == 2


def test_skip_missing_makes_no_request_when_everything_is_cached(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    dl(h, 9, "2026-11-01", "2026-11-02")
    n = len(h.calls)
    ds = dl(h, 9, "2026-11-01", "2026-11-02", skip_missing=True)
    assert len(ds) == 2 and len(h.calls) == n


def test_everything_missing_is_an_error_not_an_empty_dataset(tmp_path, fake_http):
    h = fake_http(server(tmp_path, missing={"2026-11-02"}))
    with pytest.raises(NoDataError):
        dl(h, 9, "2026-11-02", skip_missing=True)


def test_other_http_errors_still_fail(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(500, b""))
    with pytest.raises(DownloadError, match="500"):
        dl(h, 9, "2026-11-01")


# ---- size guard / available --------------------------------------------------

def test_big_requests_stop_before_any_network_and_allow_large_overrides(tmp_path, fake_http, monkeypatch):
    h = fake_http(server(tmp_path))
    with pytest.raises(RequestTooLargeError, match="allow_large"):
        dl(h, list(range(1, 68)), "2025-01-01", "2025-12-31")          # 67 ceilometers x a year
    assert h.calls == []
    monkeypatch.setattr(ucn_mod, "LARGE_DOWNLOAD_MB", 1.0)
    with pytest.raises(RequestTooLargeError):
        dl(h, 9, "2026-11-01", "2026-11-02")
    assert len(dl(h, 9, "2026-11-01", "2026-11-02", allow_large=True)) == 2


def test_cached_files_do_not_count_toward_the_download_limit(tmp_path, fake_http, monkeypatch):
    h = fake_http(server(tmp_path))
    dl(h, 9, "2026-11-01", "2026-11-02")
    monkeypatch.setattr(ucn_mod, "LARGE_DOWNLOAD_MB", 1.0)
    assert len(dl(h, 9, "2026-11-01", "2026-11-02")) == 2


def test_available_previews_a_request_without_downloading_data(tmp_path, fake_http):
    h = fake_http(server(tmp_path, listed=["2026-11-01", "2026-11-03"]))
    dl(h, 9, "2026-11-01")
    n_downloads = sum("netcdf_download" in url for url, _ in h.calls)
    av = UCN.available(9, "2026-11-01", "2026-11-03", session_handler=h, pause=0)
    assert av["listed"].tolist() == [True, False, True] and av["cached"].tolist() == [True, False, False]
    assert av["est_download_mb"].tolist() == [0.0, 1.4, 1.4]
    assert sum("netcdf_download" in url for url, _ in h.calls) == n_downloads


# ---- politeness ----------------------------------------------------------------

def test_requests_are_spaced_out():
    ucn_mod._last_request[0] = 0.0
    t = time.perf_counter()
    for _ in range(3):
        ucn_mod._pace(0.05)
    assert time.perf_counter() - t >= 0.09


def test_atmoz_identifies_itself():
    from atmoz.resources.utilities import default_session_handler
    assert default_session_handler().session().headers["User-Agent"].startswith("atmoz/")


# ---- fetch / sites / helpers ---------------------------------------------------

def test_fetch_returns_parquet_paths_only(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    out = UCN.fetch(9, "2026-11-01", "2026-11-02", silent=True, pause=0, session_handler=h)
    assert sorted(out) == ["CEILO-9_20261101_L2", "CEILO-9_20261102_L2"]
    assert all(p.suffix == ".parquet" for p in out.values())
    assert len(pd.read_parquet(out["CEILO-9_20261101_L2"])) == 6


def test_stack_joins_days_and_backscatter_makes_an_array(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    ds = dl(h, 9, "2026-11-01", "2026-11-02")
    both = UCN.stack(ds, 9)
    assert isinstance(both, AtmozFrame) and len(both) == 2 * 24 and both.t.is_monotonic_increasing
    assert both.metadata["dates"] == ["2026-11-01", "2026-11-02"] and both.units == ds["CEILO-9_20261101_L2"].units
    assert both.df["alt"].iloc[:4].tolist() == [0.01, 0.02, 0.03, 0.04]       # altitudes stay in order within a time
    arr = UCN.backscatter(both)
    assert arr.dims == ("time", "altitude") and arr.shape == (12, 4) and arr.altitude.values.tolist() == ALT.tolist()
    assert arr.attrs["units"] == "km^-1.sr^-1" and arr.values[1].tolist() == [1.0, 7.0, 13.0, 19.0]
    with pytest.raises(KeyError):
        UCN.stack(ds, 5)
    with pytest.raises(KeyError):
        UCN.stack(ds, 9, "L3")


def test_sites_parses_the_viewer_dropdown(fake_http):
    html = ('<select><option value="">Select a Ceilometer</option>'
            '<option value="1" >Ceilometer 01 - UMBC</option>'
            '<option value="9" selected>Ceilometer 09 - McMillan Reservoir, DC</option></select>')
    df = UCN.sites(session_handler=fake_http(lambda url, **kw: FakeResponse(200, text=html)))
    assert df.to_dict("list") == {"ceilometer": [1, 9], "site": ["UMBC", "McMillan Reservoir, DC"]}
