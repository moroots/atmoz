"""UCN (Unified Ceilometer Network): offline, HTTP is faked and the NetCDF files are built in memory."""
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from conftest import FakeResponse
from atmoz.exceptions import DownloadError, NoDataError
from atmoz.lidar import UCN
from atmoz.resources.atmoz_dataclasses import atmoz_dataset

EPOCH = pd.Timestamp("2026-11-01").timestamp()


def l0b_bytes(tmp_path):
    ds = xr.Dataset(
        {
            "time_UTC": ("time_dim", EPOCH + 10.0 * np.arange(6)),
            "elevation_m": ("range_dim", 10.0 * np.arange(1, 5, dtype="float32")),
            "bsc_prof": (("time_dim", "range_dim"), np.arange(24, dtype="float32").reshape(6, 4)),
            "latitude_deg": ((), np.float32(38.92)),
        }
    )
    path = tmp_path / "l0b.nc"
    ds.to_netcdf(path)
    return path.read_bytes()


def l2_bytes(tmp_path):
    ds = xr.Dataset({"nsl_ml_height": ("time", [100.0, 120.0, 140.0])},
                    coords={"time": pd.date_range("2026-11-01", periods=3, freq="10min")})
    path = tmp_path / "l2.nc"
    ds.to_netcdf(path)
    return path.read_bytes()


def server(tmp_path, *, missing=()):
    """Fake UCN: L0b/L2 files for every day except those in *missing* (dates as 'YYYY-MM-DD')."""
    l0b, l2 = l0b_bytes(tmp_path), l2_bytes(tmp_path)

    def handler(url, **kwargs):
        day = url.split("date=")[1].split("&")[0]
        if "download_file.php" in url:
            if day in missing:  # the real server answers "no file" with a 200 and an HTML page
                return FakeResponse(200, b"No matching files found.\n", headers={"Content-Type": "text/html"})
            return FakeResponse(200, l0b)
        if "level=2" in url:
            return FakeResponse(404, b"") if day in missing else FakeResponse(200, l2)
        raise AssertionError(url)

    return handler


# ---- _validate / _url -------------------------------------------------------

def test_validate_expands_days_ceilometers_levels_without_duplicates():
    jobs = UCN._validate([9, "09", 4], "2026-11-01", "2026-11-03", ["L0b", "l2", "L2"])
    assert len(jobs) == 2 * 3 * 2  # ceilometers {9, 4} x 3 days x levels {L0b, L2}
    assert {j["level"] for j in jobs} == {"L0b", "L2"}
    assert UCN._name(jobs[0]) == "CEILO-9_20261101_L0b.nc"


def test_validate_defaults_to_one_day_and_rejects_bad_input():
    assert len(UCN._validate(9, "2026-11-14")) == 1
    with pytest.raises(ValueError, match="before"):
        UCN._validate(9, "2026-11-14", "2026-11-13")
    with pytest.raises(ValueError, match="Unknown level"):
        UCN._validate(9, "2026-11-14", level="L1")


def test_urls():
    j0 = UCN._validate(9, "2026-11-14", level="L0b")[0]
    j3 = UCN._validate(9, "2026-11-14", level="L3")[0]
    assert UCN._url(j0).endswith("/viewer/download_file.php?ceilometer=9&date=2026-11-14")
    assert UCN._url(j3).endswith("/viewer/pbl_plots.php?action=netcdf_download&ceilometer=9&date=2026-11-14&level=3")


# ---- download ----------------------------------------------------------------

def test_download_returns_dataset_with_time_and_altitude_coords(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    ds = UCN.download(9, "2026-11-01", silent=True, session_handler=h)
    assert isinstance(ds, atmoz_dataset) and ds.datatype == "lidar"
    assert ds.metadata == {"endpoint": "ucn", "files": ["CEILO-9_20261101_L0b.nc"], "missing": []}
    x = ds.data["CEILO-9_20261101_L0b.nc"]
    assert x["bsc_prof"].dims == ("time", "altitude")
    assert pd.Timestamp(x.time.values[0]) == pd.Timestamp("2026-11-01") and x.sizes["time"] == 6
    assert x.altitude.values.tolist() == [10.0, 20.0, 30.0, 40.0]
    assert "time_UTC" not in x and x.attrs["ucn_ceilometer"] == 9 and x.attrs["ucn_level"] == "L0b"


def test_second_call_is_served_from_the_cache(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    UCN.download(9, "2026-10-01", silent=True, session_handler=h)  # an old (frozen) day
    n = len(h.calls)
    UCN.download(9, "2026-10-01", silent=True, session_handler=h)
    assert len(h.calls) == n
    UCN.download(9, "2026-10-01", silent=True, session_handler=h, refresh=True)
    assert len(h.calls) == n + 1


def test_level_two_keeps_its_own_coordinates(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    ds = UCN.download(9, "2026-11-01", level=["L0b", "L2"], silent=True, session_handler=h)
    assert list(ds.data) == ["CEILO-9_20261101_L0b.nc", "CEILO-9_20261101_L2.nc"]
    assert ds.data["CEILO-9_20261101_L2.nc"]["nsl_ml_height"].dims == ("time",)


# ---- missing days ------------------------------------------------------------

@pytest.mark.parametrize("level", ["L0b", "L2"])
def test_missing_day_raises_by_default_and_is_never_cached(tmp_path, fake_http, level):
    h = fake_http(server(tmp_path, missing={"2026-11-02"}))
    with pytest.raises(DownloadError, match="CEILO-9_20261102"):
        UCN.download(9, "2026-11-01", "2026-11-02", level=level, silent=True, session_handler=h)
    import atmoz
    assert not list(atmoz.cache_dir("ucn").glob("CEILO-9_20261102*"))  # no HTML masquerading as a .nc


@pytest.mark.parametrize("level", ["L0b", "L2"])
def test_skip_missing_leaves_gaps_and_lists_them(tmp_path, fake_http, level):
    h = fake_http(server(tmp_path, missing={"2026-11-02"}))
    ds = UCN.download(9, "2026-11-01", "2026-11-03", level=level, skip_missing=True, silent=True, session_handler=h)
    assert ds.metadata["missing"] == [f"CEILO-9_20261102_{level}.nc"]
    assert len(ds.data) == 2


def test_everything_missing_is_an_error_not_an_empty_dataset(tmp_path, fake_http):
    h = fake_http(server(tmp_path, missing={"2026-11-02"}))
    with pytest.raises(NoDataError):
        UCN.download(9, "2026-11-02", skip_missing=True, silent=True, session_handler=h)


def test_other_http_errors_still_fail(tmp_path, fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(500, b""))
    with pytest.raises(DownloadError, match="500"):
        UCN.download(9, "2026-11-01", skip_missing=True, silent=True, session_handler=h)


# ---- fetch / sites -----------------------------------------------------------

def test_fetch_returns_paths_only(tmp_path, fake_http):
    h = fake_http(server(tmp_path))
    out = UCN.fetch(9, "2026-11-01", "2026-11-02", silent=True, session_handler=h)
    assert sorted(out) == ["CEILO-9_20261101_L0b.nc", "CEILO-9_20261102_L0b.nc"]
    assert all(p.exists() for p in out.values())


def test_sites_parses_the_viewer_dropdown(fake_http):
    html = ('<select><option value="">Select a Ceilometer</option>'
            '<option value="1" >Ceilometer 01 - UMBC</option>'
            '<option value="9" selected>Ceilometer 09 - McMillan Reservoir, DC</option></select>')
    df = UCN.sites(session_handler=fake_http(lambda url, **kw: FakeResponse(200, text=html)))
    assert df.to_dict("list") == {"ceilometer": [1, 9], "site": ["UMBC", "McMillan Reservoir, DC"]}
