"""TEMPO: search windowing, request validation, ozone arithmetic, resumable downloads, lazy imports.

Runs without any optional dependency (no h5py, no earthaccess): Earthdata and the HTTPS session are faked.
Reading granule files is covered in ``test_tempo_granules.py`` (needs h5py).
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import requests

from atmoz.dataclasses import AtmozDataset, AtmozFrame
from atmoz.exceptions import DownloadError, NoDataError
from atmoz.satellite import tempo as tp
from atmoz.satellite.tempo import TEMPO, du_to_ppbv, integrate_layers


class FakeGranule:
    def __init__(self, stamp: str, product="TEMPO_NO2_L3", version="V04"):
        self.url = f"https://data.example.nasa/{product}_{version}_{stamp}_S001G01.nc"

    def data_links(self):
        return [self.url]


class FakeEarthData:
    def __init__(self, granules):
        self.granules, self.calls = granules, []

    def search_data(self, **params):
        self.calls.append(params)
        return list(self.granules)


GRANULES = [FakeGranule(s) for s in ("20250729T000500Z", "20250727T235800Z", "20250728T235900Z", "20250728T120000Z",
                                     "20250728T000100Z", "20250730T000000Z")]


def client(granules=GRANULES, **kw):
    return TEMPO(earthdata=FakeEarthData(granules), **kw)


# ---- imports -------------------------------------------------------------------------------------------------------------------
def test_import_needs_no_optional_dependencies():
    code = (
        "import sys, atmoz, atmoz.satellite\n"
        "from atmoz.satellite import TEMPO, du_to_ppbv, integrate_layers\n"
        "TEMPO('O3PROF_L3')\n"
        "heavy = {'h5py', 'earthaccess', 'xarray', 'geopandas', 'plotly', 'pydantic', 'cartopy'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_satellite_name_is_an_attribute_error():
    import atmoz.satellite as sat

    with pytest.raises(AttributeError):
        sat.nope


# ---- constructor ---------------------------------------------------------------------------------------------------------------
def test_unknown_product_and_version_are_rejected():
    with pytest.raises(ValueError, match=r"Unsupported product 'HCHO_L3'.*NO2_L3"):
        TEMPO("HCHO_L3")
    with pytest.raises(ValueError, match="V04"):
        TEMPO("NO2_L3", "4")


# ---- search --------------------------------------------------------------------------------------------------------------------
def test_search_keeps_scans_starting_in_the_window_oldest_first():
    t = client()
    found = t.search("2025-07-28")
    assert [g.url.split("_")[-2] for g in found] == ["20250728T000100Z", "20250728T120000Z", "20250728T235900Z"]
    call = t.earthdata.calls[0]
    assert call["short_name"] == "TEMPO_NO2_L3" and call["version"] == "V04"      # pinned: V03 and V04 would double count
    assert call["temporal"] == ("2025-07-28T00:00:00Z", "2025-07-29T00:00:00Z")


def test_search_end_date_is_inclusive_and_the_previous_evening_is_dropped():
    found = client().search("2025-07-28", "2025-07-29")
    stamps = [g.url.split("_")[-2] for g in found]
    assert stamps[0] == "20250728T000100Z" and stamps[-1] == "20250729T000500Z"
    assert "20250727T235800Z" not in stamps and "20250730T000000Z" not in stamps


def test_search_accepts_datetimes_and_timestamps():
    assert len(client().search(pd.Timestamp("2025-07-28 15:00", tz="UTC"))) == 3


def test_search_rejects_a_reversed_window_and_a_bad_date():
    with pytest.raises(ValueError, match="end_date is before start_date"):
        client().search("2025-07-29", "2025-07-28")
    with pytest.raises(ValueError, match="not a date"):
        client().search("yesterday")


def test_no_granules_is_a_no_data_error_naming_the_product_and_days():
    with pytest.raises(NoDataError, match=r"No TEMPO_O3PROF_L3 V04 granules found for 2025-01-01 to 2025-01-02"):
        client([], product="O3PROF_L3").download("2025-01-01", "2025-01-02", points=(38.9, -76.8))


def test_only_scans_keeps_granules_that_start_within_two_minutes_after_a_wanted_scan():
    t = client()
    got = t._search_or_raise("2025-07-28", None, only_scans=pd.DatetimeIndex(["2025-07-28 12:01:00"], tz="UTC"))
    assert [g.url.split("_")[-2] for g in got] == ["20250728T120000Z"]
    with pytest.raises(NoDataError, match="only_scans"):
        t._search_or_raise("2025-07-28", None, only_scans=[pd.Timestamp("2025-07-28 06:00")])


# ---- request validation (nothing is searched before these) ---------------------------------------------------------------------
@pytest.mark.parametrize("kwargs, message", [
    ({}, "Provide points and/or bbox"),
    ({"points": (95.0, 0.0)}, "outside lat"),
    ({"points": {"bbox": (1, 2)}}, "reserved"),
    ({"points": ("a", "b")}, "must be"),
    ({"bbox": (-77, 39, -78, 40)}, "west < east"),
    ({"bbox": (-77, 38)}, "bbox must be"),
    ({"points": (38.9, -76.8), "lowest_layer": True}, "profile products only"),
    ({"points": (38.9, -76.8), "max_quality": -1}, "max_quality"),
])
def test_bad_requests_fail_before_any_search(kwargs, message):
    t = client()
    with pytest.raises(ValueError, match=message):
        t.download("2025-07-28", **kwargs)
    assert t.earthdata.calls == []


def test_point_formats_are_normalised():
    assert tp._normalise_points((38.98, -76.83)) == {"38.980,-76.830": (38.98, -76.83)}
    assert tp._normalise_points([(1, 2), (3, 4)]) == {"1.000,2.000": (1.0, 2.0), "3.000,4.000": (3.0, 4.0)}
    assert tp._normalise_points({"GSFC": [38.98, -76.83]}) == {"GSFC": (38.98, -76.83)}


# ---- ozone arithmetic ----------------------------------------------------------------------------------------------------------
def test_du_to_ppbv_total_column_gives_the_familiar_mean_mixing_ratio():
    ppb = float(du_to_ppbv(300.0, 1013.25))      # 300 DU in the whole atmosphere: a mean of about 0.37 ppmv
    assert ppb == pytest.approx(373.6, rel=0.01)


def test_du_to_ppbv_is_linear_and_vectorised():
    dp = np.array([100.0, 200.0])
    out = du_to_ppbv(np.array([1.0, 1.0]), dp)
    assert out.shape == (2,) and out[0] == pytest.approx(2 * out[1])
    assert float(du_to_ppbv(2.0, 100.0)) == pytest.approx(2 * float(du_to_ppbv(1.0, 100.0)))


def profile_frame(variable="ozone_profile"):
    """Two pixels x one scan x three layers (top -> bottom), ground at 0.5 km."""
    t = pd.DatetimeIndex(["2025-07-28 18:00"] * 6, tz="UTC", name="time")
    lower, upper = [10.0, 4.5, 0.5], [30.0, 10.0, 4.5]
    df = pd.DataFrame({
        "lat": [39.0] * 3 + [39.04] * 3, "lon": [-76.8] * 6,
        "alt_lower": lower * 2, "alt_upper": upper * 2, "alt": [(a + b) / 2 for a, b in zip(lower, upper)] * 2,
        "pressure_thickness": [100.0, 400.0, 500.0] * 2, "partial_column": [10.0, 4.0, 2.0] * 2,
    }, index=t)
    df["value"] = du_to_ppbv(df["partial_column"], df["pressure_thickness"])
    return AtmozFrame(df, units={"value": "ppb", "alt": "km", "partial_column": "DU"}, lat="lat", lon="lon", alt="alt",
                      metadata={"alt_reference": "asl", "variable": variable})


def test_integrate_layers_weights_layers_by_vertical_overlap():
    out = integrate_layers(profile_frame(), 0.0, 2.5)           # 0.5-2.5 km is half of the bottom layer (0.5-4.5)
    assert isinstance(out, AtmozFrame) and len(out) == 2
    assert out.df["column"].tolist() == pytest.approx([1.0, 1.0])
    assert out.df["value"].iloc[0] == pytest.approx(float(du_to_ppbv(1.0, 250.0)))
    assert out.units == {"column": "DU", "value": "ppb", "lat": "deg", "lon": "deg"}
    assert out.axes["alt"] is None and out.location == pytest.approx((39.02, -76.8))
    assert out.metadata["layer_km"] == (0.0, 2.5) and out.metadata["layer_reference"] == "asl"
    assert str(out.t.tz) == "UTC"


def test_integrate_layers_agl_measures_from_each_pixels_ground():
    asl = integrate_layers(profile_frame(), 0.0, 2.0)            # 0.5-2.0 km: 1.5 of 4 km
    agl = integrate_layers(profile_frame(), 0.0, 2.0, reference="agl")   # ground (0.5) to 2.0 km agl = 0.5-2.5: 2 of 4 km
    assert asl.df["column"].iloc[0] == pytest.approx(2.0 * 1.5 / 4.0)
    assert agl.df["column"].iloc[0] == pytest.approx(2.0 * 2.0 / 4.0)
    assert agl.metadata["layer_reference"] == "agl"


def test_integrate_layers_on_a_dataset_and_its_errors():
    ds = AtmozDataset(name="x", source="TEMPO", frames={"a/ozone_profile": profile_frame()})
    out = integrate_layers(ds, 0, 4)
    assert isinstance(out, AtmozDataset) and list(out.keys()) == ["a/ozone_profile"]
    with pytest.raises(ValueError, match="do not add linearly"):
        integrate_layers(profile_frame("ozone_profile_error"), 0, 2)
    with pytest.raises(ValueError, match="z_min must be below"):
        integrate_layers(profile_frame(), 2, 1)
    with pytest.raises(ValueError, match="'asl' or 'agl'"):
        integrate_layers(profile_frame(), 0, 2, reference="msl")
    with pytest.raises(KeyError, match="partial_column"):
        integrate_layers(profile_frame().df.drop(columns="partial_column").pipe(
            lambda d: AtmozFrame(d, alt="alt", lat="lat", lon="lon")), 0, 2)
    with pytest.raises(TypeError):
        integrate_layers(pd.DataFrame(), 0, 2)


# ---- whole-granule downloads (resumable) ---------------------------------------------------------------------------------------
BODY = bytes(range(256)) * 40          # 10240 bytes


class Response:
    def __init__(self, status=200, body=b"", headers=None, fail_after=None):
        self.status_code, self.body, self.headers, self.fail_after = status, body, headers or {}, fail_after

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def iter_content(self, chunk_size=1):
        if self.fail_after is None:
            yield self.body
        else:
            yield self.body[:self.fail_after]
            raise requests.ConnectionError("connection reset")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Session:
    """Serves BODY, honours Range, and drops the connection after ``cut`` bytes on the first ``drops`` requests."""

    def __init__(self, drops=0, cut=4000, honour_range=True, status=None):
        self.drops, self.cut, self.honour_range, self.status, self.requests = drops, cut, honour_range, status, []

    def get(self, url, headers=None, stream=True, timeout=None):
        self.requests.append((url, dict(headers or {})))
        if self.status:
            return Response(self.status)
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers and "Range" in headers and self.honour_range else 0
        body = BODY[start:]
        drop = len(self.requests) <= self.drops
        if start:
            return Response(206, body, {"Content-Range": f"bytes {start}-{len(BODY) - 1}/{len(BODY)}"},
                            fail_after=self.cut if drop else None)
        return Response(200, body, {"Content-Length": str(len(BODY))}, fail_after=self.cut if drop else None)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(tp, "_RETRY_DELAY", 0.0)


def use_session(monkeypatch, session):
    monkeypatch.setattr(tp, "_session", lambda: session)
    return session


def test_fetch_returns_paths_of_downloaded_granules_in_the_cache(monkeypatch, fast):
    session = use_session(monkeypatch, Session())
    out = client().fetch("2025-07-28", silent=True)
    assert sorted(out) == ["TEMPO_NO2_L3_V04_20250728T000100Z_S001G01.nc", "TEMPO_NO2_L3_V04_20250728T120000Z_S001G01.nc",
                           "TEMPO_NO2_L3_V04_20250728T235900Z_S001G01.nc"]
    assert all(p.read_bytes() == BODY and p.parent.name == "tempo" for p in out.values())
    assert not list(next(iter(out.values())).parent.glob("*.part")) and len(session.requests) == 3


def test_fetch_skips_complete_files_and_refresh_downloads_again(monkeypatch, fast, tmp_path):
    session = use_session(monkeypatch, Session())
    t = client()
    t.fetch("2025-07-28", cache_dir=tmp_path / "d", silent=True)
    t.fetch("2025-07-28", cache_dir=tmp_path / "d", silent=True)
    assert len(session.requests) == 3
    t.fetch("2025-07-28", cache_dir=tmp_path / "d", silent=True, refresh=True)
    assert len(session.requests) == 6


def test_a_dropped_connection_resumes_with_a_range_request(monkeypatch, fast, tmp_path):
    session = use_session(monkeypatch, Session(drops=1, cut=4000))
    dest = tp._download_file("https://x/f.nc", tmp_path / "f.nc")
    assert dest.read_bytes() == BODY
    assert session.requests[0][1] == {} and session.requests[1][1] == {"Range": "bytes=4000-"}


def test_a_server_that_ignores_range_restarts_the_file(monkeypatch, fast, tmp_path):
    session = use_session(monkeypatch, Session(drops=1, cut=4000, honour_range=False))
    assert tp._download_file("https://x/f.nc", tmp_path / "f.nc").read_bytes() == BODY
    assert len(session.requests) == 2


def test_an_interrupted_file_never_looks_complete(monkeypatch, fast, tmp_path):
    use_session(monkeypatch, Session(drops=99, cut=100))
    with pytest.raises(OSError, match="download failed after 3 attempts"):
        tp._download_file("https://x/f.nc", tmp_path / "f.nc", retries=3)
    assert not (tmp_path / "f.nc").exists()


def test_http_errors_are_retried_then_reported_with_every_failed_granule(monkeypatch, fast):
    session = use_session(monkeypatch, Session(status=503))
    with pytest.raises(DownloadError, match=r"3 of 3 granule download\(s\) failed") as err:
        client().fetch("2025-07-28", retries=2, silent=True)
    assert "503" in str(err.value) and "TEMPO_NO2_L3_V04_20250728T120000Z" in str(err.value)
    assert len(session.requests) == 6                       # 3 granules x 2 attempts
    assert len(err.value.failed) == 3


def test_fetch_with_no_granules_is_a_no_data_error():
    with pytest.raises(NoDataError):
        client([]).fetch("2025-07-28", silent=True)
