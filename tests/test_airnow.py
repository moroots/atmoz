from datetime import date, datetime

import pandas as pd
import pytest

from conftest import FakeResponse
from atmoz.dataclasses import AtmozDataset
from atmoz.exceptions import APIError, DownloadError, MissingCredentialError
from atmoz.surface import AirNow, AirNowAPI
from atmoz.surface.AirNow_API import AirNowParams, make_windows


def dl(h, key, *args, **kw):
    """AirNowAPI.download with a fake HTTP layer."""
    return AirNowAPI.download(*args, api_key=key, session_handler=h, **kw)


KEY = "SECRET-KEY-12345"


def obs(hour="2020-01-01T00:00", value=30.0):
    return {"Latitude": 38.9, "Longitude": -77.0, "UTC": hour, "Parameter": "OZONE", "Unit": "PPB", "Value": value,
            "RawConcentration": value, "AQI": 20, "Category": 1, "SiteName": "DC", "AgencyName": "EPA",
            "FullAQSCode": "110010043", "IntlAQSCode": "840110010043"}


# ---- windows -----------------------------------------------------------------------------------

def test_windows_one_per_day_without_overlap_and_end_day_included():
    assert make_windows("2025-08-01", "2025-08-03") == [
        ("2025-08-01T00", "2025-08-01T23"), ("2025-08-02T00", "2025-08-02T23"), ("2025-08-03T00", "2025-08-03T23"),
    ]


def test_windows_start_equals_end_is_one_full_day():
    assert make_windows("2025-08-01", "2025-08-01") == [("2025-08-01T00", "2025-08-01T23")]
    assert make_windows(date(2025, 8, 1), date(2025, 8, 1)) == [("2025-08-01T00", "2025-08-01T23")]


def test_windows_exact_hours_and_custom_chunk():
    assert make_windows("2025-08-01T00", "2025-08-01T05", "2h") == [
        ("2025-08-01T00", "2025-08-01T01"), ("2025-08-01T02", "2025-08-01T03"), ("2025-08-01T04", "2025-08-01T05")]
    assert make_windows(datetime(2025, 8, 1, 6), datetime(2025, 8, 1, 6)) == [("2025-08-01T06", "2025-08-01T06")]


def test_windows_last_window_is_trimmed_and_exclusive_mode():
    assert make_windows("2025-08-01T00", "2025-08-01T08", "6h")[-1] == ("2025-08-01T06", "2025-08-01T08")
    assert make_windows("2025-08-01T00", "2025-08-01T12", "6h", end_inclusive=False)[0] == ("2025-08-01T00", "2025-08-01T06")


def test_windows_reject_reversed_range():
    with pytest.raises(ValueError):
        make_windows("2025-08-02", "2025-08-01")


# ---- params / URL / key handling ---------------------------------------------------------------

def test_params_normalise_bbox_keeping_precision_and_sign():
    p = AirNowParams(BBOX=[-80.655479, 35.574398, -72.086143, 41.415693], parameters="OZONE,PM25")
    assert p.BBOX[0] == "-80.655479" and p.parameters == ["OZONE", "PM25"]
    with pytest.raises(ValueError):
        AirNowParams(BBOX=[1, 2, 3])


def test_build_url_hides_key_unless_given():
    a = AirNowAPI(api_key=KEY)
    job = {"startDate": "2020-01-01T00", "endDate": "2020-01-01T23"}
    assert KEY not in a._url(job)
    assert KEY in a._url(job, with_key=True)
    assert KEY not in repr(a)


def test_missing_key_raises_and_never_prompts():
    with pytest.raises(MissingCredentialError):
        AirNowAPI().api_key


def test_key_found_in_env_file():
    AirNow.set_api_key(KEY)
    assert AirNowAPI().api_key == KEY


def test_connection_error_message_does_not_leak_key(fake_http):
    import requests

    def boom(url, **kw):
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")

    a = AirNowAPI(api_key=KEY, session_handler=fake_http(boom))
    with pytest.raises(APIError) as err:
        a._fetch_one({"startDate": "2020-01-01T00", "endDate": "2020-01-01T23"})
    assert KEY not in str(err.value)
    assert err.value.__suppress_context__  # original exception (which has the key) is not chained into tracebacks


def test_http_error_body_is_redacted(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(403, text=f"bad key {KEY}"))
    with pytest.raises(APIError) as err:
        AirNowAPI(api_key=KEY, session_handler=h)._fetch_one({})
    assert KEY not in str(err.value) and err.value.status_code == 403


# ---- fetching ----------------------------------------------------------------------------------

def test_download_one_request_per_day_in_order(fake_http):
    def handler(url, **kw):
        day = "2020-01-0" + url.split("startDate=2020-01-0")[1][0]
        return FakeResponse(200, json_data=[obs(f"{day}T00:00")])

    h = fake_http(handler)
    ds = dl(h, KEY, "2020-01-01", "2020-01-03", silent=True, use_cache=False)
    assert len(h.calls) == 3
    assert all("startDate=2020-01-0" in c[0] and KEY in c[0] for c in h.calls)
    assert ds.frame.t.strftime("%Y-%m-%dT%H:%M").tolist() == ["2020-01-01T00:00", "2020-01-02T00:00", "2020-01-03T00:00"]


def test_same_day_range_works(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False)
    assert len(h.calls) == 1


def test_failure_is_all_or_nothing_and_redacted(fake_http):
    def handler(url, **kw):
        return FakeResponse(500, text=f"oops {KEY}") if "startDate=2020-01-02" in url else FakeResponse(200, json_data=[obs()])

    h = fake_http(handler)
    with pytest.raises(DownloadError) as err:
        dl(h, KEY, "2020-01-01", "2020-01-03", silent=True, use_cache=False)
    assert KEY not in str(err.value) and "2020-01-02" in str(err.value) and len(err.value.failed) == 1


def test_json_error_object_with_http_200_is_an_error(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data={"WebServiceError": [{"Message": "bad"}]}))
    with pytest.raises(DownloadError):
        dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False)


def test_settled_windows_are_cached_recent_ones_are_not(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    dl(h, KEY, "2020-01-01", "2020-01-02", silent=True)
    n = len(h.calls)
    dl(h, KEY, "2020-01-01", "2020-01-02", silent=True)
    assert len(h.calls) == n  # served from the file cache
    dl(h, KEY, "2020-01-01", "2020-01-02", silent=True, refresh=True)
    assert len(h.calls) == 2 * n

    today = datetime.now().strftime("%Y-%m-%d")
    before = len(h.calls)
    dl(h, KEY, today, today, silent=True)
    dl(h, KEY, today, today, silent=True)
    assert len(h.calls) == before + 2  # recent data may still change: never cached


def test_cache_key_ignores_api_key_but_not_the_query(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    dl(h, "one", "2020-01-01", "2020-01-01", silent=True)
    dl(h, "two", "2020-01-01", "2020-01-01", silent=True)
    assert len(h.calls) == 1
    dl(h, "two", "2020-01-01", "2020-01-01", silent=True, parameters=["PM25"])
    assert len(h.calls) == 2


def test_fetch_writes_parquet_without_the_key(fake_http, tmp_path):
    import pandas as pd
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    files = AirNowAPI.fetch("2020-01-01", "2020-01-02", output_dir=tmp_path, silent=True, api_key=KEY, session_handler=h)
    (path,) = files.values()
    assert path.exists() and ":" not in path.name
    assert len(pd.read_parquet(path)) == 1  # identical rows from both days are de-duplicated
    assert KEY not in path.read_bytes().decode("latin-1")


def test_half_specified_range_is_rejected():
    with pytest.raises(ValueError):
        AirNowAPI.download(start_date="2020-01-01", api_key=KEY)


def test_download_returns_atmoz_dataset_with_one_frame_per_site_and_parameter(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs("2020-01-01T00:00")]))
    ds = dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False)
    assert isinstance(ds, AtmozDataset) and list(ds) == ["840110010043/OZONE"]
    assert ds.name == "airnow_2020-01-01T00_2020-01-01T23" and ds.source == "AirNow API"
    assert ds.metadata["endpoint"] == "airnow" and ds.metadata["rows"] == {ds.name: 1}
    assert ds.metadata["start"] == "2020-01-01T00" and ds.metadata["end"] == "2020-01-01T23"
    assert "sites" in ds.metadata
    assert ds.frame is ds["840110010043/OZONE"]


def test_frame_has_utc_time_index_value_lat_lon_and_units(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs("2020-01-01T01:00", 31.0), obs("2020-01-01T00:00")]))
    af = dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False).frame
    assert isinstance(af.t, pd.DatetimeIndex) and str(af.t.tz) == "UTC" and af.t.name == "time"
    assert af.t.is_monotonic_increasing and af.df["value"].tolist() == [30.0, 31.0]
    assert {"value", "lat", "lon", "site_name", "agency"} <= set(af.df.columns)
    assert af.axes == {"t": "index", "lat": "lat", "lon": "lon", "alt": None}
    assert (af.lat == 38.9).all() and (af.lon == -77.0).all()
    assert af.units == {"value": "ppb"}
    assert af.metadata["site"] == "840110010043" and af.metadata["parameter"] == "OZONE"
    assert af.metadata["site_name"] == "DC" and af.metadata["agency"] == "EPA"


def test_units_are_normalised(fake_http):
    pm = {**obs(), "Parameter": "PM2.5", "Unit": "UG/M3"}
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[pm]))
    ds = dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False)
    assert ds["840110010043/PM2.5"].units == {"value": "µg/m3"}


def test_multiple_sites_and_parameters_get_their_own_frames(fake_http):
    rows = [obs("2020-01-01T00:00"), obs("2020-01-01T01:00"),
            {**obs(), "Parameter": "PM2.5", "Unit": "UG/M3"},
            {**obs(), "IntlAQSCode": "840240053001", "FullAQSCode": "240053001", "Latitude": 39.3, "Longitude": -76.5}]
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=rows))
    ds = dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False)
    assert sorted(ds) == ["840110010043/OZONE", "840110010043/PM2.5", "840240053001/OZONE"]
    assert len(ds["840110010043/OZONE"]) == 2
    assert ds["840240053001/OZONE"].lat.iloc[0] == 39.3
    assert ds["840110010043/PM2.5"].units["value"] == "µg/m3" and ds["840110010043/OZONE"].units["value"] == "ppb"


def test_empty_result_is_an_empty_dataset(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[]))
    ds = dl(h, KEY, "2020-01-01", "2020-01-01", silent=True, use_cache=False)
    assert isinstance(ds, AtmozDataset) and len(ds) == 0
    assert ds.metadata["endpoint"] == "airnow" and ds.metadata["rows"] == {ds.name: 0}


def test_fetch_of_an_empty_result_still_raises(fake_http, tmp_path):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[]))
    with pytest.raises(ValueError):
        AirNowAPI.fetch("2020-01-01", "2020-01-01", output_dir=tmp_path, silent=True, api_key=KEY, session_handler=h)


def test_facade_download_returns_the_same_dataset(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    ds = AirNow.download(endpoint="airnow", start_date="2020-01-01", end_date="2020-01-01", api_key=KEY,
                         session_handler=h, silent=True, use_cache=False)
    assert isinstance(ds, AtmozDataset) and list(ds) == ["840110010043/OZONE"]
