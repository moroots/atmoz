from datetime import date, datetime

import pytest

from conftest import FakeResponse
from atmoz.exceptions import APIError, DownloadError, MissingCredentialError
from atmoz.surface import AirNow
from atmoz.surface.AirNow import AirNowParams, make_windows

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
    a = AirNow(api_key=KEY)
    assert KEY not in a._build_url(startDate="2020-01-01T00", endDate="2020-01-01T23")
    assert KEY in a._build_url(api_key=KEY, startDate="2020-01-01T00", endDate="2020-01-01T23")
    assert KEY not in repr(a)


def test_missing_key_raises_and_never_prompts():
    with pytest.raises(MissingCredentialError):
        AirNow().api_key


def test_key_found_in_env_file():
    AirNow.set_api_key(KEY)
    assert AirNow().api_key == KEY


def test_connection_error_message_does_not_leak_key(fake_http):
    import requests

    def boom(url, **kw):
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")

    a = AirNow(api_key=KEY, session_handler=fake_http(boom))
    with pytest.raises(APIError) as err:
        a._pull(startDate="2020-01-01T00", endDate="2020-01-01T23")
    assert KEY not in str(err.value)
    assert err.value.__suppress_context__  # original exception (which has the key) is not chained into tracebacks


def test_http_error_body_is_redacted(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(403, text=f"bad key {KEY}"))
    with pytest.raises(APIError) as err:
        AirNow(api_key=KEY, session_handler=h)._pull()
    assert KEY not in str(err.value) and err.value.status_code == 403


# ---- fetching ----------------------------------------------------------------------------------

def test_import_data_one_request_per_day_in_order(fake_http):
    def handler(url, **kw):
        day = "2020-01-0" + url.split("startDate=2020-01-0")[1][0]
        return FakeResponse(200, json_data=[obs(f"{day}T00:00")])

    h = fake_http(handler)
    ds = AirNow(api_key=KEY, session_handler=h).import_data("2020-01-01", "2020-01-03", silent=True, use_cache=False)
    assert len(h.calls) == 3
    assert all("startDate=2020-01-0" in c[0] and KEY in c[0] for c in h.calls)
    assert ds.data["UTC"].tolist() == ["2020-01-01T00:00", "2020-01-02T00:00", "2020-01-03T00:00"]


def test_same_day_range_works(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    AirNow(api_key=KEY, session_handler=h).import_data("2020-01-01", "2020-01-01", silent=True, use_cache=False)
    assert len(h.calls) == 1


def test_failure_is_all_or_nothing_and_redacted(fake_http):
    def handler(url, **kw):
        return FakeResponse(500, text=f"oops {KEY}") if "startDate=2020-01-02" in url else FakeResponse(200, json_data=[obs()])

    a = AirNow(api_key=KEY, session_handler=fake_http(handler))
    with pytest.raises(DownloadError) as err:
        a.import_data("2020-01-01", "2020-01-03", silent=True, use_cache=False)
    assert KEY not in str(err.value) and "2020-01-02" in str(err.value) and len(err.value.failed) == 1


def test_json_error_object_with_http_200_is_an_error(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data={"WebServiceError": [{"Message": "bad"}]}))
    with pytest.raises(DownloadError):
        AirNow(api_key=KEY, session_handler=h).import_data("2020-01-01", "2020-01-01", silent=True, use_cache=False)


def test_settled_windows_are_cached_recent_ones_are_not(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    a = AirNow(api_key=KEY, session_handler=h)
    a.import_data("2020-01-01", "2020-01-02", silent=True)
    n = len(h.calls)
    a.import_data("2020-01-01", "2020-01-02", silent=True)
    assert len(h.calls) == n  # served from the file cache
    a.import_data("2020-01-01", "2020-01-02", silent=True, refresh=True)
    assert len(h.calls) == 2 * n

    today = datetime.now().strftime("%Y-%m-%d")
    before = len(h.calls)
    a.import_data(today, today, silent=True)
    a.import_data(today, today, silent=True)
    assert len(h.calls) == before + 2  # recent data may still change: never cached


def test_cache_key_ignores_api_key_but_not_the_query(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    AirNow(api_key="one", session_handler=h).import_data("2020-01-01", "2020-01-01", silent=True)
    AirNow(api_key="two", session_handler=h).import_data("2020-01-01", "2020-01-01", silent=True)
    assert len(h.calls) == 1
    AirNow(api_key="two", session_handler=h).import_data("2020-01-01", "2020-01-01", silent=True, parameters=["PM25"])
    assert len(h.calls) == 2


def test_download_data_writes_parquet_without_the_key(fake_http, tmp_path):
    import pandas as pd
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[obs()]))
    path = AirNow(api_key=KEY, session_handler=h).download_data("2020-01-01", "2020-01-02", output_dir=tmp_path, silent=True)
    assert path.exists() and ":" not in path.name
    assert len(pd.read_parquet(path)) == 1  # identical rows from both days are de-duplicated
    assert KEY not in path.read_bytes().decode("latin-1")


def test_half_specified_range_is_rejected():
    with pytest.raises(ValueError):
        AirNow(api_key=KEY).import_data(start_date="2020-01-01")
