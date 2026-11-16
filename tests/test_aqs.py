import io
import zipfile

import pandas as pd
import pytest

from conftest import FakeResponse
from atmoz.exceptions import DownloadError
from atmoz.dataclasses import AtmozDataset
from atmoz.surface import AirNow, EPA_PreGen, read_aqs_zip

HEADER = ('"State Code","County Code","Site Num","Parameter Code","POC","Latitude","Longitude","Datum",'
          '"Parameter Name","Date Local","Time Local","Date GMT","Time GMT","Sample Measurement",'
          '"Units of Measure","MDL","Uncertainty","Qualifier","Method Type","Method Code","Method Name",'
          '"State Name","County Name","Date of Last Change"')
ROWS = [
    '"01","073","0023","44201",1,33.553056,-86.815,"WGS84","Ozone","2024-01-01","00:00","2024-01-01","06:00",0.031,'
    '"Parts per million",0.005,"","","FEM","087","UV ABSORPTION","Alabama","Jefferson","2025-04-04"',
    '"06","037","0002","44201",1,34.0689999,-118.4567891,"NAD83","Ozone","2024-01-01","01:00","2024-01-01","09:00",0.02,'
    '"Parts per million",0.005,"","IA","FEM","087","UV ABSORPTION","California","Los Angeles","2025-04-04"',
    '"02","090","0034","44201",2,64.845,-147.727,"WGS84","Ozone","2024-01-01","02:00","2024-01-01","11:00",0.01,'
    '"Parts per million",0.005,"","","FEM","087","UV ABSORPTION","Alaska","Fairbanks","2025-04-04"',
]


def make_zip(path, name="hourly_44201_2024.csv"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, "\n".join([HEADER, *ROWS]) + "\n")
    return path


@pytest.fixture
def aqs_zip(tmp_path):
    return make_zip(tmp_path / "hourly_44201_2024.zip")


def test_read_preserves_precision_sign_and_leading_zeros(aqs_zip):
    df = read_aqs_zip(aqs_zip)
    assert df["Latitude"].dtype == "float64" and df["Longitude"].dtype == "float64"
    assert df["Latitude"].tolist() == [33.553056, 34.0689999, 64.845]
    assert df["Longitude"].tolist() == [-86.815, -118.4567891, -147.727]  # exact, negative signs kept
    assert df["Method Code"].astype(str).tolist() == ["087"] * 3
    assert df["Site Num"].astype(str).tolist() == ["0023", "0002", "0034"]
    assert isinstance(df["State Name"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(df["Date Local"])


def test_columns_subset_and_bbox(aqs_zip):
    df = read_aqs_zip(aqs_zip, columns=["Latitude", "Sample Measurement"], bbox=(-125, 30, -70, 40))
    assert list(df.columns) == ["Latitude", "Sample Measurement"]
    assert len(df) == 2  # Alabama + California, Alaska is outside the box


def test_unknown_column_is_a_clear_error(aqs_zip):
    with pytest.raises(KeyError, match="Nope"):
        read_aqs_zip(aqs_zip, columns=["Nope"])


def test_pandas_fallback_matches_arrow(aqs_zip):
    import zipfile as zf_mod
    from atmoz.surface.AQS import _read_aqs_arrow, _read_aqs_pandas, _read_header
    with zf_mod.ZipFile(aqs_zip) as zf:
        member = zf.namelist()[0]
        names = _read_header(zf, member)
        a = _read_aqs_arrow(zf, member, names, None, None)
        b = _read_aqs_pandas(zf, member, names, None, None)
    assert a["Latitude"].tolist() == b["Latitude"].tolist()
    assert a["Longitude"].tolist() == b["Longitude"].tolist()
    assert a["Method Code"].astype(str).tolist() == b["Method Code"].astype(str).tolist()


# ---- validator ---------------------------------------------------------------------------------

def test_validator_accepts_names_codes_and_ints():
    combos = EPA_PreGen._validate(["ozone", "44201", "SPEC"], ["hourly", "daily"], [2023, "2024"], silent=True)
    # ozone and 44201 are the same file -> deduplicated; non-numeric codes work with several resolutions
    assert {(c["parameter"], c["resolution"], c["year"]) for c in combos} == {
        (p, r, y) for p in ("44201", "SPEC") for r in ("hourly", "daily") for y in (2023, 2024)
    }
    assert len(combos) == 8


def test_validator_scalar_year_int():
    assert EPA_PreGen._validate("ozone", "hourly", 2024) == [
        {"resolution": "hourly", "parameter": "44201", "year": 2024}
    ]


def test_validator_unknown_parameter_and_resolution():
    with pytest.raises(ValueError, match="Did you mean"):
        EPA_PreGen._validate("ozon", "hourly", 2024)
    with pytest.raises(ValueError, match="Unknown resolution"):
        EPA_PreGen._validate("ozone", "weekly", 2024)


def test_validator_warns_on_unpublished_combo():
    with pytest.warns(UserWarning, match="annual"):
        assert EPA_PreGen._validate("ozone", ["annual", "hourly"], 2024)[0]["resolution"] == "hourly"


# ---- download ----------------------------------------------------------------------------------

EXTRA_ROWS = [
    # same Alabama site, a second hour, and a collocated second monitor (POC 2)
    '"01","073","0023","44201",1,33.553056,-86.815,"WGS84","Ozone","2024-01-01","01:00","2024-01-01","07:00",0.033,'
    '"Parts per million",0.005,"","","FEM","087","UV ABSORPTION","Alabama","Jefferson","2025-04-04"',
    '"01","073","0023","44201",2,33.553056,-86.815,"WGS84","Ozone","2024-01-01","00:00","2024-01-01","06:00",0.041,'
    '"Parts per million",0.005,"","","FEM","047","UV ABSORPTION","Alabama","Jefferson","2025-04-04"',
    # a Maryland site on EST: local 2024-01-02 00:00 is 05:00 GMT
    '"24","031","3001","44201",1,39.1,-77.1,"NAD83","Ozone","2024-01-02","00:00","2024-01-02","05:00",0.015,'
    '"Parts per million",0.005,"","","FEM","087","UV ABSORPTION","Maryland","Montgomery","2025-04-04"',
    # another parameter in the same file
    '"01","073","0023","42101",1,33.553056,-86.815,"WGS84","Carbon monoxide","2024-01-01","00:00","2024-01-01","06:00",0.4,'
    '"Parts per million",0.5,"","","FEM","054","NDIR","Alabama","Jefferson","2025-04-04"',
]
KEYS = {"010730023/44201", "010730023/42101", "060370002/44201", "020900034/44201", "240313001/44201"}
FILE = "hourly_44201_2024.csv"


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(FILE, "\n".join([HEADER, *ROWS, *EXTRA_ROWS]) + "\n")
    return buf.getvalue()


@pytest.fixture
def aqs(fake_http):
    body = _zip_bytes()
    h = fake_http(lambda url, **kw: FakeResponse(200, body, {"Content-Length": str(len(body))}))

    def run(**kwargs):
        return EPA_PreGen.download("ozone", "hourly", 2024, session_handler=h, silent=True, **kwargs)

    run.handler = h
    return run


def test_download_end_to_end_uses_cache(fake_http):
    body = _zip_bytes()
    h = fake_http(lambda url, **kw: FakeResponse(200, body, {"Content-Length": str(len(body))}))
    ds = EPA_PreGen.download("ozone", "hourly", 2024, session_handler=h, silent=True)
    assert isinstance(ds, AtmozDataset) and ds.source == "EPA AQS" and ds.name == "EPA AQS hourly_44201_2024"
    assert ds.metadata["endpoint"] == "aqs" and ds.metadata["files"] == [FILE]
    assert ds.metadata["rows"] == {FILE: 7} and ds.metadata["filters"] == {}
    assert h.calls[0][0] == "https://aqs.epa.gov/aqsweb/airdata/hourly_44201_2024.zip"  # no double slash
    EPA_PreGen.download("ozone", "hourly", 2024, session_handler=h, silent=True)
    assert len(h.calls) == 1  # second call came from the file cache


def test_one_frame_per_site_and_parameter(aqs):
    ds = aqs()
    assert set(ds) == KEYS
    assert ds.schema == {"lat": "lat", "lon": "lon", "alt": None}
    assert len(ds["010730023/44201"]) == 3 and len(ds["010730023/42101"]) == 1


def test_frame_has_utc_index_local_time_and_units(aqs):
    af = aqs()["010730023/44201"]
    assert af.t.name == "time" and str(af.t.tz) == "UTC"
    assert list(af.t) == list(pd.to_datetime(["2024-01-01 06:00", "2024-01-01 06:00", "2024-01-01 07:00"], utc=True))
    assert list(af.df["local_time"]) == list(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"]))
    assert af.df["local_time"].dt.tz is None
    assert list(af.df["poc"]) == [1, 2, 1]  # collocated monitors share the frame
    assert list(af.df["value"]) == [0.031, 0.041, 0.033]
    assert af.units == {"value": "ppm"} and af.axes["lat"] == "lat" and af.axes["lon"] == "lon"
    assert af.df["lat"].iloc[0] == 33.553056 and af.df["lon"].iloc[0] == -86.815
    assert {"mdl", "qualifier", "method_type", "method_code"} <= set(af.df.columns)
    assert af.df["method_code"].iloc[0] == "087"
    assert af.metadata["site_id"] == "010730023" and af.metadata["pocs"] == [1, 2]
    assert af.metadata["parameter_code"] == "44201" and af.metadata["parameter_name"] == "Ozone"
    assert af.metadata["state_name"] == "Alabama" and af.metadata["units_raw"] == "Parts per million"


def test_utc_time_comes_from_the_gmt_columns_not_a_guess(aqs):
    md = aqs()["240313001/44201"]
    assert md.t[0] == pd.Timestamp("2024-01-02 05:00", tz="UTC")
    assert md.df["local_time"].iloc[0] == pd.Timestamp("2024-01-02 00:00")


def test_bbox_filter(aqs):
    ds = aqs(bbox=(-125, 30, -70, 40))
    assert set(ds) == KEYS - {"020900034/44201"}  # Alaska is outside
    assert aqs(bbox=(-90, 33, -85, 34)).metadata["filters"] == {"bbox": (-90.0, 33.0, -85.0, 34.0)}
    assert set(aqs(bbox=(-90, 33, -85, 34))) == {"010730023/44201", "010730023/42101"}


def test_date_filters_are_on_the_local_date_and_inclusive(aqs):
    assert set(aqs(start_date="2024-01-02")) == {"240313001/44201"}
    assert "240313001/44201" not in aqs(end_date="2024-01-01")
    assert set(aqs(start_date="2024-01-02", end_date=pd.Timestamp("2024-01-02"))) == {"240313001/44201"}
    assert aqs(start_date="2024-01-02").metadata["filters"] == {"start_date": "2024-01-02"}


def test_site_state_and_county_filters(aqs):
    assert set(aqs(site="240313001")) == {"240313001/44201"}
    assert set(aqs(site=("24", "31", "3001"))) == {"240313001/44201"}
    assert set(aqs(site=["240313001", "020900034"])) == {"240313001/44201", "020900034/44201"}
    assert set(aqs(site="840240313001")) == {"240313001/44201"}
    assert set(aqs(state="Maryland")) == {"240313001/44201"}
    assert set(aqs(state="maryland")) == {"240313001/44201"}
    assert set(aqs(state=["24", 1])) == {"240313001/44201", "010730023/44201", "010730023/42101"}
    assert set(aqs(county="Jefferson")) == {"010730023/44201", "010730023/42101"}
    assert set(aqs(state="01", county="073")) == {"010730023/44201", "010730023/42101"}
    assert set(aqs(county=37)) == {"060370002/44201"}


def test_parameter_and_poc_filters(aqs):
    assert set(aqs(parameter=42101)) == {"010730023/42101"}
    assert set(aqs(parameter="ozone")) == KEYS - {"010730023/42101"}
    af = aqs(poc=2)["010730023/44201"]
    assert list(af.df["poc"]) == [2] and af.metadata["pocs"] == [2]


def test_where_filters_any_column(aqs):
    assert set(aqs(where={"Method Code": "047"})) == {"010730023/44201"}
    assert len(aqs(where={"Method Code": "047"})["010730023/44201"]) == 1
    assert set(aqs(where={"State Name": ["Alaska", "Maryland"]})) == {"020900034/44201", "240313001/44201"}
    assert set(aqs(where={"Date Local": "2024-01-02"})) == {"240313001/44201"}
    assert aqs(where={"Method Code": "047"}).metadata["filters"] == {"where": {"Method Code": ["047"]}}
    assert len(aqs(parameter=44201, poc=1, state="Alabama")["010730023/44201"]) == 2  # filters combine


def test_unknown_where_column_and_bad_filters_are_clear_errors(aqs):
    with pytest.raises(DownloadError, match="Nope"):
        aqs(where={"Nope": 1})
    with pytest.raises(ValueError, match="9 digits"):
        aqs(site="123")
    with pytest.raises(ValueError, match="west"):
        aqs(bbox=(10, 0, -10, 5))
    with pytest.raises(ValueError, match="after end_date"):
        aqs(start_date="2024-02-01", end_date="2024-01-01")
    assert len(aqs.handler.calls) == 1  # only the unknown column needed the file; the other mistakes failed up front


def test_empty_result_is_an_empty_dataset_with_metadata(aqs):
    ds = aqs(site="999999999")
    assert isinstance(ds, AtmozDataset) and len(ds) == 0 and ds.time_start is None
    assert ds.source == "EPA AQS" and ds.metadata["rows"] == {FILE: 0} and ds.metadata["filters"] == {"site": [("99", "999", "9999")]}


def test_parquet_cache_is_reused_without_another_request(aqs, isolated_home, monkeypatch):
    aqs()
    parquet = isolated_home / "cache" / "aqs" / "hourly_44201_2024.parquet"
    assert parquet.exists() and len(aqs.handler.calls) == 1
    stamp = parquet.stat().st_mtime_ns
    import atmoz.surface.AQS as aqs_module
    monkeypatch.setattr(aqs_module, "read_aqs_zip", lambda *a, **k: pytest.fail("the zip was parsed again"))
    assert set(aqs(state="Maryland")) == {"240313001/44201"}  # a different filter reads the same parquet
    assert set(aqs(site="020900034")) == {"020900034/44201"}
    assert len(aqs.handler.calls) == 1 and parquet.stat().st_mtime_ns == stamp


def test_a_refreshed_zip_is_parsed_again(aqs, isolated_home):
    aqs()
    parquet = isolated_home / "cache" / "aqs" / "hourly_44201_2024.parquet"
    parquet.unlink()
    assert set(aqs()) == KEYS and parquet.exists()  # rebuilt from the cached zip, still no new request
    assert len(aqs.handler.calls) == 1
    aqs(refresh=True)
    assert len(aqs.handler.calls) == 2 and set(aqs()) == KEYS


def test_use_cache_false_still_filters(aqs, isolated_home):
    assert set(aqs(use_cache=False, state="Maryland")) == {"240313001/44201"}
    assert not (isolated_home / "cache" / "aqs").exists()


def test_other_resolutions_are_not_downloadable_as_frames(aqs):
    with pytest.raises(ValueError, match="hourly"):
        EPA_PreGen.download("ozone", "daily", 2024, silent=True)


def test_frames_need_gmt_time():
    from atmoz.surface.AQS import _frames_from_aqs
    df = pd.DataFrame({"State Code": ["24"], "County Code": ["031"], "Site Num": ["3001"], "Parameter Code": ["44201"],
                       "Sample Measurement": [0.03], "Latitude": [39.0], "Longitude": [-77.0]})
    with pytest.raises(ValueError, match="UTC offset"):
        _frames_from_aqs(df, "daily_44201_2024.csv")


def test_unknown_units_are_kept_as_spelled():
    from atmoz.surface.AQS import _norm_unit
    assert _norm_unit("Parts per million") == "ppm" and _norm_unit("Micrograms/cubic meter (LC)") == "µg/m3"
    assert _norm_unit("Degrees Fahrenheit") == "Degrees Fahrenheit" and _norm_unit(None) is None


def test_download_is_all_or_nothing(fake_http):
    body = _zip_bytes()

    def handler(url, **kw):
        return FakeResponse(404) if "2023" in url else FakeResponse(200, body, {"Content-Length": str(len(body))})

    h = fake_http(handler)
    with pytest.raises(DownloadError, match="hourly_44201_2023") as err:
        EPA_PreGen.download("ozone", "hourly", [2023, 2024], session_handler=h, silent=True)
    assert len(err.value.failed) == 1


def test_use_cache_false_keeps_nothing_on_disk(fake_http, isolated_home):
    body = _zip_bytes()
    h = fake_http(lambda url, **kw: FakeResponse(200, body, {"Content-Length": str(len(body))}))
    EPA_PreGen.download("ozone", "hourly", 2024, session_handler=h, silent=True, use_cache=False)
    assert not (isolated_home / "cache" / "aqs").exists()


def test_fetch_returns_zip_paths_only(fake_http):
    body = _zip_bytes()
    h = fake_http(lambda url, **kw: FakeResponse(200, body, {"Content-Length": str(len(body))}))
    paths = EPA_PreGen.fetch("ozone", "hourly", 2024, session_handler=h, silent=True)
    assert paths["hourly_44201_2024.csv"].suffix == ".zip" and paths["hourly_44201_2024.csv"].exists()


def test_facade_dispatches_to_the_backends(monkeypatch):
    from atmoz.surface import AirNowAPI

    monkeypatch.setattr(EPA_PreGen, "download", classmethod(lambda cls, **kw: ("aqs", kw)))
    monkeypatch.setattr(AirNowAPI, "download", classmethod(lambda cls, **kw: ("airnow", kw)))
    assert AirNow.download(endpoint="aqs", years=2024) == ("aqs", {"years": 2024})
    assert AirNow.download(start_date="2020-01-01") == ("airnow", {"start_date": "2020-01-01"})
    with pytest.raises(ValueError, match="Unsupported endpoint"):
        AirNow.download(endpoint="nope")
