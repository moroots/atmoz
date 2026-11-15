import io
import zipfile

import pandas as pd
import pytest

from conftest import FakeResponse
from atmoz.exceptions import DownloadError
from atmoz.resources.atmoz_dataclasses import atmoz_dataset
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

def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hourly_44201_2024.csv", "\n".join([HEADER, *ROWS]) + "\n")
    return buf.getvalue()


def test_download_end_to_end_uses_cache(fake_http):
    body = _zip_bytes()
    h = fake_http(lambda url, **kw: FakeResponse(200, body, {"Content-Length": str(len(body))}))
    ds = EPA_PreGen.download("ozone", "hourly", 2024, session_handler=h, silent=True)
    assert isinstance(ds, atmoz_dataset) and ds.datatype == "surface"
    out = ds.data
    assert list(out) == ["hourly_44201_2024.csv"] and len(out["hourly_44201_2024.csv"]) == 3
    assert ds.metadata["endpoint"] == "aqs" and ds.metadata["rows"] == {"hourly_44201_2024.csv": 3}
    assert h.calls[0][0] == "https://aqs.epa.gov/aqsweb/airdata/hourly_44201_2024.zip"  # no double slash
    EPA_PreGen.download("ozone", "hourly", 2024, session_handler=h, silent=True)
    assert len(h.calls) == 1  # second call came from the file cache


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
