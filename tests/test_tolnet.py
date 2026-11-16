"""Offline tests for atmoz.lidar.TOLNET (fake HTTP, no network, no login)."""

import matplotlib

matplotlib.use("Agg")

import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from conftest import FakeResponse

from atmoz.dataclasses import AtmozDataset, AtmozFrame
from atmoz.exceptions import DownloadError, NoDataError
from atmoz.lidar import TOLNET

ALT = [0.5, 1.0, 1.5]


def file_row(i, day, nrt="false"):
    return {"row": i, "count": 2, "id": i, "file_name": f"tolnet_{day}.json", "file_server_location": "x",
            "author": "a", "instrument_group_id": 1, "product_type_id": 4, "file_type_id": 2,
            "start_data_date": f"{day} 00:00:00", "end_data_date": f"{day} 23:59:00", "upload_date": f"{day} 23:59:59",
            "public": True, "instrument_group_name": "NASA JPL SMOL-2", "folder_name": "f", "current_pi": "p",
            "doi": "d", "citation_url": "c", "product_type_name": "HIRES", "processing_type_name": "Centrally Processed",
            "file_type_name": "json", "revision": 1, "near_real_time": nrt, "file_size": 10, "latitude": 39.2,
            "longitude": -76.4, "altitude": 10.0, "isAccessible": True}


def payload(day):
    return {"LATITUDE.INSTRUMENT": 39.24, "LONGITUDE.INSTRUMENT": -76.363,
            "fileInfo": {"start_data_date": f"{day} 00:00:00", "instrument_group_name": "NASA JPL SMOL-2",
                         "processing_type_name": "Centrally Processed"},
            "altitude": {"data": ALT, "units": "km"},
            "datetime": {"data": [f"{day}T00:00:00Z", f"{day}T01:00:00Z"]},
            "value": {"data": [[10.0, 20.0, 30.0], [11.0, "bad", 31.0]], "units": "ppbv"},
            "uncertainty": {"data": [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], "units": "ppbv"}}


DAYS = {1: "2025-07-28", 2: "2025-07-29"}


def handler(fail_ids=()):
    def h(url, **kw):
        if "/data/json/" in url:
            fid = int(url.rstrip("/").split("/")[-1])
            if fid in fail_ids:
                return FakeResponse(500, text="boom")
            body = json.dumps(payload(DAYS[fid])).encode()
            return FakeResponse(200, body=body, headers={"Content-Length": str(len(body))})
        page = url.split("?")[0].rstrip("/").split("/")[-1]
        if url.split("?")[0].rstrip("/").split("/")[-2] == "data" and page.isdigit():   # the paginated file list
            rows = [file_row(i, d) for i, d in DAYS.items()] if page == "1" else []
            return FakeResponse(200, json_data=rows)
        return FakeResponse(404)
    return h


def test_download_returns_dataset_with_long_frames_and_metadata(fake_http):
    h = fake_http(handler())
    ds = TOLNET.download("2025-07-28", "2025-07-29", session_handler=h, silent=True)
    assert isinstance(ds, AtmozDataset) and ds.name == "TOLNet" and ds.source == "TOLNet"
    assert set(ds) == {"tolnet_2025-07-28.json", "tolnet_2025-07-29.json"}
    f = ds["tolnet_2025-07-28.json"]
    assert isinstance(f, AtmozFrame) and len(f) == 6                      # 2 times x 3 altitudes
    assert f.df.index.name == "time" and str(f.df.index.tz) == "UTC"
    assert f.t[0] == pd.Timestamp("2025-07-28T00:00:00Z") and f.t[3] == pd.Timestamp("2025-07-28T01:00:00Z")
    assert f.df["alt"].tolist() == ALT * 2 and f.axes == {"t": "index", "lat": "lat", "lon": "lon", "alt": "alt"}
    assert f.df["value"].iloc[0] == 10.0 and pd.isna(f.df["value"].iloc[4])       # non-numeric -> NaN
    assert f.df["uncertainty"].tolist() == [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
    assert set(f.df["lat"]) == {39.24} and set(f.df["lon"]) == {-76.363}
    assert f.units["value"] == "ppb" and f.units["alt"] == "km" and f.units["uncertainty"] == "ppb"
    assert f.metadata["instrument"] == "NASA JPL SMOL-2" and f.metadata["processing"] == "Centrally Processed"
    assert f.metadata["product"] == "HIRES" and f.metadata["file_id"] == 1 and f.metadata["near_real_time"] is False
    assert ds.metadata["endpoint"] == "tolnet" and len(ds.metadata["file_list"]) == 2
    assert ds.metadata["rows"] == {n: 6 for n in ds}


def test_group_gives_instrument_site_then_day(fake_http):
    ds = TOLNET.download("2025-07-28", "2025-07-29", session_handler=fake_http(handler()), silent=True)
    groups = TOLNET.group(ds)
    assert list(groups) == [("NASA JPL SMOL-2", "Centrally Processed", "39.24x-76.363")]
    days = next(iter(groups.values()))
    assert list(days) == ["2025-07-28", "2025-07-29"] and all(isinstance(f, AtmozFrame) for f in days.values())
    assert days["2025-07-29"] is ds["tolnet_2025-07-29.json"]


def test_second_download_uses_the_cache(fake_http):
    h = fake_http(handler())
    TOLNET.download("2025-07-28", "2025-07-29", session_handler=h, silent=True)
    n = sum("/data/json/" in c[0] for c in h.calls)
    TOLNET.download("2025-07-28", "2025-07-29", session_handler=h, silent=True)
    assert n == 2 and sum("/data/json/" in c[0] for c in h.calls) == 2


def test_failure_is_all_or_nothing(fake_http):
    h = fake_http(handler(fail_ids=(2,)))
    with pytest.raises(DownloadError) as err:
        TOLNET.download("2025-07-28", "2025-07-29", session_handler=h, silent=True, max_workers=1)
    assert "tolnet_2025-07-29.json" in str(err.value)


def test_no_files_raises_nodata(fake_http):
    h = fake_http(lambda url, **kw: FakeResponse(200, json_data=[]))
    with pytest.raises(NoDataError):
        TOLNET.download("2025-07-28", "2025-07-29", session_handler=h, silent=True)


def test_bad_query_is_rejected_before_any_request(fake_http):
    h = fake_http(handler())
    with pytest.raises(ValueError):
        TOLNET.download("2025-07-30", "2025-07-29", session_handler=h, silent=True)
    assert h.calls == []


def test_old_geos_cf_kwarg_points_to_new_api(fake_http):
    with pytest.raises(TypeError, match="GEOS_CF"):
        TOLNET.download("2025-07-28", "2025-07-29", GEOS_CF=True, session_handler=fake_http(handler()))


def test_fetch_saves_raw_files(fake_http, tmp_path):
    out = TOLNET.fetch("2025-07-28", "2025-07-29", dest_dir=tmp_path / "raw", session_handler=fake_http(handler()),
                       silent=True)
    assert set(out) == {"tolnet_2025-07-28.json", "tolnet_2025-07-29.json"}
    assert all(p.exists() and p.parent == tmp_path / "raw" for p in out.values())
    assert json.loads(out["tolnet_2025-07-28.json"].read_text())["fileInfo"]["start_data_date"].startswith("2025-07-28")


def test_plot_curtain_and_plot_curtains(fake_http):
    ds = TOLNET.download("2025-07-28", "2025-07-29", session_handler=fake_http(handler()), silent=True)
    group = next(iter(TOLNET.group(ds).values()))
    fig, ax = TOLNET.plot_curtain(group)
    assert ax.get_title("left") and len(fig.axes) >= 2    # curtain + colorbar (+ watermark)
    fig1, _ = TOLNET.plot_curtain(ds["tolnet_2025-07-28.json"])          # one long frame is pivoted to time x altitude
    from atmoz.plotting.curtain import _wide
    wide = _wide(ds["tolnet_2025-07-28.json"])
    assert wide.shape == (2, 3) and wide.columns.tolist() == ALT and wide.iloc[0].tolist() == [10.0, 20.0, 30.0]
    figs = TOLNET.plot_curtains(ds)
    assert len(figs) == 1
    plt.close("all")


def test_altitude_in_metres_is_converted_and_missing_units_warn(tmp_path):
    import json
    import pytest
    from atmoz.lidar.tolnet import TOLNET
    p = payload("2025-07-28")
    p["altitude"] = {"data": [500, 1000, 1500], "units": "m"}
    f = tmp_path / "m.json"
    f.write_text(json.dumps(p))
    assert list(TOLNET._parse_one(f)[1]["alt"].iloc[:3]) == [0.5, 1.0, 1.5]
    p["altitude"] = {"data": ALT}
    f.write_text(json.dumps(p))
    with pytest.warns(UserWarning, match="altitude units"):
        TOLNET._parse_one(f)
    p["altitude"] = {"data": ALT, "units": "ft"}
    f.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="unsupported altitude unit"):
        TOLNET._parse_one(f)


def test_real_file_layout_units_and_altitude_reference(tmp_path):
    import json
    p = payload("2025-07-28")
    p["altitude"] = {"attributes": {"VAR_DESCRIPTION": "Altitude above sea level (km)", "VAR_UNITS": "km"}, "data": ALT}
    p["value"] = {"attributes": {"VAR_UNITS": "ppbv"}, "data": p["value"]["data"]}
    p["fileInfo"]["instrument_altitude"] = 90
    f = tmp_path / "1.json"
    f.write_text(json.dumps(p))
    meta, df = TOLNET._parse_one(f)
    assert TOLNET._unit_of(meta["value"]) == "ppbv" and TOLNET._alt_reference(meta) == "asl"
    assert TOLNET._unit_of({"units": "km"}) == "km" and TOLNET._unit_of({}) is None
