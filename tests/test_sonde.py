import numpy as np
import pandas as pd
import pytest

import atmoz
from atmoz import AtmozFrame
from atmoz.resources.icartt import read_icartt
from atmoz.sonde import Sonde


def ict(path, *, ozone_unit="ppbv", ozone_scale=1.0, start="2023-07-18", seconds=None, n=30, gps=True, with_mpa=True):
    seconds = seconds if seconds is not None else list(range(66561, 66561 + n))
    n = len(seconds)
    alt = np.linspace(0.18, 17.0, n)
    p = 1000.0 * np.exp(-alt / 7.0)
    ppb = 30.0 + 200.0 * alt
    mpa = ppb * p / 1e4
    cols = ["Index_number", "Seconds_UTC", "Pressure_hPa", "Altitude_km", "Ozone_mPa", "Ozone_ppmv", "Latitude_deg", "Longitude_deg"]
    var_lines = ["Seconds_UTC, seconds, start", "Pressure_hPa, hPa, p", "Altitude_km, km, alt", "Ozone_mPa, mPa, pO3",
                 f"Ozone_ppmv, {ozone_unit}, O3", "Latitude_deg, deg, lat", "Longitude_deg, deg, lon"]
    header = ["0, 1001", "Doe, Jane", "Univ", "Ozonesondes", "STAQS", "1, 1", f"{start.replace('-', ', ')}, 2023, 08, 21", "1",
              "Index_number, number, idx", str(len(var_lines)), ", ".join(["1"] * len(var_lines)),
              ", ".join(["-99999"] * len(var_lines)), *var_lines, "0", "4", "PLATFORM: Test Site", "LOCATION: 42.5, -87.8, 180 m altitude AMSL",
              "ULOD_FLAG: -77777", "REVISION: R0", ", ".join(cols)]
    header[0] = f"{len(header)}, 1001"
    rows = []
    for i in range(n):
        lat, lon = (42.5 + 0.001 * i, -87.8 + 0.001 * i) if gps else (-99999, -99999)
        rows.append(f"{i}, {seconds[i]}, {p[i]:.2f}, {alt[i]:.4f}, {mpa[i]:.4f}, {ppb[i] * ozone_scale:.6f}, {lat}, {lon}")
    path.write_text("\n".join(header + rows) + "\n", encoding="latin-1")
    return ppb


def test_read_icartt_header_and_fill_values(tmp_path):
    f = tmp_path / "a.ict"
    ict(f, gps=False)
    h, df = read_icartt(f)
    assert h["start_date"] == "2023-07-18" and h["variables"]["Altitude_km"]["units"] == "km"
    assert h["comments"]["PLATFORM"] == "Test Site" and -99999.0 in h["flags"] and -77777.0 in h["flags"]
    assert df["Latitude_deg"].isna().all() and df["Altitude_km"].notna().all()


def test_not_icartt_or_wrong_format_raises(tmp_path):
    f = tmp_path / "x.ict"
    f.write_text("hello\nworld\n")
    with pytest.raises(ValueError, match="not an ICARTT"):
        read_icartt(f)
    f.write_text("5, 2110\n\n\n\n\n")
    with pytest.raises(ValueError, match="not supported"):
        read_icartt(f)


def test_sonde_frame_shape_units_and_metadata(tmp_path):
    f = tmp_path / "flight.ict"
    ppb = ict(f)
    ds = Sonde.read(f)
    fr = ds["flight.ict"]
    assert isinstance(fr, AtmozFrame) and str(fr.t.tz) == "UTC"
    assert fr.t[0] == pd.Timestamp("2023-07-18 18:29:21", tz="UTC")
    assert fr.units["value"] == "ppb" and fr.units["alt"] == "km" and np.allclose(fr.df["value"], ppb, atol=0.01)
    assert fr.alt_reference == "asl" and fr.metadata["site_elevation_km"] == 0.18 and fr.metadata["platform"] == "Test Site"
    assert fr.location is not None and 42.5 < fr.location[0] < 42.6


def test_ozone_unit_is_checked_against_the_files_physics(tmp_path):
    mislabeled = tmp_path / "mis.ict"
    ict(mislabeled, ozone_unit="ppmv")                    # says ppmv, values are ppb
    with pytest.warns(UserWarning, match="declared in ppmv but matches ppbv"):
        fr = Sonde.read(mislabeled)["mis.ict"]
    assert np.isclose(fr.df["value"].iloc[0], 66.0, atol=0.1)
    honest = tmp_path / "ok.ict"
    ict(honest, ozone_unit="ppmv", ozone_scale=1e-3)      # truly ppmv
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fr = Sonde.read(honest)["ok.ict"]
    assert np.isclose(fr.df["value"].iloc[0], 66.0, atol=0.1)


def test_clock_wrapping_past_midnight_and_missing_gps(tmp_path):
    f = tmp_path / "w.ict"
    ict(f, seconds=[86390, 86395, 86399, 1, 5, 10, 20, 30, 40, 50, 60, 70], gps=False, ozone_unit="ppbv")
    fr = Sonde.read(f)["w.ict"]
    assert fr.t.is_monotonic_increasing and fr.t[-1] > pd.Timestamp("2023-07-19", tz="UTC")
    assert fr.location is None and fr.df["lat"].isna().all()


def test_read_directory_and_list_and_missing(tmp_path):
    ict(tmp_path / "a.ict")
    ict(tmp_path / "b.ict")
    assert list(Sonde.read(tmp_path)) == ["a.ict", "b.ict"]
    assert len(Sonde.read([tmp_path / "a.ict"])) == 1
    with pytest.raises(FileNotFoundError):
        Sonde.read(tmp_path / "empty_dir_that_has_no_ict")  if False else Sonde.read(tmp_path / "nothing")
    assert atmoz.Sonde is Sonde


def test_sonde_collocates_onto_a_lidar_profile(tmp_path):
    f = tmp_path / "s.ict"
    ict(f, n=600, seconds=list(range(66561, 66561 + 600 * 5, 5)))
    sonde = Sonde.read(f)["s.ict"]
    t = pd.DatetimeIndex([sonde.t[0] + pd.Timedelta("10min")])
    rows = [(t[0], a, 0.0) for a in (0.5, 1.5, 2.5)]
    df = pd.DataFrame(rows, columns=["time", "alt", "value"]).set_index("time")
    df["lat"], df["lon"] = 42.5, -87.8
    lidar = AtmozFrame(df, units={"value": "ppb", "alt": "km"}, lat="lat", lon="lon", alt="alt",
                       metadata={"alt_reference": "asl"})
    out = atmoz.collocate([lidar, sonde], names=["lidar", "sonde"], window="30min")
    assert out.df["sonde"].notna().all() and out.alt_reference == "asl"
    assert (np.diff(out.df["sonde"].to_numpy()) > 0).all()
