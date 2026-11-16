"""Offline tests for atmoz.met (MetPy parts skip when MetPy is not installed)."""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import atmoz
from atmoz import AtmozFrame


def test_import_does_not_load_metpy():
    code = "import atmoz, atmoz.plotting, atmoz.met, sys; atmoz.plot.available(); sys.exit('metpy' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_missing_metpy_gives_helpful_error(monkeypatch):
    for name in ("metpy", "metpy.calc", "metpy.units", "metpy.plots"):
        monkeypatch.setitem(sys.modules, name, None)
    frame = AtmozFrame(pd.DataFrame({"Temp_degC": [1.0]}, index=pd.DatetimeIndex(["2025-01-01"], tz="UTC")),
                       units={"Temp_degC": "°C"})
    with pytest.raises(ImportError, match=r"atmoz\[met\]"):
        atmoz.met.theta(frame)
    with pytest.raises(ImportError, match=r"atmoz\[met\]"):
        atmoz.met.skewt(frame)


pytest.importorskip("metpy")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _frame(**cols):
    n = len(next(iter(cols.values())))
    df = pd.DataFrame(cols, index=pd.date_range("2025-07-28 12:00", periods=n, freq="1min", tz="UTC"))
    units = {"Temp_degC": "°C", "Pressure_hPa": "hPa", "RH_percent": "%", "WindSpeed_mps": "m/s", "WindDirection_deg": "deg"}
    return AtmozFrame(df, units={k: v for k, v in units.items() if k in cols}, metadata={"site": "x"})


def test_theta_equals_temperature_at_1000_hpa():
    out = atmoz.met.theta(_frame(Temp_degC=[10.0, 20.0], Pressure_hPa=[1000.0, 1000.0]))
    assert out.df["theta"].to_numpy() == pytest.approx([283.15, 293.15])
    assert out.units["theta"] == "K" and out.units["Temp_degC"] == "°C" and out.metadata == {"site": "x"}


def test_theta_uses_declared_units():
    frame = _frame(Temp_degC=[10.0], Pressure_hPa=[100000.0]).with_units(Pressure_hPa="Pa")
    assert atmoz.met.theta(frame).df["theta"].iloc[0] == pytest.approx(283.15)
    with pytest.raises(ValueError, match="no declared unit"):
        atmoz.met.theta(AtmozFrame(frame.df, units={"Temp_degC": "°C"}))


def test_dewpoint():
    frame = _frame(Temp_degC=[15.0, 15.0, 15.0], RH_percent=[100.0, 50.0, 0.0])
    out = atmoz.met.dewpoint(frame)
    assert out.df["dewpoint"].iloc[0] == pytest.approx(15.0, abs=0.02)
    assert out.df["dewpoint"].iloc[1] == pytest.approx(4.7, abs=0.2)
    assert np.isnan(out.df["dewpoint"].iloc[2])
    assert out.units["dewpoint"] == "°C"
    frac = frame.with_units(RH_percent="1")
    frac.df["RH_percent"] = [1.0, 0.5, 0.0]
    assert atmoz.met.dewpoint(frac).df["dewpoint"].iloc[0] == pytest.approx(15.0, abs=0.02)
    with pytest.raises(ValueError, match="relative humidity"):
        atmoz.met.dewpoint(frame.with_units(RH_percent="hPa"))


def test_wind_components_and_back():
    frame = _frame(WindSpeed_mps=[5.0] * 4, WindDirection_deg=[0.0, 90.0, 180.0, 270.0])
    out = atmoz.met.wind_components(frame)
    assert out.df["u"].to_numpy() == pytest.approx([0, -5, 0, 5], abs=1e-9)
    assert out.df["v"].to_numpy() == pytest.approx([-5, 0, 5, 0], abs=1e-9)
    assert out.units["u"] == out.units["v"] == "m/s"
    back = atmoz.met.wind_speed_direction(out)
    assert back.df["speed"].to_numpy() == pytest.approx(5.0)
    assert back.df["direction"].to_numpy() == pytest.approx([360, 90, 180, 270])   # MetPy: north is 360
    assert back.units["direction"] == "deg"


def _profiles(times, mixed_km, ref="agl", elevation=None, n=40):
    """Mixed layer (constant theta 300 K) up to mixed_km, then +10 K per km."""
    alt = np.linspace(0.0, 3.0, n)
    theta_k = 300.0 + np.where(alt > mixed_km, (alt - mixed_km) * 10.0, 0.0)
    rows = [pd.DataFrame({"alt": alt + (elevation or 0.0), "theta": theta_k}, index=pd.DatetimeIndex([t] * n, tz="UTC"))
            for t in times]
    meta = {"alt_reference": ref, **({"site_elevation_km": elevation} if elevation is not None else {})}
    return AtmozFrame(pd.concat(rows), units={"alt": "km", "theta": "K"}, alt="alt", metadata=meta)


def test_pbl_height_known_mixed_layer():
    frame = _profiles(["2025-07-28 12:00", "2025-07-28 13:00"], mixed_km=1.2)
    out = atmoz.met.pbl_height(frame)
    assert list(out.df.index) == list(pd.DatetimeIndex(["2025-07-28 12:00", "2025-07-28 13:00"], tz="UTC"))
    assert out.df["pbl_height"].to_numpy() == pytest.approx(1.2 + 0.15, abs=1e-6)   # 1.5 K over the 10 K/km gradient
    assert out.units["pbl_height"] == "km" and out.metadata["alt_reference"] == "agl"


def test_pbl_height_asl_conversion_and_missing_elevation():
    asl = _profiles(["2025-07-28 12:00"], mixed_km=1.2, ref="asl", elevation=0.4)
    assert atmoz.met.pbl_height(asl).df["pbl_height"].iloc[0] == pytest.approx(1.35, abs=1e-6)
    with pytest.raises(ValueError, match="site_elevation_km"):
        atmoz.met.pbl_height(asl.with_metadata(site_elevation_km=None))
    with pytest.raises(ValueError, match="not stated"):
        atmoz.met.pbl_height(asl.with_metadata(alt_reference=None))


def test_pbl_height_nan_and_short_profiles():
    frame = _profiles(["2025-07-28 12:00", "2025-07-28 13:00", "2025-07-28 14:00"], mixed_km=1.2)
    df = frame.df.copy()
    col = df.columns.get_loc("theta")
    df.iloc[40:80, col] = 300.0                       # never warms: no PBL top
    df.iloc[80:, col] = np.nan
    df.iloc[80:83, col] = [300.0, 301.0, 305.0]       # too few valid levels
    out = atmoz.met.pbl_height(AtmozFrame(df, units=frame.units, alt="alt", metadata=frame.metadata))
    assert out.df["pbl_height"].isna().tolist() == [False, True, True]


def test_pbl_height_single_profile_ascent_only():
    up = _profiles(["2025-07-28 12:00"], mixed_km=0.8)
    df = pd.concat([up.df, up.df.iloc[::-1]])
    df.index = pd.date_range("2025-07-28 12:00", periods=len(df), freq="10s", tz="UTC")
    frame = AtmozFrame(df, units=up.units, alt="alt", metadata=up.metadata)
    out = atmoz.met.pbl_height(frame, group=None)
    assert len(out) == 1 and out.df.index[0] == df.index[0]
    assert out.df["pbl_height"].iloc[0] == pytest.approx(0.95, abs=1e-6)
    assert len(atmoz.met.pbl_height(frame)) == len(df)                     # grouped by time: every row its own profile


def test_pbl_height_celsius_theta():
    frame = _profiles(["2025-07-28 12:00"], mixed_km=1.2)
    df = frame.df.assign(theta=frame.df["theta"] - 273.15)
    out = atmoz.met.pbl_height(AtmozFrame(df, units={"alt": "km", "theta": "°C"}, alt="alt", metadata=frame.metadata))
    assert out.df["pbl_height"].iloc[0] == pytest.approx(1.35, abs=1e-6)


def test_skewt_returns_figure():
    p = np.linspace(1000, 200, 30)
    frame = _frame(Temp_degC=20 - (1000 - p) * 0.07, Pressure_hPa=p, RH_percent=[70.0] * 30,
                   WindSpeed_mps=[8.0] * 30, WindDirection_deg=[270.0] * 30)
    frame = atmoz.met.wind_components(atmoz.met.dewpoint(frame))
    fig, ax = atmoz.met.skewt(frame, dewpoint="dewpoint", wind=("u", "v"))
    assert ax.figure is fig and ax.get_ylim() == (1000.0, 100.0)
    fig2, _ = atmoz.plot.skewt(frame, figsize=(5, 5))
    assert tuple(fig2.get_size_inches()) == (5, 5)
    fig3, ax3 = plt.subplots()
    fig4, ax4 = atmoz.plot("skewt", frame, ax=ax3)
    assert fig4 is fig3


def test_functions_keep_frame_structure():
    frame = _frame(Temp_degC=[10.0], Pressure_hPa=[900.0])
    out = atmoz.met.theta(frame, out="th")
    assert isinstance(out, AtmozFrame) and "th" in out.df and "th" not in frame.df
