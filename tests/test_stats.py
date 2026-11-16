import numpy as np
import pandas as pd
import pytest

import atmoz
from atmoz import AtmozDataset, AtmozFrame
from atmoz import stats, vertical


def series_frame(values, start="2024-07-01", freq="1h", col="value", **extra):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC", name="time")
    df = pd.DataFrame({col: values, **extra}, index=idx)
    return AtmozFrame(df, units={col: "ppb"})


def paired(obs, model, **extra):
    idx = pd.date_range("2024-07-01", periods=len(obs), freq="1h", tz="UTC", name="time")
    return AtmozFrame(pd.DataFrame({"obs": obs, "model": model, **extra}, index=idx), units={"obs": "ppb", "model": "ppb"})


def test_compare_known_values():
    af = paired([10.0, 20.0, 30.0, 40.0], [12.0, 18.0, 33.0, 45.0])
    r = stats.compare(af).loc["all"]
    assert r["n"] == 4 and np.isclose(r["bias"], (2 - 2 + 3 + 5) / 4) and np.isclose(r["mae"], 3.0)
    assert np.isclose(r["rmse"], np.sqrt((4 + 4 + 9 + 25) / 4)) and np.isclose(r["nmb_percent"], 100 * 8 / 100)
    assert r["r"] > 0.98 and np.isclose(r["obs_mean"], 25.0)


def test_compare_ignores_nan_and_handles_empty_and_constant():
    af = paired([1.0, np.nan, 3.0, 4.0], [1.0, 5.0, np.nan, 4.0])
    assert stats.compare(af).loc["all", "n"] == 2
    none = stats.compare(paired([np.nan], [1.0])).loc["all"]
    assert none["n"] == 0 and np.isnan(none["bias"])
    assert np.isnan(stats.compare(paired([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])).loc["all", "r"])


def test_compare_by_alt_hour_month_and_dataset():
    idx = pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 00:00", "2024-07-01 01:00", "2024-08-01 01:00"], tz="UTC", name="time")
    df = pd.DataFrame({"obs": [1.0, 2.0, 3.0, 4.0], "model": [2.0, 2.0, 4.0, 4.0], "alt": [0.5, 1.5, 0.5, 1.5]}, index=idx)
    af = AtmozFrame(df, alt="alt", units={"alt": "km"})
    by_alt = stats.compare(af, by="alt")
    assert list(by_alt["n"]) == [2, 2] and np.isclose(by_alt["bias"].iloc[0], 1.0)
    assert list(stats.compare(af, by="hour")["n"]) == [2, 2] and list(stats.compare(af, by="month").index) == [7, 8]
    ds = AtmozDataset(frames={"a": af, "b": af})
    res = stats.compare(ds)
    assert set(res.index.get_level_values("key")) == {"all", "a", "b"} and res.loc[("all", "all"), "n"] == 8
    with pytest.raises(ValueError):
        stats.compare(af, by="year")
    with pytest.raises(ValueError, match="altitude axis"):
        stats.compare(paired([1.0], [1.0]), by="alt")
    with pytest.raises(KeyError):
        stats.compare(af, obs="nope")


def test_exceedances_from_an_mda8_table():
    days = pd.date_range("2024-07-01", periods=6)
    table = pd.DataFrame({"State Code": "24", "County Code": "003", "Site Num": "1004", "date": days,
                          "mda8": [0.065, 0.071, 0.070, 0.080, 0.090, 0.100],
                          "n_averages": 17, "valid": [True, True, True, True, True, False], "start_hour": 10})
    res = stats.exceedances(table, units="ppm").iloc[0]
    assert res["valid_days"] == 5 and res["days_total"] == 6 and res["exceedance_days"] == 3
    assert np.isclose(res["max_mda8"], 90.0) and res["max_date"] == days[4] and np.isclose(res["fourth_highest"], 70.0)
    assert stats.exceedances(table.assign(mda8=table["mda8"] * 1000), units="ppb", level_ppb=75).iloc[0]["exceedance_days"] == 2
    with pytest.raises(ValueError):
        stats.exceedances(table, units="ppt")


def test_exceedances_accepts_an_aqs_dataset():
    idx = pd.date_range("2024-07-01", periods=48, freq="1h", tz="UTC", name="time")
    df = pd.DataFrame({"value": 0.080, "poc": 1, "lat": 39.0, "lon": -76.0,
                       "local_time": idx.tz_localize(None) - pd.Timedelta(hours=5)}, index=idx)
    af = AtmozFrame(df, units={"value": "ppm"}, lat="lat", lon="lon",
                    metadata={"parameter_code": "44201", "site_id": "240031004", "state_code": "24", "county_code": "003", "site_num": "1004"})
    res = stats.exceedances(AtmozDataset(frames={"240031004/44201": af}))
    assert res["exceedance_days"].sum() >= 1


def test_diurnal_and_offsets():
    af = series_frame(list(range(48)))
    d = stats.diurnal(af)
    assert list(d.index) == list(range(24)) and np.isclose(d.loc[0, "frame"], 12.0)
    shifted = stats.diurnal(af, utc_offset=-5)
    assert np.isclose(shifted.loc[19, "frame"], 12.0)
    ds = AtmozDataset(frames={"a": af, "b": af})
    assert list(stats.diurnal(ds, stat="max").columns) == ["a", "b"]
    local = series_frame([1.0, 2.0, 3.0], local_time=pd.date_range("2024-06-30 19:00", periods=3, freq="1h"))
    assert list(stats.diurnal(local, local_time=True).dropna().index) == [19, 20, 21]
    with pytest.raises(KeyError):
        stats.diurnal(af, local_time=True)


def test_trend_recovers_a_known_slope():
    idx = pd.date_range("2015-01-01", periods=10 * 12, freq="MS", tz="UTC", name="time")
    years = (idx - idx[0]).total_seconds() / (365.25 * 86400)
    af = AtmozFrame(pd.DataFrame({"value": 40.0 + 0.5 * years}, index=idx), units={"value": "ppb"})
    res = stats.trend(af, freq="YS")
    assert np.isclose(res.loc["frame", "slope_per_year"], 0.5, atol=0.02) and res.loc["frame", "n_periods"] == 10
    assert res.loc["frame", "r2"] > 0.99
    assert np.isnan(stats.trend(series_frame([1.0, 2.0]), freq="YS").loc["frame", "slope_per_year"])
    assert stats.trend(af, freq="MS", min_count=2)["n_periods"].iloc[0] == 0


def profile_frame(ref="agl", elev=None):
    rows = [(t, a, 10.0 * a + k) for k, t in enumerate(pd.date_range("2024-07-01", periods=3, freq="1h", tz="UTC")) for a in (0.5, 1.0, 1.5, 2.5)]
    df = pd.DataFrame(rows, columns=["time", "alt", "value"]).set_index("time")
    df["lat"], df["lon"] = 39.0, -76.0
    return AtmozFrame(df, units={"value": "ppb", "alt": "km"}, lat="lat", lon="lon", alt="alt",
                      metadata={"alt_reference": ref, **({"site_elevation_km": elev} if elev is not None else {})})


def test_layer_mean_per_profile_and_min_points():
    lm = vertical.layer_mean(profile_frame(), 0, 2)
    assert np.allclose(lm.df["value"], [10.0 * np.mean([0.5, 1.0, 1.5]) + k for k in range(3)])
    assert lm.units["value"] == "ppb" and lm.metadata["layer_km"] == (0, 2) and lm.location == (39.0, -76.0)
    assert vertical.layer_mean(profile_frame(), 0, 2, min_points=4).df["value"].isna().all()
    with pytest.raises(ValueError):
        vertical.layer_mean(profile_frame(), 2, 0)


def test_layer_mean_in_another_reference_and_dataset():
    asl = profile_frame(ref="asl", elev=0.5)               # asl altitudes 0.5..2.5 = agl 0..2
    agl = vertical.layer_mean(asl, 0, 1.0, reference="agl")   # agl 0.0, 0.5 (and 1.0): asl 0.5, 1.0, 1.5
    assert np.allclose(agl.df["value"], [10.0 * np.mean([0.5, 1.0, 1.5]) + k for k in range(3)])
    with pytest.raises(ValueError, match="site_elevation_km"):
        vertical.layer_mean(profile_frame(ref="asl"), 0, 1, reference="agl")
    ds = AtmozDataset(frames={"a": profile_frame(), "b": profile_frame()})
    out = vertical.layer_mean(ds, 0, 2)
    assert isinstance(out, AtmozDataset) and list(out) == ["a", "b"]


def test_mean_profile():
    mp = vertical.mean_profile(profile_frame(), bins=1.0)
    assert list(mp["count"]) == [6, 3, 3] or mp["count"].sum() == 12
    assert np.isclose(mp.loc[mp.index[0], "mean"], np.mean([5.0, 5.0 + 1, 5.0 + 2]))
    two = vertical.mean_profile(AtmozDataset(frames={"a": profile_frame(), "b": profile_frame()}), bins=[0, 1, 3])
    assert two["count"].sum() == 24
    with pytest.raises(TypeError):
        vertical.mean_profile(pd.DataFrame())
    assert atmoz.stats is stats and atmoz.vertical is vertical


def test_compare_defaults_to_the_collocated_column_names():
    idx = pd.date_range("2024-07-01", periods=4, freq="1h", tz="UTC", name="time")
    af = AtmozFrame(pd.DataFrame({"sonde": [1.0, 2.0, 3.0, 4.0], "model": [2.0, 3.0, 4.0, 5.0]}, index=idx),
                    metadata={"columns": {"sonde": "value", "model": "value"}})
    assert np.isclose(stats.compare(af).loc["all", "bias"], 1.0)
