import numpy as np
import pandas as pd
import pytest

import atmoz
from atmoz import AtmozFrame


def profile_frame(times, alts, f, unit="ppb", lat=39.0, lon=-76.0):
    rows = [(t, a, f(t, a)) for t in pd.DatetimeIndex(times) for a in alts]
    df = pd.DataFrame(rows, columns=["time", "alt", "value"]).set_index("time")
    df["lat"], df["lon"] = lat, lon
    return AtmozFrame(df, units={"value": unit, "alt": "km"}, lat="lat", lon="lon", alt="alt")


def test_profile_interpolated_in_time_and_altitude():
    model_times = pd.date_range("2024-07-01 00:30", periods=3, freq="1h", tz="UTC")
    model = profile_frame(model_times, [0.0, 1.0, 2.0, 3.0], lambda t, a: 10 * a + (t.hour * 1.0))
    obs = profile_frame(pd.DatetimeIndex(["2024-07-01 01:00"], tz="UTC"), [0.5, 1.5, 2.5], lambda t, a: np.nan)
    out = atmoz.collocate(obs, model)
    # model at 00:30 is 10a + 0 and at 01:30 is 10a + 1; halfway in time: 10a + 0.5
    assert np.allclose(out.df["model"], [5.5, 15.5, 25.5])
    assert out.axes["alt"] == "alt" and out.units["obs"] == "ppb" and out.units["alt"] == "km"


def test_model_altitudes_may_differ_per_time():
    t0, t1 = pd.Timestamp("2024-07-01 00:00", tz="UTC"), pd.Timestamp("2024-07-01 01:00", tz="UTC")
    a = profile_frame([t0], [0.0, 2.0], lambda t, z: z)          # value = alt
    b = profile_frame([t1], [0.0, 4.0], lambda t, z: z)          # value = alt, on a different grid
    model = AtmozFrame(pd.concat([a.df, b.df]), units=a.units, lat="lat", lon="lon", alt="alt")
    obs = profile_frame(pd.DatetimeIndex([t0 + pd.Timedelta("30min")]), [1.0], lambda t, z: 0.0)
    assert np.allclose(atmoz.collocate(obs, model).df["model"], [1.0])


def test_outside_altitude_range_and_time_gap_are_nan():
    model_times = pd.date_range("2024-07-01 00:00", periods=2, freq="1h", tz="UTC")
    model = profile_frame(model_times, [0.0, 1.0], lambda t, a: a)
    obs = profile_frame(pd.DatetimeIndex(["2024-07-01 00:30", "2024-07-01 05:00"], tz="UTC"), [0.5, 5.0], lambda t, a: 1.0)
    out = atmoz.collocate(obs, model).df
    inside = out.loc["2024-07-01 00:30"]
    assert np.isclose(inside["model"].iloc[0], 0.5) and np.isnan(inside["model"].iloc[1])
    assert out.loc["2024-07-01 05:00", "model"].isna().all()


def test_max_gap_controls_tolerance():
    model = profile_frame(pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 06:00"], tz="UTC"), [0.0, 1.0], lambda t, a: a)
    obs = profile_frame(pd.DatetimeIndex(["2024-07-01 03:00"], tz="UTC"), [0.5], lambda t, a: 1.0)
    assert atmoz.collocate(obs, model).df["model"].isna().all()
    assert np.allclose(atmoz.collocate(obs, model, max_gap="6h").df["model"], [0.5])


def test_units_are_converted_to_the_observation_unit():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    model = profile_frame(t, [0.0, 1.0], lambda tt, a: 0.05, unit="ppm")
    obs = profile_frame(t, [0.5], lambda tt, a: 40.0, unit="ppb")
    assert np.allclose(atmoz.collocate(obs, model).df["model"], [50.0])


def test_surface_series():
    idx = pd.date_range("2024-07-01", periods=3, freq="1h", tz="UTC", name="time")
    model = AtmozFrame(pd.DataFrame({"value": [0.0, 10.0, 20.0]}, index=idx), units={"value": "ppb"})
    obs_idx = pd.DatetimeIndex(["2024-07-01 00:30", "2024-07-01 01:30"], tz="UTC", name="time")
    obs = AtmozFrame(pd.DataFrame({"value": [1.0, 2.0]}, index=obs_idx), units={"value": "ppb"})
    out = atmoz.collocate(obs, model)
    assert np.allclose(out.df["model"], [5.0, 15.0]) and np.allclose(out.df["obs"], [1.0, 2.0])


def test_bad_input_raises():
    idx = pd.date_range("2024-07-01", periods=2, freq="1h", tz="UTC")
    surface = AtmozFrame(pd.DataFrame({"value": [1.0, 2.0]}, index=idx))
    with pytest.raises(TypeError):
        atmoz.collocate(surface.df, surface)
    with pytest.raises(KeyError):
        atmoz.collocate(surface, surface, obs_col="nope")


def test_surface_obs_with_model_profile_uses_lowest_level_or_alt():
    idx = pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 01:00"], tz="UTC", name="time")
    model = profile_frame(idx, [0.1, 1.0, 2.0], lambda t, a: 10 * a)       # 1, 10, 20 at each time
    obs = AtmozFrame(pd.DataFrame({"value": [5.0, 6.0]}, index=idx), units={"value": "ppb"})
    low = atmoz.collocate(obs, model)
    assert np.allclose(low.df["model"], [1.0, 1.0]) and low.axes["alt"] is None
    assert low.metadata["model"]["reduced_to_surface"] == "lowest level"
    at = atmoz.collocate(obs, model, alt=1.5)
    assert np.allclose(at.df["model"], [15.0, 15.0])


def test_profile_obs_with_surface_model_is_reduced_to_the_obs_surface():
    idx = pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 01:00"], tz="UTC", name="time")
    obs = profile_frame(idx, [0.2, 1.0], lambda t, a: 100 * a)             # lowest level: 20
    model = AtmozFrame(pd.DataFrame({"value": [1.0, 3.0]}, index=idx), units={"value": "ppb"})
    out = atmoz.collocate(obs, model)
    assert np.allclose(out.df["obs"], [20.0, 20.0]) and np.allclose(out.df["model"], [1.0, 3.0])
    assert out.metadata["obs"]["reduced_to_surface"] == "lowest level"
