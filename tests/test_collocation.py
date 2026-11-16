import numpy as np
import pandas as pd
import pytest

import atmoz
from atmoz import AtmozFrame


def profile_frame(times, alts, f, unit="ppb", lat=39.0, lon=-76.0, ref="agl", elevation=None):
    rows = [(t, a, f(t, a)) for t in pd.DatetimeIndex(times) for a in alts]
    df = pd.DataFrame(rows, columns=["time", "alt", "value"]).set_index("time")
    df["lat"], df["lon"] = lat, lon
    meta = {"alt_reference": ref, **({"site_elevation_km": elevation} if elevation is not None else {})}
    return AtmozFrame(df, units={"value": unit, "alt": "km"}, lat="lat", lon="lon", alt="alt", metadata=meta)


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
    model = AtmozFrame(pd.concat([a.df, b.df]), units=a.units, lat="lat", lon="lon", alt="alt", metadata=a.metadata)
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
        atmoz.collocate(surface, surface, cols="nope")


def test_surface_obs_with_model_profile_uses_lowest_level_or_alt():
    idx = pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 01:00"], tz="UTC", name="time")
    model = profile_frame(idx, [0.1, 1.0, 2.0], lambda t, a: 10 * a)       # 1, 10, 20 at each time
    obs = AtmozFrame(pd.DataFrame({"value": [5.0, 6.0]}, index=idx), units={"value": "ppb"})
    low = atmoz.collocate(obs, model)
    assert np.allclose(low.df["model"], [1.0, 1.0]) and low.axes["alt"] is None
    assert low.metadata["sources"]["model"]["reduced_to_surface"] == "lowest level"
    at = atmoz.collocate(obs, model, alt=1.5)
    assert np.allclose(at.df["model"], [15.0, 15.0])


def test_profile_obs_with_surface_model_is_reduced_to_the_obs_surface():
    idx = pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 01:00"], tz="UTC", name="time")
    obs = profile_frame(idx, [0.2, 1.0], lambda t, a: 100 * a)             # lowest level: 20
    model = AtmozFrame(pd.DataFrame({"value": [1.0, 3.0]}, index=idx), units={"value": "ppb"})
    out = atmoz.collocate(obs, model)
    assert np.allclose(out.df["obs"], [20.0, 20.0]) and np.allclose(out.df["model"], [1.0, 3.0])
    assert out.metadata["sources"]["obs"]["reduced_to_surface"] == "lowest level"


def test_asl_lidar_is_converted_to_agl_before_comparing():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    model = profile_frame(t, [0.0, 1.0, 2.0], lambda tt, a: 10 * a, ref="agl")                 # value = 10 * height above ground
    obs = profile_frame(t, [0.5, 1.5, 2.5], lambda tt, a: 0.0, ref="asl", elevation=0.5)      # ASL 0.5/1.5/2.5 = AGL 0/1/2
    out = atmoz.collocate(obs, model)
    assert np.allclose(out.df["alt"], [0.0, 1.0, 2.0]) and np.allclose(out.df["model"], [0.0, 10.0, 20.0])
    assert out.alt_reference == "agl"


def test_same_reference_needs_no_elevation_and_keeps_it():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    model = profile_frame(t, [0.0, 1.0], lambda tt, a: a, ref="asl")
    obs = profile_frame(t, [0.5], lambda tt, a: 0.0, ref="asl")
    out = atmoz.collocate(obs, model)
    assert np.allclose(out.df["model"], [0.5]) and out.alt_reference == "asl"


def test_unknown_reference_or_elevation_raises_unless_assumed():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    model = profile_frame(t, [0.0, 1.0], lambda tt, a: a, ref=None)
    obs = profile_frame(t, [0.5], lambda tt, a: 0.0, ref="asl", elevation=0.1)
    with pytest.raises(ValueError, match="not stated"):
        atmoz.collocate(obs, model)
    assert np.allclose(atmoz.collocate(obs, model, assume_same_reference=True).df["model"], [0.5])
    no_elev = profile_frame(t, [0.5], lambda tt, a: 0.0, ref="asl")
    agl_model = profile_frame(t, [0.0, 1.0], lambda tt, a: a, ref="agl")
    with pytest.raises(ValueError, match="site_elevation_km"):
        atmoz.collocate(no_elev, agl_model)


def test_with_alt_reference_round_trip():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    f = profile_frame(t, [1.0, 2.0], lambda tt, a: a, ref="asl", elevation=0.25)
    agl = f.with_alt_reference("agl")
    assert np.allclose(agl.df["alt"], [0.75, 1.75]) and agl.alt_reference == "agl"
    assert np.allclose(agl.with_alt_reference("asl").df["alt"], [1.0, 2.0])
    with pytest.raises(ValueError):
        f.with_alt_reference("msl")


# ---- multi-source, sondes, vertical methods, datasets ----------------------------------------------------------------

from atmoz import AtmozDataset


def sonde_frame(start, alts, f, ref="agl", lat=39.0, lon=-76.0, minutes_per_point=0.5):
    """One flight: every row has its own time (a trajectory), per-row lat/lon."""
    times = pd.DatetimeIndex([pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=minutes_per_point * i) for i in range(len(alts))],
                             name="time")
    df = pd.DataFrame({"alt": alts, "value": [f(a) for a in alts], "lat": lat, "lon": lon}, index=times)
    return AtmozFrame(df, units={"value": "ppb", "alt": "km"}, lat="lat", lon="lon", alt="alt", metadata={"alt_reference": ref})


def test_list_form_and_three_sources_with_names():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    lidar = profile_frame(t, [0.5, 1.5, 2.5], lambda tt, a: 1.0)
    model = profile_frame(t, [0.0, 1.0, 2.0, 3.0], lambda tt, a: 10 * a)
    sonde = sonde_frame("2024-07-01 00:00", np.arange(0.0, 3.01, 0.1), lambda a: 100 * a)
    out = atmoz.collocate([lidar, model, sonde], names=["lidar", "model", "sonde"], window="30min")
    assert list(out.df.columns) == ["alt", "lat", "lon", "lidar", "model", "sonde"]
    assert np.allclose(out.df["model"], [5.0, 15.0, 25.0])
    assert np.allclose(out.df["sonde"], [45.0, 145.0, 245.0], atol=1.0)       # bin means of 0.1 km points
    assert out.metadata["method"] == "auto" and set(out.metadata["sources"]) == {"lidar", "model", "sonde"}
    same = atmoz.collocate(lidar, model, sonde, names=["lidar", "model", "sonde"], window="30min")
    assert same.df.equals(out.df)


def test_trajectory_is_averaged_in_the_time_window_only():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    lidar = profile_frame(t, [0.5, 1.5], lambda tt, a: 1.0)
    sonde = sonde_frame("2024-07-01 00:00", np.arange(0.0, 2.01, 0.25), lambda a: 10 * a, minutes_per_point=10)
    near = atmoz.collocate(lidar, sonde, window="25min").df["model"].to_numpy()
    assert np.isfinite(near[0]) and np.isnan(near[1])           # only points within 25 min (alt <= 0.75 km) are used
    assert np.isfinite(atmoz.collocate(lidar, sonde, window="3h").df["model"]).all()


def test_auto_uses_bin_mean_for_finer_profiles_and_interp_for_coarser():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    coarse_anchor = profile_frame(t, [0.0, 1.0, 2.0], lambda tt, a: 0.0)
    fine = profile_frame(t, np.arange(0.0, 2.01, 0.1), lambda tt, a: 0.0 if a < 0.5 else 100.0)
    auto = atmoz.collocate(coarse_anchor, fine).df["model"].to_numpy()
    mean = atmoz.collocate(coarse_anchor, fine, method="mean").df["model"].to_numpy()
    interp = atmoz.collocate(coarse_anchor, fine, method="interp").df["model"].to_numpy()
    assert np.allclose(auto, mean, equal_nan=True) and mean[0] == 0.0 and mean[1] == 100.0
    assert np.isclose(interp[1], 100.0)
    fine_anchor = profile_frame(t, np.arange(0.0, 2.01, 0.1), lambda tt, a: 0.0)
    coarse = profile_frame(t, [0.0, 1.0, 2.0], lambda tt, a: 10 * a)
    assert np.allclose(atmoz.collocate(fine_anchor, coarse).df["model"], np.arange(0.0, 2.01, 0.1) * 10)


def test_bad_arguments():
    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC")
    p = profile_frame(t, [0.0, 1.0], lambda tt, a: a)
    with pytest.raises(ValueError, match="at least two"):
        atmoz.collocate([p])
    with pytest.raises(ValueError, match="method"):
        atmoz.collocate(p, p, method="nearest")
    with pytest.raises(ValueError, match="names"):
        atmoz.collocate(p, p, names=["a", "a"])


def site_dataset(sites, hours=3, ref="agl"):
    frames = {}
    for key, (lat, lon, start) in sites.items():
        idx = pd.date_range(start, periods=hours, freq="1h", tz="UTC", name="time")
        df = pd.DataFrame({"value": np.arange(hours, dtype=float), "lat": lat, "lon": lon}, index=idx)
        frames[key] = AtmozFrame(df, units={"value": "ppb"}, lat="lat", lon="lon")
    return AtmozDataset("x", "src", frames)


def test_datasets_pair_by_nearest_site_and_report_unpaired():
    surface = site_dataset({"near": (39.0, -76.5, "2024-07-01"), "far": (45.0, -100.0, "2024-07-01")})
    other = site_dataset({"A": (39.05, -76.5, "2024-07-01"), "B": (30.0, -90.0, "2024-07-01")})
    out = atmoz.collocate(surface, other)
    assert isinstance(out, AtmozDataset) and list(out) == ["near"]
    assert out.metadata["pairs"] == {"near": {"model": "A"}} and out.metadata["unpaired"] == {"far": ["model"]}
    assert np.allclose(out["near"].df["model"], [0.0, 1.0, 2.0])
    assert len(atmoz.collocate(surface, other, max_distance_km=2000.0)) == 2


def test_dataset_tie_between_files_at_one_site_prefers_time_overlap():
    surface = site_dataset({"s": (39.0, -76.5, "2024-07-02")})
    other = site_dataset({"old": (39.0, -76.5, "2024-07-01"), "new": (39.0, -76.5, "2024-07-02")}, hours=3)
    out = atmoz.collocate(surface, other)
    assert out.metadata["pairs"]["s"]["model"] == "new"


def test_frame_anchor_with_datasets_and_three_way():
    anchor = AtmozFrame(pd.DataFrame({"value": [1.0, 2.0], "lat": 39.0, "lon": -76.5},
                                     index=pd.date_range("2024-07-01", periods=2, freq="1h", tz="UTC", name="time")),
                        units={"value": "ppb"}, lat="lat", lon="lon")
    a = site_dataset({"A": (39.0, -76.5, "2024-07-01")})
    b = site_dataset({"B": (39.1, -76.5, "2024-07-01")})
    out = atmoz.collocate([anchor, a, b], names=["x", "a", "b"])
    assert isinstance(out, AtmozFrame) and list(out.df.columns)[-3:] == ["x", "a", "b"]
    with pytest.raises(ValueError, match="No frame within"):
        atmoz.collocate(anchor, site_dataset({"far": (10.0, 10.0, "2024-07-01")}))
