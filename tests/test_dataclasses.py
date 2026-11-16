import pickle

import numpy as np
import pandas as pd
import pytest

from atmoz import AtmozDataset, AtmozFrame


def make_df(n=6, start="2024-07-01", tz="UTC", **extra):
    idx = pd.date_range(start, periods=n, freq="30min", tz=tz, name="time")
    data = {"value": np.arange(n, dtype=float), "lat": 39.0, "lon": -76.0}
    data.update(extra)
    return pd.DataFrame(data, index=idx)


def make_frame(**kw):
    return AtmozFrame(make_df(**kw), units={"value": "ppb"}, lat="lat", lon="lon", metadata={"site": "A"})


def test_time_is_the_index():
    af = make_frame()
    assert isinstance(af.t, pd.DatetimeIndex) and str(af.t.tz) == "UTC"
    assert af.axes == {"t": "index", "lat": "lat", "lon": "lon", "alt": None}
    assert af.alt is None and (af.lat == 39.0).all()


def test_requires_datetimeindex_and_timezone():
    with pytest.raises(TypeError):
        AtmozFrame(pd.DataFrame({"value": [1.0]}))
    with pytest.raises(ValueError, match="timezone"):
        AtmozFrame(make_df(tz=None))


def test_other_timezones_become_utc():
    af = AtmozFrame(make_df(tz="America/New_York"))
    assert str(af.t.tz) == "UTC"


def test_missing_axis_or_unit_column_raises():
    with pytest.raises(KeyError):
        AtmozFrame(make_df(), alt="alt")
    with pytest.raises(KeyError):
        AtmozFrame(make_df(), units={"nope": "ppb"})


def test_ppbv_is_normalised_and_unknown_unit_warns():
    assert AtmozFrame(make_df(), units={"value": "ppbv"}).units == {"value": "ppb"}
    with pytest.warns(UserWarning):
        AtmozFrame(make_df(), units={"value": "furlongs"})


def test_metadata_is_kept_and_copied():
    meta = {"site": "A"}
    af = AtmozFrame(make_df(), metadata=meta)
    meta["site"] = "B"
    assert af.metadata == {"site": "A"}


def test_pandas_methods_keep_axes_units_metadata():
    out = make_frame().resample("1h").mean()
    assert isinstance(out, AtmozFrame)
    assert out.units == {"value": "ppb"} and out.metadata == {"site": "A"} and out.axes["lat"] == "lat"
    assert len(out) == 3


def test_dropping_axis_column_drops_the_axis():
    out = make_frame().drop(columns="lon")
    assert out.axes["lon"] is None and out.axes["lat"] == "lat"


def test_losing_the_time_index_returns_plain_pandas():
    assert isinstance(make_frame().reset_index(), pd.DataFrame)


def test_convert_units():
    af = AtmozFrame(make_df(), units={"value": "ppb"}).convert_units("value", "ppm")
    assert af.units["value"] == "ppm" and np.allclose(af.df["value"], np.arange(6) / 1000)


def test_with_helpers_and_pickle():
    af = make_frame().with_units(lat="deg").with_metadata(extra=1)
    assert af.units["lat"] == "deg" and af.metadata["extra"] == 1
    back = pickle.loads(pickle.dumps(af))
    assert back.units == af.units and back.df.equals(af.df)


def test_dataset_time_range_and_access():
    ds = AtmozDataset("t", "src", {"a": make_frame(), "b": make_frame(start="2024-07-02")}, metadata={"q": 1})
    assert list(ds) == ["a", "b"] and len(ds) == 2 and "a" in ds
    assert ds.time_start == pd.Timestamp("2024-07-01", tz="UTC")
    assert ds.time_end == pd.Timestamp("2024-07-02 02:30", tz="UTC")
    assert ds.metadata == {"q": 1} and "n_frames=2" in repr(ds)
    with pytest.raises(ValueError):
        ds.frame


def test_dataset_single_frame_shortcut_and_empty():
    assert AtmozDataset(frames={"a": make_frame()}).frame.metadata == {"site": "A"}
    empty = AtmozDataset()
    assert len(empty) == 0 and empty.time_start is None and empty.schema is None


def test_dataset_schema_must_match():
    other = AtmozFrame(make_df(), units={"value": "ppb"}, lat="lat")   # no lon axis
    with pytest.raises(ValueError, match="schema"):
        AtmozDataset(frames={"a": make_frame(), "b": other})
    ds = AtmozDataset(frames={"a": make_frame()})
    with pytest.raises(ValueError, match="schema"):
        ds["b"] = other
    with pytest.raises(TypeError):
        ds["c"] = make_df()


def test_dataset_select():
    ds = AtmozDataset(frames={"s1/OZONE": make_frame(), "s1/PM2.5": make_frame(), "s2/OZONE": make_frame()})
    assert list(ds.select("*/OZONE")) == ["s1/OZONE", "s2/OZONE"]
    assert len(ds.select("nothing*")) == 0


def test_dataset_broadcast_and_chained_ops():
    ds = AtmozDataset("t", "src", {"a": make_frame(), "b": make_frame()}, metadata={"q": 1})
    hourly = ds.resample("1h").mean()
    assert isinstance(hourly, AtmozDataset) and hourly.metadata == {"q": 1}
    assert all(len(f) == 3 for f in hourly.values())
    assert isinstance(ds.dropna(), AtmozDataset)
    with pytest.raises(AttributeError):
        ds.not_a_pandas_method


def test_dataset_pickle_and_delete():
    ds = AtmozDataset(frames={"a": make_frame(), "b": make_frame()})
    back = pickle.loads(pickle.dumps(ds))
    assert list(back) == ["a", "b"]
    del ds["a"]
    assert list(ds) == ["b"]


def test_units_may_differ_between_frames():
    pm = AtmozFrame(make_df(), units={"value": "µg/m3"}, lat="lat", lon="lon")
    ds = AtmozDataset(frames={"a/OZONE": make_frame(), "a/PM2.5": pm})
    assert ds["a/PM2.5"].units["value"] == "µg/m3" and ds["a/OZONE"].units["value"] == "ppb"
