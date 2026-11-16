"""MDA8 against the rules of 40 CFR 50 Appendix U (docs/references/epa-40cfr50-appendix-u)."""
import numpy as np
import pandas as pd
import pytest

from atmoz.surface.mda8 import combine_collocated, mda8, mda8_aqs


def hours(day, values, start=0):
    idx = pd.date_range(f"{day} {start:02d}:00", periods=len(values), freq="h")
    return pd.Series(values, index=idx, dtype=float)


def day_series(value=50.0, day="2024-06-01"):
    """48 hours covering the day and the next morning, so all 17 windows are complete."""
    return hours(day, [value] * 48)


def test_constant_ozone_gives_that_value_with_all_17_averages():
    r = mda8(day_series(50)).loc["2024-06-01"]
    assert r.mda8 == 50 and r.n_averages == 17 and r.valid


def test_hourly_values_and_averages_are_truncated_not_rounded():
    # 70.9 truncates to 70 each hour (3(a)); the average of 70s is 70, not 71
    assert mda8(day_series(70.9)).loc["2024-06-01", "mda8"] == 70
    # an 8-hour average of 8 hours summing 75 = 9.375 truncates to 9 (3(b))
    x = hours("2024-06-01", [0.0] * 7 + [9.0] * 7 + [3.0] + [0.0] * 33)
    assert mda8(x).loc["2024-06-01", "mda8"] == 8                    # 66/8 = 8.25 -> 8


def test_the_max_is_the_highest_of_the_windows_starting_7_to_23():
    x = day_series(40)
    x["2024-06-01 13:00":"2024-06-01 20:00"] = 100.0                   # a clean 8-hour block starting 13:00
    r = mda8(x).loc["2024-06-01"]
    assert r.mda8 == 100 and r.start_hour == 13


def test_windows_starting_before_7am_are_ignored():
    x = day_series(40)
    x["2024-06-01 00:00":"2024-06-01 06:00"] = 200.0                   # only affects the window from 00:00..06:00 (start < 7)
    assert mda8(x).loc["2024-06-01", "mda8"] < 200


def test_last_window_runs_into_the_next_day():
    x = hours("2024-06-01", [0.0] * 48)
    x["2024-06-01 23:00":"2024-06-02 06:00"] = 90.0
    r = mda8(x)
    assert r.loc["2024-06-01", "mda8"] == 90 and r.loc["2024-06-01", "start_hour"] == 23


def test_six_or_seven_hours_use_that_many_as_the_divisor():
    x = hours("2024-06-01", [np.nan] * 7 + [60.0] * 6 + [np.nan] * 2 + [np.nan] * 33)   # 6 hours 07:00..12:00, then gap
    r = mda8(x).loc["2024-06-01"]
    assert r.mda8 == 60                                                # 360 / 6, not 360 / 8


def test_five_or_fewer_hours_valid_only_above_567_sum():
    low = hours("2024-06-01", [np.nan] * 7 + [100.0] * 5 + [np.nan] * 36)    # sum 500 <= 567 -> no valid average
    assert np.isnan(mda8(low).loc["2024-06-01", "mda8"])
    high = hours("2024-06-01", [np.nan] * 7 + [120.0] * 5 + [np.nan] * 36)   # sum 600 > 567 -> 600/8 = 75
    r = mda8(high).loc["2024-06-01"]
    assert r.mda8 == 75 and r.valid                                    # valid although < 13 averages (3(d): above the level)


def test_day_validity_needs_13_of_17_or_a_value_above_the_level():
    x = day_series(50)
    x["2024-06-01 13:00":"2024-06-02 06:00"] = np.nan                  # kills the windows starting 7..: most of the 17
    r = mda8(x).loc["2024-06-01"]
    assert r.n_averages < 13 and not r.valid
    y = x.copy()
    y["2024-06-01 07:00":"2024-06-01 12:00"] = 150.0                   # 6 hours, average 150 > the level: valid despite < 13
    r = mda8(y).loc["2024-06-01"]
    assert r.n_averages < 13 and r.valid and r.mda8 == 150


def test_exceptional_event_hours_are_missing_for_the_average_but_count_for_completeness():
    x = day_series(50)
    x["2024-06-01 12:00":"2024-06-01 13:00"] = 150.0
    ee = pd.Series(False, index=x.index)
    ee["2024-06-01 12:00":"2024-06-01 13:00"] = True
    with_events = mda8(x).loc["2024-06-01"]
    flagged = mda8(x, exceptional_events=ee).loc["2024-06-01"]
    assert flagged.mda8 < with_events.mda8                             # flagged hours do not raise the average
    assert flagged.n_averages == 17 and flagged.valid                  # but they still count toward the 13 of 17


def test_utc_input_needs_an_offset_and_shifts_to_local_standard_time():
    x = day_series(40)
    x.index = x.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="utc_offset"):
        mda8(x)
    peak = day_series(40)
    peak["2024-06-01 17:00":"2024-06-02 00:00"] = 80.0                # 17:00 UTC = 12:00 at UTC-5
    peak.index = peak.index.tz_localize("UTC")
    r = mda8(peak, utc_offset=-5)
    assert r.loc["2024-06-01", "start_hour"] == 12 and r.loc["2024-06-01", "mda8"] == 80


def test_ppm_in_ppm_out_with_truncation_to_three_decimals():
    r = mda8(day_series(0.0705), units="ppm").loc["2024-06-01"]
    assert r.mda8 == pytest.approx(0.070)                              # 0.0705 ppm -> 0.070 (truncated)
    with pytest.raises(ValueError, match="units"):
        mda8(day_series(), units="ppt")


def test_rejects_non_hourly_timestamps_and_non_series():
    bad = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-06-01 00:30", "2024-06-01 01:30"]))
    with pytest.raises(ValueError, match="on the hour"):
        mda8(bad)
    with pytest.raises(TypeError):
        mda8(bad.to_frame())


def test_collocated_monitors_primary_then_secondary_fill():
    t = pd.date_range("2024-06-01", periods=3, freq="h")
    df = pd.DataFrame({"POC": [1, 1, 2, 2, 2, 3], "t": [t[0], t[1], t[0], t[1], t[2], t[2]],
                       "v": [10.0, np.nan, 99.0, 20.0, 30.0, 50.0]})
    out = combine_collocated(df, time="t", value="v")
    assert out.tolist() == [10.0, 20.0, 40.0]                          # primary 10; gap -> 20; t2 -> mean(30, 50)
    out2 = combine_collocated(df, time="t", value="v", primary=2)
    assert out2.iloc[0] == 99.0


def test_mda8_aqs_runs_per_site_on_an_aqs_style_frame():
    t = pd.date_range("2024-06-01", periods=48, freq="h")
    frames = []
    for site, level in (("0001", 0.040), ("0002", 0.060)):
        frames.append(pd.DataFrame({"State Code": "24", "County Code": "031", "Site Num": site, "POC": 1,
                                    "Date Local": t.normalize().astype("datetime64[ns]"),
                                    "Time Local": t.strftime("%H:%M"), "Sample Measurement": level}))
    out = mda8_aqs(pd.concat(frames, ignore_index=True))
    day1 = out[out["date"] == "2024-06-01"].set_index("Site Num")
    assert day1["mda8"].to_dict() == pytest.approx({"0001": 0.040, "0002": 0.060})
    assert day1["valid"].all() and list(out.columns[:3]) == ["State Code", "County Code", "Site Num"]
