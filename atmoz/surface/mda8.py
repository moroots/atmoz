"""Ozone 8-hour averages and the daily maximum 8-hour average (MDA8), as defined by the EPA.

Implements 40 CFR Part 50, Appendix U, sections 2(b) and 3(a)-(d) (the spec is in ``docs/references/epa-40cfr50-appendix-u/``):

* hourly values are truncated to 3 decimals in **ppm** (= whole ppb) [3(a)];
* an 8-hour average is stored at its **start hour**; valid with at least 6 of 8 hours (divisor 6, 7 or 8, the hours available);
  with 5 or fewer hours it is still valid if the sum of the available hours is above 0.567 ppm; truncated to 3 decimals [3(b)];
* MDA8 = the highest of the 17 averages starting 07:00 to 23:00 local standard time (the last one runs into the next day) [3(c)];
* a day is valid with at least 13 of those 17 averages, or if its MDA8 is above the 0.070 ppm level [3(d)];
* exceptional-event hours count as missing for the averages but count as present for the completeness test [3(b), 3(d)];
* collocated monitors are combined into one site record: a primary monitor, gaps filled by the mean of the others [2(b)].

Hours are **hour-beginning, local standard time** (no daylight saving), which is how EPA AQS reports ``Time Local``.
Everything is computed in whole ppb so that truncation is exact (0.070 ppm = 70 ppb).

>>> mda8(hourly_ppb_series)                                   # naive index = already local standard time
>>> mda8(hourly_ppb_series_utc, utc_offset=-5)                # UTC index at a site in EST
>>> mda8_aqs(ds.data["hourly_44201_2024.csv"])                # EPA AQS hourly file, every site at once
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

LEVEL_PPB = 70.0          # the NAAQS level, 0.070 ppm [4(a)]
SUM_THRESHOLD_PPB = 567.0  # 3(b): with <= 5 hours, valid if the sum of available hours exceeds 0.567 ppm
_EPS = 1e-9

_UNIT_TO_PPB = {"ppb": 1.0, "ppbv": 1.0, "ppm": 1000.0, "ppmv": 1000.0}


def _trunc(x):
    """Truncate (not round) to whole ppb, i.e. 3 decimals in ppm. The epsilon guards against 29.999999999999996."""
    return np.floor(np.asarray(x, dtype=float) + _EPS)


def _to_ppb(values: pd.Series, units: str) -> pd.Series:
    try:
        factor = _UNIT_TO_PPB[units.lower()]
    except KeyError:
        raise ValueError(f"units must be one of {sorted(_UNIT_TO_PPB)}, got {units!r}") from None
    return values.astype(float) * factor


def _hourly_local(series: pd.Series, utc_offset: Optional[float]) -> pd.Series:
    """A regular hourly Series (naive local standard time, hour-beginning) with duplicates averaged."""
    s = series.copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        if utc_offset is None:
            raise ValueError("The index is timezone-aware: pass utc_offset (hours from UTC of the site's local STANDARD time, "
                             "e.g. -5 for US Eastern) so that hours are local standard time, as EPA requires. "
                             "Daylight saving time must not be used.")
        idx = idx.tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=utc_offset)
    elif utc_offset is not None:
        idx = idx + pd.Timedelta(hours=utc_offset)   # naive index given as UTC
    s.index = idx
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    bad = s.index[(s.index.minute != 0) | (s.index.second != 0)]
    if len(bad):
        raise ValueError(f"MDA8 needs hourly data on the hour; found {bad[0]} (resample or round it first).")
    return s.groupby(level=0).mean()


def _window(x: pd.Series, n: int = 8) -> pd.DataFrame:
    """For every hour t: the sum and count of the n hours starting at t (t .. t+n-1)."""
    present = x.notna().astype(float)
    filled = x.fillna(0.0)
    total = sum(filled.shift(-k, fill_value=0.0) for k in range(n))
    count = sum(present.shift(-k, fill_value=0.0) for k in range(n))
    return pd.DataFrame({"sum": total, "count": count})


def _averages(hourly: pd.Series) -> pd.DataFrame:
    """8-hour averages stored at their start hour: columns ``avg`` (truncated ppb, NaN if invalid) and ``count``."""
    w = _window(hourly)
    divisor = w["count"].where(w["count"].isin([6, 7]), 8.0)   # 6 or 7 hours: that many; 8: eight; <= 5: zeros substituted
    valid = (w["count"] >= 6) | ((w["count"] <= 5) & (w["sum"] > SUM_THRESHOLD_PPB))
    avg = pd.Series(_trunc(w["sum"] / divisor), index=w.index).where(valid)
    return pd.DataFrame({"avg": avg, "count": w["count"]})


def mda8(
    data: pd.Series,
    *,
    units: str = "ppb",
    utc_offset: Optional[float] = None,
    exceptional_events: Optional[pd.Series] = None,
    level_ppb: float = LEVEL_PPB,
) -> pd.DataFrame:
    """Daily maximum 8-hour average ozone for one site.

    Parameters
    ----------
    data:
        Hourly ozone, indexed by the **start** of each hour. A naive index is taken as local standard time. For a
        timezone-aware index (e.g. UTC) pass *utc_offset*. Missing hours may be absent or NaN. Several readings for one
        hour (collocated monitors) are averaged; use :func:`mda8_aqs` to apply EPA's primary/secondary rule instead.
    units:
        Units of *data*: ``"ppb"`` (default, AirNow API) or ``"ppm"`` (EPA AQS ozone files). The result is in the same units.
    utc_offset:
        Hours between UTC and the site's local **standard** time (e.g. ``-5`` for US Eastern, whatever the season).
    exceptional_events:
        Optional boolean Series (same index): hours EPA has approved as exceptional events. They are treated as missing for
        the averages but count toward the 13-of-17 completeness test, as 3(b) and 3(d) require.
    level_ppb:
        The level of the standard (default 70 ppb) used by the "valid if above the level" rules.

    Returns
    -------
    DataFrame
        One row per local calendar day that has any data (index ``date``, midnight timestamps), with columns

        * ``mda8``: the daily maximum 8-hour average in the units of *data*, truncated (NaN when no average is valid);
        * ``n_averages``: how many of the 17 candidate averages are valid (exceptional-event hours counted as present);
        * ``valid``: whether the day meets 3(d) (``n_averages >= 13`` or ``mda8`` above the level);
        * ``start_hour``: the start hour (7 to 23) of the 8-hour period that gave the maximum.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("data must be a pandas Series of hourly ozone indexed by time (use mda8_aqs for AQS DataFrames)")
    scale = _UNIT_TO_PPB.get(units.lower())
    ppb = _to_ppb(data, units)
    hourly = _hourly_local(ppb, utc_offset)
    hourly = pd.Series(_trunc(hourly.to_numpy()), index=hourly.index)                # 3(a): NaN stays NaN
    full = pd.date_range(hourly.index.min().normalize(), hourly.index.max().normalize() + pd.Timedelta(days=2), freq="h")
    hourly = hourly.reindex(full)
    ee = None
    if exceptional_events is not None:
        ee = _hourly_local(exceptional_events.astype(float), utc_offset).reindex(full).fillna(0).astype(bool)

    averages = _averages(hourly.mask(ee) if ee is not None else hourly)           # for the maximum
    for_count = _averages(hourly) if ee is not None else averages                  # 3(d): event hours count as present

    starts = averages.index[(averages.index.hour >= 7)]
    avg = averages.loc[starts, "avg"]
    counted = for_count.loc[starts, "avg"].notna()
    days = starts.normalize()
    out = pd.DataFrame({
        "mda8": avg.groupby(days).max(),
        "n_averages": counted.groupby(days).sum().astype(int),
    })
    best = avg.dropna()
    first_max = best.groupby(best.index.normalize()).idxmax() if len(best) else pd.Series(dtype="datetime64[ns]")
    out["start_hour"] = pd.Series(pd.DatetimeIndex(first_max).hour, index=first_max.index)
    out["valid"] = (out["n_averages"] >= 13) | (out["mda8"] > level_ppb)

    has_data = hourly.notna().groupby(hourly.index.normalize()).any()               # 3(c): days with monitoring data
    out = out[has_data.reindex(out.index).fillna(False)]
    out["mda8"] = out["mda8"] / scale
    out["start_hour"] = out["start_hour"].astype("Int64")
    out.index.name = "date"
    return out[["mda8", "n_averages", "valid", "start_hour"]]


def combine_collocated(df: pd.DataFrame, *, poc: str = "POC", time: str, value: str,
                       primary: Optional[Union[int, Sequence[int]]] = None) -> pd.Series:
    """EPA 2(b): one site record from collocated monitors (POCs). Returns a Series indexed by ``time``.

    The *primary* POC's values are used; hours it lacks are filled with the mean of the other monitors that have a value.
    *primary* defaults to the lowest POC number (EPA's agencies designate it; pass it if you know it).
    """
    wide = df.pivot_table(index=time, columns=poc, values=value, aggfunc="mean")
    if wide.shape[1] == 1:
        return wide.iloc[:, 0]
    first = primary if primary is not None else wide.columns.min()
    first = [first] if np.isscalar(first) else list(first)
    lead = wide[first].mean(axis=1) if len(first) > 1 else wide[first[0]]
    others = wide.drop(columns=first)
    return lead.where(lead.notna(), others.mean(axis=1))


def mda8_aqs(
    df: pd.DataFrame,
    *,
    value: str = "Sample Measurement",
    date: str = "Date Local",
    hour: str = "Time Local",
    site: Sequence[str] = ("State Code", "County Code", "Site Num"),
    poc: Optional[str] = "POC",
    units: str = "ppm",
    primary: Optional[dict] = None,
    level_ppb: float = LEVEL_PPB,
) -> pd.DataFrame:
    """MDA8 for every site of an EPA AQS hourly ozone file (``EPA_PreGen`` output, parameter 44201).

    ``Date Local`` / ``Time Local`` are local standard time, hour-beginning, as the rules need. Collocated monitors (POCs) at
    a site are combined per EPA 2(b) (see :func:`combine_collocated`). *primary* optionally maps a site tuple to its
    primary POC, e.g. ``{("24", "031", "3001"): 1}``.

    Returns a DataFrame with the *site* columns followed by the columns of :func:`mda8` (``date``, ``mda8`` in *units*,
    ``n_averages``, ``valid``, ``start_hour``), one row per site and day.
    """
    day = df[date] if pd.api.types.is_datetime64_any_dtype(df[date]) else pd.to_datetime(df[date].astype(str).str.strip())
    clock = pd.to_datetime(df[hour].astype(str).str.strip(), format="%H:%M")
    stamp = day.dt.normalize() + (clock - clock.dt.normalize())
    work = df[list(site) + ([poc] if poc and poc in df else []) + [value]].copy()
    work["_time"] = stamp
    rows = []
    for key, grp in work.groupby(list(site), observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        series = (combine_collocated(grp, poc=poc, time="_time", value=value, primary=(primary or {}).get(key))
                  if poc and poc in grp else grp.set_index("_time")[value])
        res = mda8(series, units=units, level_ppb=level_ppb).reset_index()
        for col, val in zip(site, key):
            res[col] = val
        rows.append(res)
    if not rows:
        return pd.DataFrame(columns=[*site, "date", "mda8", "n_averages", "valid", "start_hour"])
    return pd.concat(rows, ignore_index=True)[[*site, "date", "mda8", "n_averages", "valid", "start_hour"]]
