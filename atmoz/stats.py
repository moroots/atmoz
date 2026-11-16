"""Statistics for comparing, summarising and trending atmospheric data.

* :func:`compare`: bias, errors and correlation between two columns (usually the output of :func:`atmoz.collocate`),
  overall or by altitude bin, hour of day or month, and per site for a dataset.
* :func:`exceedances`: days above the ozone standard from daily MDA8 values, per site.
* :func:`diurnal`: the mean (or another statistic) by hour of day.
* :func:`trend`: period means (monthly, yearly, ...) and their linear trend.

All of them accept an :class:`~atmoz.AtmozFrame` or an :class:`~atmoz.AtmozDataset` (one result per frame) and return pandas objects.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozDataset import AtmozDataset
from atmoz.dataclasses.AtmozFrame import AtmozFrame

Data = Union[AtmozFrame, AtmozDataset]


def _frames(data: Data) -> Dict[str, AtmozFrame]:
    if isinstance(data, AtmozFrame):
        return {"frame": data}
    if isinstance(data, AtmozDataset):
        return dict(data.items())
    raise TypeError(f"expected an AtmozFrame or AtmozDataset, got {type(data).__name__}.")


def _metrics(obs: pd.Series, model: pd.Series) -> dict:
    ok = obs.notna() & model.notna()
    o, m = obs[ok].to_numpy(dtype=float), model[ok].to_numpy(dtype=float)
    n = len(o)
    if n == 0:
        return {"n": 0, "obs_mean": np.nan, "model_mean": np.nan, "bias": np.nan, "mae": np.nan, "rmse": np.nan,
                "nmb_percent": np.nan, "nme_percent": np.nan, "r": np.nan}
    diff = m - o
    obs_sum = o.sum()
    r = float(np.corrcoef(o, m)[0, 1]) if n > 1 and o.std() > 0 and m.std() > 0 else np.nan
    return {"n": n, "obs_mean": float(o.mean()), "model_mean": float(m.mean()), "bias": float(diff.mean()),
            "mae": float(np.abs(diff).mean()), "rmse": float(np.sqrt((diff ** 2).mean())),
            "nmb_percent": float(100 * diff.sum() / obs_sum) if obs_sum != 0 else np.nan,
            "nme_percent": float(100 * np.abs(diff).sum() / obs_sum) if obs_sum != 0 else np.nan, "r": r}


def compare(
    data: Data,
    obs: Optional[str] = None,
    model: Optional[str] = None,
    *,
    by: Optional[str] = None,
    bins: Union[float, Sequence[float], None] = None,
) -> pd.DataFrame:
    """Comparison statistics between two columns.

    Parameters
    ----------
    data:
        A frame (or dataset of frames) holding both columns, e.g. the result of ``atmoz.collocate``.
    obs, model:
        Column names; the statistics treat *obs* as the reference (bias = model - obs). Default: the first two sources of a
        collocated frame (``frame.metadata["columns"]``), else ``"obs"`` and ``"model"``.
    by:
        ``None`` (all rows), ``"alt"`` (altitude bins, needs the frame's ``alt`` axis), ``"hour"`` (UTC hour of day) or
        ``"month"``.
    bins:
        For ``by="alt"``: the bin width in km (default 1) or the bin edges.

    Returns
    -------
    DataFrame
        Columns ``n``, ``obs_mean``, ``model_mean``, ``bias``, ``mae``, ``rmse``, ``nmb_percent``, ``nme_percent`` (normalised
        mean bias / error) and ``r`` (Pearson). Index: the group (``"all"`` or the bin / hour / month); for a dataset a
        MultiIndex ``(key, group)`` with a pooled ``("all", group)`` block when it has several frames. Only rows with both
        values count.
    """
    if by not in (None, "alt", "hour", "month"):
        raise ValueError("by must be None, 'alt', 'hour' or 'month'.")
    frames = _frames(data)
    first = next(iter(frames.values()), None)
    named = list(first.metadata.get("columns", {})) if first is not None else []
    obs = obs or (named[0] if len(named) >= 2 else "obs")
    model = model or (named[1] if len(named) >= 2 else "model")
    out = {}

    def summarise(af: AtmozFrame, label: str) -> pd.DataFrame:
        for c in (obs, model):
            if c not in af.df.columns:
                raise KeyError(f"Column {c!r} is not in {label!r} (columns: {list(af.df.columns)}).")
        return _grouped(af.df[[obs, model]].assign(**_group_columns(af, by, bins)), obs, model, by)

    for key, af in frames.items():
        out[key] = summarise(af, key)
    if isinstance(data, AtmozFrame):
        return out["frame"]
    if len(frames) > 1:
        pooled = pd.concat([af.df[[obs, model]].assign(**_group_columns(af, by, bins)) for af in frames.values()])
        out = {"all": _grouped(pooled, obs, model, by), **out}
    if not out:
        return pd.DataFrame(columns=["n", "obs_mean", "model_mean", "bias", "mae", "rmse", "nmb_percent", "nme_percent", "r"])
    return pd.concat(out, names=["key", "group"])


def _group_columns(af: AtmozFrame, by: Optional[str], bins) -> dict:
    if by is None:
        return {}
    if by == "alt":
        if af.axes["alt"] is None:
            raise ValueError("by='alt' needs a frame with an altitude axis.")
        alt = af.df[af.axes["alt"]]
        if bins is None or np.isscalar(bins):
            width = 1.0 if bins is None else float(bins)
            top = float(np.nanmax(alt)) if len(alt) and alt.notna().any() else width
            edges = np.arange(0.0, top + width, width)
        else:
            edges = np.asarray(bins, dtype=float)
        return {"_group": pd.cut(alt, edges, right=False).to_numpy()}
    if by == "hour":
        return {"_group": af.df.index.hour}
    return {"_group": af.df.index.month}


def _grouped(df: pd.DataFrame, obs: str, model: str, by: Optional[str]) -> pd.DataFrame:
    if by is None:
        return pd.DataFrame([_metrics(df[obs], df[model])], index=pd.Index(["all"], name="group"))
    rows = {g: _metrics(sub[obs], sub[model]) for g, sub in df.groupby("_group", observed=True)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group") if rows else pd.DataFrame(
        columns=["n", "obs_mean", "model_mean", "bias", "mae", "rmse", "nmb_percent", "nme_percent", "r"])


# ---- exceedances ----------------------------------------------------------------------------------------------------------


def exceedances(
    mda8_values: Union[pd.DataFrame, Data],
    *,
    level_ppb: float = 70.0,
    units: str = "ppm",
    value: str = "mda8",
    date: str = "date",
    valid: Optional[str] = "valid",
) -> pd.DataFrame:
    """Per-site summary of daily MDA8 ozone against the standard.

    An exceedance is a valid day whose MDA8 is above *level_ppb* (EPA: MDA8 > 0.070 ppm, values truncated to whole ppb).

    Parameters
    ----------
    mda8_values:
        The table returned by :func:`atmoz.surface.mda8_aqs` (site columns, ``date``, ``mda8``, ``valid``, ...), or an AQS
        :class:`AtmozDataset`/:class:`AtmozFrame` (MDA8 is computed for you).
    level_ppb:
        The level of the standard (default 70 ppb).
    units:
        Units of the ``mda8`` column (``"ppm"`` for AQS, ``"ppb"`` otherwise). Ignored when MDA8 is computed here.

    Returns
    -------
    DataFrame
        One row per site (the non-result columns of the input, or ``key``): ``valid_days``, ``days_total``, ``exceedance_days``,
        ``max_mda8`` and ``max_date`` (in ppb), ``fourth_highest`` (ppb, the value EPA's design value is built from; NaN with
        fewer than four valid days) and ``first``/``last`` date.
    """
    if not isinstance(mda8_values, pd.DataFrame):
        from atmoz.surface.mda8 import mda8_aqs
        mda8_values, units = mda8_aqs(mda8_values, units="ppm"), "ppm"
    df = mda8_values.copy()
    factor = {"ppb": 1.0, "ppbv": 1.0, "ppm": 1000.0, "ppmv": 1000.0}.get(units.lower())
    if factor is None:
        raise ValueError("units must be 'ppb' or 'ppm'.")
    for c in (value, date):
        if c not in df.columns:
            raise KeyError(f"Column {c!r} not found; pass the table returned by mda8_aqs.")
    result_cols = {value, date, "n_averages", "start_hour", *(valid and [valid] or [])}
    sites = [c for c in df.columns if c not in result_cols]
    df["_ppb"] = df[value] * factor
    ok = df[valid].astype(bool) if valid and valid in df.columns else pd.Series(True, index=df.index)
    df["_ok"] = ok & df["_ppb"].notna()

    def summarise(g: pd.DataFrame) -> pd.Series:
        good = g[g["_ok"]]
        top = good["_ppb"].sort_values(ascending=False)
        return pd.Series({
            "valid_days": int(len(good)), "days_total": int(len(g)),
            "exceedance_days": int((good["_ppb"] > level_ppb).sum()),
            "max_mda8": float(top.iloc[0]) if len(top) else np.nan,
            "max_date": good.loc[good["_ppb"].idxmax(), date] if len(good) else pd.NaT,
            "fourth_highest": float(top.iloc[3]) if len(top) >= 4 else np.nan,
            "first": g[date].min(), "last": g[date].max()})

    if not sites:
        return summarise(df).to_frame().T.rename_axis(None)
    return df.groupby(sites, sort=True).apply(summarise, include_groups=False).reset_index()


# ---- diurnal cycle and trends ---------------------------------------------------------------------------------------------


def diurnal(
    data: Data,
    col: str = "value",
    *,
    stat: str = "mean",
    utc_offset: Optional[float] = None,
    local_time: bool = False,
) -> pd.DataFrame:
    """The statistic of *col* by hour of day (0-23), one column per frame.

    Parameters
    ----------
    stat:
        Any pandas aggregation name: ``"mean"`` (default), ``"median"``, ``"max"``, ...
    utc_offset:
        Hours from UTC of the local **standard** time to use (e.g. ``-5`` for US Eastern); default is UTC hours.
    local_time:
        Use the frame's ``local_time`` column (AQS) for the hour instead of the UTC index.
    """
    frames = _frames(data)
    cols = {}
    for key, af in frames.items():
        if col not in af.df.columns:
            raise KeyError(f"Column {col!r} is not in {key!r}.")
        if local_time:
            if "local_time" not in af.df.columns:
                raise KeyError(f"{key!r} has no 'local_time' column.")
            hours = pd.DatetimeIndex(af.df["local_time"]).hour
        elif utc_offset is not None:
            hours = (af.df.index + pd.Timedelta(hours=utc_offset)).hour
        else:
            hours = af.df.index.hour
        cols[key] = af.df[col].groupby(hours).agg(stat)
    out = pd.DataFrame(cols).reindex(range(24))
    out.index.name = "hour"
    return out


def trend(
    data: Data,
    col: str = "value",
    *,
    freq: str = "YS",
    agg: str = "mean",
    min_count: int = 1,
) -> pd.DataFrame:
    """Period statistics and their ordinary least-squares linear trend, one row per frame.

    Parameters
    ----------
    freq:
        pandas period frequency of the aggregation: ``"MS"`` monthly, ``"YS"`` yearly (default), ...
    agg:
        Aggregation within a period (``"mean"``, ``"median"``, ``"max"``, ...).
    min_count:
        Periods with fewer valid values are dropped.

    Returns
    -------
    DataFrame
        Indexed by key: ``slope_per_year`` (change of the period value per year, in the column's unit), ``intercept`` (at the
        first period), ``r2``, ``n_periods``, ``start`` and ``end``. NaN slope with fewer than three periods. The period values
        themselves are ``frame.df[col].resample(freq).agg(agg)`` (``atmoz`` frames pass pandas methods through).
    """
    rows = {}
    for key, af in _frames(data).items():
        if col not in af.df.columns:
            raise KeyError(f"Column {col!r} is not in {key!r}.")
        s = af.df[col]
        counts = s.resample(freq).count()
        per = s.resample(freq).agg(agg)[counts >= min_count].dropna()
        row = {"slope_per_year": np.nan, "intercept": np.nan, "r2": np.nan, "n_periods": int(len(per)),
               "start": per.index.min() if len(per) else pd.NaT, "end": per.index.max() if len(per) else pd.NaT}
        if len(per) >= 3:
            years = (per.index - per.index[0]).total_seconds().to_numpy() / (365.25 * 86400)
            slope, intercept = np.polyfit(years, per.to_numpy(dtype=float), 1)
            fit = slope * years + intercept
            ss_res, ss_tot = float(((per.to_numpy() - fit) ** 2).sum()), float(((per.to_numpy() - per.mean()) ** 2).sum())
            row.update(slope_per_year=float(slope), intercept=float(intercept), r2=1 - ss_res / ss_tot if ss_tot else np.nan)
        rows[key] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("key") if rows else pd.DataFrame(
        columns=["slope_per_year", "intercept", "r2", "n_periods", "start", "end"])
