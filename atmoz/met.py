"""Meteorology on AtmozFrames: potential temperature, dewpoint, wind components, PBL height and the Skew-T plot.

Needs the optional MetPy dependency (``pip install "atmoz[met]"``), imported only when a function is called. Every
function takes and returns :class:`~atmoz.AtmozFrame` objects, reads each input column in the unit the frame declares
in ``frame.units`` (a column without a declared unit raises ``ValueError``) and declares the unit of every new column.

>>> import atmoz as az
>>> frame = az.Sonde.read("flight.ict").frame
>>> frame = az.met.theta(frame)                    # adds a "theta" column in K
>>> pbl = az.met.pbl_height(frame, group=None)     # AtmozFrame with a "pbl_height" column in km (agl)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozFrame import AtmozFrame
from atmoz.plotting.core import styled

__all__ = ["theta", "dewpoint", "wind_components", "wind_speed_direction", "pbl_height", "skewt"]

_PINT = {"%": "percent", "°C": "degC", "1": "dimensionless", "deg": "degree", "µg": "microgram"}
_NEEDS_METPY = "atmoz.met needs MetPy, which is not installed. Install it with:  pip install 'atmoz[met]'"


def _metpy():
    """The MetPy ``units`` registry and ``calc`` module, or a helpful ImportError."""
    try:
        import metpy.calc as calc
        from metpy.units import units
    except ImportError as err:
        raise ImportError(_NEEDS_METPY) from err
    return units, calc


def _quantity(frame: AtmozFrame, col: str, units):
    """The column as a pint Quantity in the unit the frame declares for it."""
    if col not in frame.df.columns:
        raise KeyError(f"Column {col!r} is not in the frame (columns: {list(frame.df.columns)}).")
    unit = frame.units.get(col)
    if unit is None:
        raise ValueError(f"Column {col!r} has no declared unit; set it with frame.with_units({col}='...') first.")
    return units.Quantity(frame.df[col].to_numpy(dtype=float), _PINT.get(unit, unit))


def _with_columns(frame: AtmozFrame, columns: dict, unit_map: dict) -> AtmozFrame:
    df = frame.df.copy()
    for name, values in columns.items():
        df[name] = np.asarray(values, dtype=float)
    ax = frame.axes
    return AtmozFrame(df, units={**frame.units, **unit_map}, lat=ax["lat"], lon=ax["lon"], alt=ax["alt"],
                      metadata=frame.metadata)


def theta(frame: AtmozFrame, *, temperature: str = "Temp_degC", pressure: str = "Pressure_hPa",
          out: str = "theta") -> AtmozFrame:
    """Add potential temperature (referenced to 1000 hPa).

    Parameters
    ----------
    frame : AtmozFrame
        Needs temperature and pressure columns with declared units.
    temperature, pressure : str
        Column names.
    out : str
        Name of the new column, declared in K.

    Returns
    -------
    AtmozFrame
        A copy of *frame* with the new column.

    Examples
    --------
    >>> frame = az.met.theta(sonde_frame)
    >>> frame.units["theta"]
    'K'
    """
    units, calc = _metpy()
    p, t = _quantity(frame, pressure, units), _quantity(frame, temperature, units)
    values = calc.potential_temperature(p, t).to("K").magnitude
    return _with_columns(frame, {out: values}, {out: "K"})


def dewpoint(frame: AtmozFrame, *, temperature: str = "Temp_degC", rh: str = "RH_percent",
             out: str = "dewpoint") -> AtmozFrame:
    """Add the dewpoint temperature from temperature and relative humidity.

    Parameters
    ----------
    frame : AtmozFrame
    temperature : str
        Temperature column (any temperature unit).
    rh : str
        Relative humidity column, declared in ``%`` or as a fraction (``1``).
    out : str
        Name of the new column, declared in degC.

    Returns
    -------
    AtmozFrame
        A copy of *frame* with the new column. Rows with RH of 0 or less give NaN.

    Examples
    --------
    >>> frame = az.met.dewpoint(sonde_frame)    # equals the temperature where RH is 100 %
    """
    units, calc = _metpy()
    t, h = _quantity(frame, temperature, units), _quantity(frame, rh, units)
    if not h.check("[]"):
        raise ValueError(f"Column {rh!r} must be a relative humidity in % or as a fraction, not {frame.units[rh]!r}.")
    h = h.to("dimensionless")
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(h.magnitude > 0, calc.dewpoint_from_relative_humidity(t, h).to("degC").magnitude, np.nan)
    return _with_columns(frame, {out: values}, {out: "°C"})


def wind_components(frame: AtmozFrame, *, speed: str = "WindSpeed_mps", direction: str = "WindDirection_deg",
                    out: tuple[str, str] = ("u", "v")) -> AtmozFrame:
    """Add the eastward (u) and northward (v) wind components.

    *direction* is the meteorological direction the wind blows from, in degrees clockwise from north, so a wind from
    the west (270) has a positive u.

    Parameters
    ----------
    frame : AtmozFrame
    speed, direction : str
        Wind speed (any speed unit) and direction (deg) columns.
    out : (str, str)
        Names of the new columns, both declared in m/s.

    Returns
    -------
    AtmozFrame

    Examples
    --------
    >>> frame = az.met.wind_components(sonde_frame)     # adds "u" and "v" in m/s
    """
    units, calc = _metpy()
    s, d = _quantity(frame, speed, units), _quantity(frame, direction, units)
    u, v = calc.wind_components(s, d)
    return _with_columns(frame, {out[0]: u.to("m/s").magnitude, out[1]: v.to("m/s").magnitude},
                         {out[0]: "m/s", out[1]: "m/s"})


def wind_speed_direction(frame: AtmozFrame, u: str = "u", v: str = "v",
                         out: tuple[str, str] = ("speed", "direction")) -> AtmozFrame:
    """Add wind speed (m/s) and meteorological direction (deg, the direction the wind blows from) from u and v.

    Parameters
    ----------
    frame : AtmozFrame
    u, v : str
        Eastward and northward component columns (any speed unit).
    out : (str, str)
        Names of the new columns, declared in m/s and deg. North is 360 (MetPy's convention); calm wind (u = v = 0) has direction 0.

    Returns
    -------
    AtmozFrame

    Examples
    --------
    >>> frame = az.met.wind_speed_direction(frame, u="u", v="v")
    """
    units, calc = _metpy()
    qu, qv = _quantity(frame, u, units), _quantity(frame, v, units)
    speed = calc.wind_speed(qu, qv).to("m/s").magnitude
    direction = calc.wind_direction(qu, qv, convention="from").to("deg").magnitude
    return _with_columns(frame, {out[0]: speed, out[1]: direction}, {out[0]: "m/s", out[1]: "deg"})


def _profile_pbl(alt: np.ndarray, th: np.ndarray, threshold: float, min_levels: int) -> float:
    """Parcel method on one profile (alt km, theta K): where theta first exceeds the lowest level's by threshold."""
    ok = np.isfinite(alt) & np.isfinite(th)
    alt, th = alt[ok], th[ok]
    if alt.size < min_levels:
        return np.nan
    order = np.argsort(alt, kind="stable")
    alt, th = alt[order], th[order]
    above = np.nonzero(th[1:] > th[0] + threshold)[0]
    if above.size == 0:
        return np.nan
    i = above[0] + 1                                  # first level warmer than the threshold
    t0, t1, z0, z1 = th[i - 1], th[i], alt[i - 1], alt[i]
    return float(z0 + (th[0] + threshold - t0) / (t1 - t0) * (z1 - z0))


def pbl_height(frame: AtmozFrame, *, theta: str = "theta", threshold: float = 1.5, group: Optional[str] = "time",
               min_levels: int = 5, out: str = "pbl_height") -> AtmozFrame:
    """Boundary-layer height per profile with the parcel method on potential temperature.

    The height is where theta first exceeds the theta of the lowest valid level of the profile by *threshold* (K),
    interpolated linearly between the two levels that bracket it. Levels with NaN altitude or theta are ignored. A
    profile with fewer than *min_levels* valid levels, or whose theta never exceeds the threshold, gives NaN.

    The method suits a daytime convective boundary layer (a well-mixed layer capped by a warmer inversion). In a
    stable (night-time) layer, or with a superadiabatic surface layer, the result is not a mixed-layer height: the
    lowest level is the reference and no surface-layer correction is made.

    Parameters
    ----------
    frame : AtmozFrame
        Long format: an ``alt`` axis (km) and a theta column (K or degC) with declared units. ``alt`` needs
        ``metadata["alt_reference"]``; for ``"asl"`` it also needs ``metadata["site_elevation_km"]``.
    theta : str
        Potential temperature column (see :func:`theta`).
    threshold : float
        Excess over the lowest level's theta, in K. Default 1.5.
    group : {"time", None}
        ``"time"``: every distinct timestamp is a profile (a profile set repeats the timestamp once per level).
        ``None``: the whole frame is one profile (e.g. a sonde, whose rows each carry their own time); only the rows
        up to the highest altitude (the ascent) are used and the result is indexed by the first timestamp.
    min_levels : int
        Fewest valid levels a profile needs.
    out : str
        Name of the result column.

    Returns
    -------
    AtmozFrame
        Indexed by profile time, one column *out* in km **above ground**. Altitudes above sea level are converted
        with ``site_elevation_km``. ``ValueError`` if the reference or the needed elevation is missing.

    Examples
    --------
    >>> frame = az.met.theta(sonde_frame)
    >>> az.met.pbl_height(frame, group=None).df["pbl_height"]
    """
    _metpy()
    alt_col = frame.axes["alt"]
    if alt_col is None:
        raise ValueError("pbl_height needs a frame with an alt axis.")
    if group not in ("time", None):
        raise ValueError("group must be 'time' or None.")
    if frame.units.get(alt_col, "km") != "km":
        raise ValueError("pbl_height needs alt in km.")
    ref = frame.alt_reference
    if ref not in ("asl", "agl"):
        raise ValueError("The altitude reference is not stated: set metadata['alt_reference'] to 'asl' or 'agl' "
                         "(frame.with_metadata(alt_reference=...)).")
    elevation = 0.0
    if ref == "asl":
        elevation = frame.metadata.get("site_elevation_km")
        if elevation is None or pd.isna(elevation):
            raise ValueError("alt is above sea level and metadata['site_elevation_km'] is missing; it is needed to "
                             "give the height above ground.")
    th_unit = frame.units.get(theta)
    if th_unit is None:
        raise ValueError(f"Column {theta!r} has no declared unit; set it with frame.with_units({theta}='K') first.")
    if th_unit not in ("K", "°C"):
        raise ValueError(f"Column {theta!r} must be in K or °C, not {th_unit!r}.")
    th_all = frame.df[theta].to_numpy(dtype=float) + (273.15 if th_unit == "°C" else 0.0)
    alt_all = frame.df[alt_col].to_numpy(dtype=float) - float(elevation)
    index = frame.df.index

    times, heights = [], []
    if group is None:
        if len(index):
            top = int(np.nanargmax(alt_all)) + 1 if np.isfinite(alt_all).any() else 0
            times.append(index[0])
            heights.append(_profile_pbl(alt_all[:top], th_all[:top], threshold, min_levels))
    else:
        for time, rows in pd.Series(np.arange(len(index)), index=index).groupby(level=0, sort=True):
            pos = rows.to_numpy()
            times.append(time)
            heights.append(_profile_pbl(alt_all[pos], th_all[pos], threshold, min_levels))

    result = pd.DataFrame({out: np.array(heights, dtype=float)}, index=pd.DatetimeIndex(times, tz="UTC", name="time"))
    metadata = {**frame.metadata, "alt_reference": "agl", "pbl_method": "parcel", "pbl_threshold_K": threshold}
    return AtmozFrame(result, units={out: "km"}, metadata=metadata)


@styled("skewt")
def skewt(frame: AtmozFrame, *, temperature: str = "Temp_degC", dewpoint: Optional[str] = None,
          pressure: str = "Pressure_hPa", wind: Optional[tuple[str, str]] = None, ax=None, style=None,
          figsize=None, **kw):
    """Skew-T log-p diagram of one profile (needs MetPy).

    Also available as ``atmoz.plot.skewt``.

    Parameters
    ----------
    frame : AtmozFrame
        One profile with pressure and temperature columns with declared units. Rows with NaN pressure or
        temperature are dropped.
    temperature, pressure : str
        Column names.
    dewpoint : str, optional
        Dewpoint column (see :func:`dewpoint`), drawn as a second line.
    wind : (str, str), optional
        ``(u, v)`` columns (see :func:`wind_components`), drawn as barbs.
    ax : matplotlib Axes, optional
        Replace this axes with the Skew-T (its figure and position are reused).
    style, figsize
        See :mod:`atmoz.plotting`.
    **kw
        Passed to the temperature line (e.g. ``color``, ``linewidth``).

    Returns
    -------
    (fig, ax)
        The figure and the Skew-T axes.

    Examples
    --------
    >>> frame = az.met.wind_components(az.met.dewpoint(sonde_frame))
    >>> fig, ax = az.met.skewt(frame, dewpoint="dewpoint", wind=("u", "v"))
    """
    import matplotlib.pyplot as plt

    try:
        from metpy.plots import SkewT
    except ImportError as err:
        raise ImportError(_NEEDS_METPY) from err
    units, _ = _metpy()
    cols = {"p": _quantity(frame, pressure, units).to("hPa").magnitude,
            "t": _quantity(frame, temperature, units).to("degC").magnitude}
    if dewpoint:
        cols["td"] = _quantity(frame, dewpoint, units).to("degC").magnitude
    if wind:
        cols["u"] = _quantity(frame, wind[0], units).to("kn").magnitude
        cols["v"] = _quantity(frame, wind[1], units).to("kn").magnitude
    data = pd.DataFrame(cols).dropna(subset=["p", "t"]).sort_values("p", ascending=False)

    if ax is not None:
        fig, spec = ax.figure, ax.get_subplotspec()
        ax.remove()
        skew = SkewT(fig, subplot=spec)
    else:
        skew = SkewT(plt.figure())
        fig = skew.ax.figure
    skew.plot(data["p"], data["t"], **{"color": "tab:red", **kw})
    if dewpoint:
        skew.plot(data["p"], data["td"], color="tab:green")
    if wind:
        w = data.dropna(subset=["u", "v"])
        skew.plot_barbs(w["p"], w["u"], w["v"])
    skew.plot_dry_adiabats(alpha=0.25)
    skew.plot_moist_adiabats(alpha=0.25)
    skew.plot_mixing_lines(alpha=0.25)
    skew.ax.set_ylim(max(1000.0, float(data["p"].max())) if len(data) else 1000.0, 100)
    skew.ax.set_xlim(-40, 50)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    return fig, skew.ax
