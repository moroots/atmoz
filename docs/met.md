# atmoz.met

Meteorology on `AtmozFrame`s: potential temperature, dewpoint, wind components, boundary-layer height and a Skew-T plot.

```python
import atmoz as az
az.met.theta(frame)
```

Install the extra first: `pip install "atmoz[met]"` (adds [MetPy](https://unidata.github.io/MetPy/)). `import atmoz` does not
import MetPy; calling a function without it raises an `ImportError` that says how to install it.

Every function takes an `AtmozFrame` and returns a new one that keeps the index, axes and metadata. Inputs are read in the unit
the frame **declares** in `frame.units` (a column with no declared unit raises `ValueError`; declare it with
`frame.with_units(Temp_degC="°C")`), so `Pressure_hPa` in hPa or Pa both work. Every new column declares its unit.

| Function | Adds / returns | Unit |
|----------|----------------|------|
| `theta(frame, *, temperature="Temp_degC", pressure="Pressure_hPa", out="theta")` | potential temperature (1000 hPa reference) | K |
| `dewpoint(frame, *, temperature="Temp_degC", rh="RH_percent", out="dewpoint")` | dewpoint; RH declared in `%` or as a fraction (`1`); RH of 0 gives NaN | °C |
| `wind_components(frame, *, speed="WindSpeed_mps", direction="WindDirection_deg", out=("u", "v"))` | eastward and northward wind; `direction` is where the wind blows **from** | m/s |
| `wind_speed_direction(frame, u="u", v="v", out=("speed", "direction"))` | speed and from-direction; north is 360 (MetPy), calm is 0 | m/s, deg |
| `pbl_height(frame, *, theta="theta", threshold=1.5, group="time", min_levels=5, out="pbl_height")` | one row per profile: boundary-layer height above ground | km |
| `skewt(frame, *, temperature="Temp_degC", dewpoint=None, pressure="Pressure_hPa", wind=None, ax=None, **kw)` | `(fig, ax)`; also `az.plot.skewt` | |

## Sonde example

```python
import atmoz as az

ds = az.Sonde.read("sonde_20250728.ict")
sonde = ds.frame                                    # per-row Pressure_hPa, Temp_degC, RH_percent, wind, alt (km asl)

sonde = az.met.theta(sonde)
sonde = az.met.dewpoint(sonde)
sonde = az.met.wind_components(sonde)

pbl = az.met.pbl_height(sonde, group=None)          # a sonde is one profile
pbl.df["pbl_height"].iloc[0], pbl.units             # km above ground, {"pbl_height": "km"}

fig, ax = az.met.skewt(sonde, dewpoint="dewpoint", wind=("u", "v"))
```

A sonde frame is in altitude above sea level, so `pbl_height` needs `metadata["site_elevation_km"]` (the launch elevation, which the
reader fills in when the file gives it). If it is missing, add it with `sonde.with_metadata(site_elevation_km=0.05)`.

## GEOS-CF example

GEOS-CF returns one frame per variable with a `value` column. Build one frame from the surface meteorology variables and
declare the units:

```python
import pandas as pd
import atmoz as az
from atmoz.models import GEOS_CF

ds = GEOS_CF.download(39.055, -76.878, "2025-07-28", "2025-07-30", collection="met", variables=["u10m", "v10m", "t2m"])
df = pd.DataFrame({k: ds[k].df["value"] for k in ("u10m", "v10m", "t2m")})
met = az.AtmozFrame(df, units={"u10m": "m/s", "v10m": "m/s", "t2m": "K"}, lat=None, metadata=ds["u10m"].metadata)

wind = az.met.wind_speed_direction(met, u="u10m", v="v10m")      # speed (m/s) and direction (deg)
```

## PBL height method

`pbl_height` uses the **parcel method** on potential temperature. For each profile it takes the lowest valid level as the
reference and finds the first level above it where theta exceeds the reference theta by `threshold` (default 1.5 K). The height is
interpolated linearly between the two levels that bracket the threshold. The result is in km **above ground**: an `alt` frame
in `"agl"` is used as is, an `"asl"` frame is converted with `metadata["site_elevation_km"]`; a missing reference or elevation
raises `ValueError`. The result's `metadata["alt_reference"]` is `"agl"`.

* **Profiles**: with `group="time"` every distinct timestamp is a profile (a profile set repeats its timestamp once per level, e.g.
  TOLNet or GEOS-CF profiles with a theta column). With `group=None` the whole frame is one profile (a sonde, whose rows each have
  their own time); only the rows up to the highest altitude (the ascent) are used and the result is indexed by the first timestamp.
* **NaN**: levels with NaN altitude or theta are ignored. A profile with fewer than `min_levels` valid levels, or whose theta never
  exceeds the threshold within the profile, gives NaN.
* **Limits**: the method suits a daytime convective boundary layer capped by an inversion. In a stable (night-time) boundary layer
  theta increases from the ground, so the result is the height of a 1.5 K rise, not a mixed-layer top. A superadiabatic surface
  layer is not corrected for, and the result depends on the vertical resolution of the profile and on `threshold`.
  Compare against another estimate (for example GEOS-CF `zpbl`) before relying on it.
