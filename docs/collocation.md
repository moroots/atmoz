# Collocation

`atmoz.collocate` puts model output on the times and altitudes of an observation, in the observation's unit, so the two can be compared row by row.

```python
import atmoz
out = atmoz.collocate(obs, model, obs_col="value", model_col="value", max_gap="1h")
```

`obs` and `model` are `AtmozFrame`s (see [Conventions](conventions.md#containers)). The result is an `AtmozFrame` on the observation's rows with the
columns `obs` and `model`, plus `alt`, `lat` and `lon` from the observation, a UTC time index, `units` for `obs` and `model` (the observation
unit) and `metadata` holding both inputs' metadata, `max_gap` and the columns used.

## TOLNet versus GEOS-CF

```python
import pandas as pd
import atmoz
from atmoz.lidar import TOLNET
from atmoz.models import GEOS_CF

# One TOLNet file: a long AtmozFrame (UTC time, alt in km, ozone value in ppb)
tol = TOLNET.download("2025-07-28", "2025-07-28", product_type=[4], processing_type=[1])
obs = tol[next(iter(tol))]
lat, lon = obs.location                                   # the instrument

# GEOS-CF ozone profiles at the same place, covering the observation days
cf = GEOS_CF.download(lat, lon, obs.t.min().date(), obs.t.max().date() + pd.Timedelta("1D"),
                      collection="chm", variables=["o3"])
model = cf["o3"]                                          # long: value (ppb), alt (km), lev

out = atmoz.collocate(obs, model)                         # columns: alt, lat, lon, obs, model
out.units                                                 # {"obs": "ppb", "model": "ppb", "alt": "km"}
out.df["bias"] = out.df["model"] - out.df["obs"]
out.df.groupby("alt")["bias"].mean()                      # mean model bias by altitude
```

Add one day to the end date so the last model time after the final observation is there to interpolate to.

What `collocate` does:

* **Time.** For every observation time the model is interpolated linearly between its two nearest times (GEOS-CF is hourly, at `00:30`, `01:30`, ...).
  If either neighbour is further than `max_gap` away (default `"1h"`; use `"2h"` to bridge a missing hour) the result is NaN.
* **Altitude.** Each model profile is interpolated onto the observation's altitudes using the model's own `alt` column, which differs per time
  for GEOS-CF (layer heights change). Outside the model's altitude range the result is NaN.
* **Units.** The model column is converted to the observation unit when both are declared and differ (for example ppm to ppb).
* **Kinds.** Profile against profile is level by level; surface against surface in time only. Surface against profile is allowed too: the profile side is reduced to its lowest valid level at each time, or to `alt=` (km), and the result is a surface-style frame on the surface frame's times (`metadata["obs"|"model"]["reduced_to_surface"]` says which side was reduced).

Check that both altitudes use the same reference before reading the difference: GEOS-CF's `alt` is the layer height above the surface, and the lidar
altitude is whatever the TOLNet file reports (it may be above sea level).
