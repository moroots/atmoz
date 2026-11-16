# Analysis: stats and vertical

Two small modules turn frames into numbers. They work on an `AtmozFrame` or an `AtmozDataset` and return plain pandas
objects (the vertical layer mean returns frames).

```python
import atmoz as az
az.stats      # compare, exceedances, diurnal, trend
az.vertical   # layer_mean, mean_profile
```

## `atmoz.stats`

### `compare(data, obs=None, model=None, *, by=None, bins=None)`

Bias, errors and correlation between two columns, usually the output of [`atmoz.collocate`](collocation.md). Only rows where both
values exist count; bias is `model - obs`. By default the first two sources of a collocated frame are used
(`frame.metadata["columns"]`), else columns named `obs` and `model`; pass `obs=` / `model=` to pick others.

```python
out = az.collocate(lidar, model)                  # columns obs, model
az.stats.compare(out)                             # one row, index "all"
az.stats.compare(out, by="alt", bins=1.0)         # altitude bins of 1 km (or pass the bin edges)
az.stats.compare(out, by="hour")                  # UTC hour of day
az.stats.compare(out, by="month")
```

Returns a DataFrame with the columns `n`, `obs_mean`, `model_mean`, `bias`, `mae`, `rmse`, `nmb_percent` and `nme_percent`
(normalised mean bias and error, relative to the sum of `obs`) and `r` (Pearson correlation; NaN when fewer than two points or a
constant series). The index is `group` (`"all"`, the altitude interval, the hour or the month). `by="alt"` needs the frame's `alt` axis; the bins start at 0 km.
For a dataset the index is `(key, group)`, with an extra pooled `("all", group)` block when the dataset holds several frames.

### `exceedances(mda8_values, *, level_ppb=70.0, units="ppm", ...)`

Days above the ozone standard, per site, from daily MDA8 values. Pass the table from `atmoz.surface.mda8_aqs`, or an AQS
`AtmozDataset` / `AtmozFrame` (MDA8 is computed for you):

```python
from atmoz.surface import AirNow, mda8_aqs

ds = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025, state="Maryland")
table = az.stats.exceedances(ds)                  # same as exceedances(mda8_aqs(ds), units="ppm")
table[["State Code", "County Code", "Site Num", "exceedance_days", "fourth_highest"]]
```

One row per site (the site columns of the input) with `valid_days`, `days_total`, `exceedance_days` (valid days with MDA8 above
`level_ppb`), `max_mda8` and `max_date`, `fourth_highest` (the value an EPA design value is built from; NaN with fewer than four
valid days) and `first` / `last` date. The MDA8 values and the level are in ppb in the result; `units` declares the unit of a table you pass
(`"ppm"` for AQS, `"ppb"` otherwise). See [Surface](surface.md) for MDA8 itself.

### `diurnal(data, col="value", *, stat="mean", utc_offset=None, local_time=False)`

The statistic of `col` by hour of day, one column per frame, index `hour` (0-23).

```python
az.stats.diurnal(ds)                              # UTC hours
az.stats.diurnal(ds, utc_offset=-5)               # hours of local standard time at UTC-5
az.stats.diurnal(ds, local_time=True)             # use the AQS "local_time" column
az.stats.diurnal(ds, stat="max")                  # any pandas aggregation name
```

### `trend(data, col="value", *, freq="YS", agg="mean", min_count=1)`

Period statistics (`freq`: `"MS"` monthly, `"YS"` yearly, ...) and their least-squares linear trend, one row per frame (a single
frame is keyed `"frame"`). Columns: `slope_per_year` (in the column's unit), `intercept` (at the first period), `r2`,
`n_periods`, `start`, `end`. The slope is NaN with fewer than three periods; periods with fewer than `min_count` values are dropped.
The period values themselves are `frame.resample(freq).mean()`.

## `atmoz.vertical`

### `layer_mean(data, bottom, top, col="value", *, reference=None, min_points=3)`

The mean of `col` between `bottom` and `top` km at every profile time, as a time series. Returns an `AtmozFrame` (or an
`AtmozDataset` for a dataset) with a `value` column in the unit of `col` and `lat` / `lon` kept. Profiles with fewer than
`min_points` valid points in the layer give NaN. `reference="agl"` or `"asl"` says what the limits are measured from; frames are converted
with `with_alt_reference`, which needs `metadata["site_elevation_km"]`. Without it each frame's own reference is used.

```python
bl = az.vertical.layer_mean(tolnet_frame, 0, 2, reference="agl")        # 0-2 km above ground
bl.df["value"].resample("1h").mean()
```

### `mean_profile(data, col="value", *, bins=0.5, reference=None)`

Mean, standard deviation and count of `col` per altitude bin, over all times (and all frames of a dataset). `bins` is a width
in km or the bin edges. Returns a DataFrame indexed by the bin centre (`alt`, km) with `mean`, `std` and `count`.

```python
az.vertical.mean_profile(tolnet_frame, bins=0.5, reference="agl")
```

The comparison plots built on these are in [Plotting](plotting.md): `plot.comparison_profile` and `plot.exceedances`.
