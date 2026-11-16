# Conventions

Every data source in atmoz (AirNow API, EPA AQS files, TOLNet, UCN and GEOS-CF) follows the same
process with the same names and returns the same containers, so you only learn it once.

## Public entry points

| Method | Returns | What it does |
|--------|---------|--------------|
| `download(...)` | `AtmozDataset` | Get the data, parsed, ready to use (see [Containers](#containers)). |
| `fetch(...)` | `{name: Path}` | Only put the raw files on disk (cached). You parse them. |

`AirNow.download(endpoint=...)` / `AirNow.fetch(endpoint=...)` pick the backend for you.

## Process (private steps, in this order)

| Step | Name | Does |
|------|------|------|
| 1 | `_validate` | Check and normalise the request, expand it into jobs (one per file / window). |
| 2 | `_url` | Build the request URL (secrets only when asked for). |
| 3 | `_fetch_one` | Network: one job -> a file on disk (or the raw payload). |
| 4 | `_parse_one` | One raw file / payload -> a DataFrame. |
| - | `_load_one` | `_fetch_one` + `_parse_one` (+ the cache) for one job. |
| 5 | `_assemble` | Results -> `AtmozDataset`. |

The same helper runs steps 3-4 for every source: `atmoz.resources.utilities.run_jobs`. It uses a thread pool
(downloads wait on the network) and is **all-or-nothing**: if any job fails it raises `DownloadError` and returns nothing.

## Naming

* Modules: one per backend, named for the data source (`AQS.py`, `AirNow_API.py`); classes `EPA_PreGen`, `AirNowAPI`.
* Shared plumbing lives in `atmoz.resources` (`utilities`, `cache`, `credentials`, `urls`, `sessionHandler`).
* Common options keep the same names everywhere: `use_cache`, `refresh`, `cache_dir`, `max_workers`, `silent`, `show_traceback`, `session_handler`.

## Status

AirNow API, EPA AQS, TOLNet, UCN and GEOS-CF follow this process and return an `AtmozDataset` of `AtmozFrame`s.

## Containers

```python
import atmoz
atmoz.AtmozFrame, atmoz.AtmozDataset      # also atmoz.dataclasses
```

An **`AtmozFrame`** is one table: a pandas DataFrame plus its axes, units and metadata.

* `frame.df` is the DataFrame. **Time is the index** (the axis `t`, `frame.t`): always a timezone-aware UTC `DatetimeIndex`, named `time`.
  Long-format profiles repeat a timestamp once per altitude, so the index can contain duplicates.
* The other axes are **columns**, named `lat`, `lon` and `alt` (altitude, **km**); `frame.lat`, `frame.lon`, `frame.alt` return
  those columns (or `None` when the frame has no such axis). `frame.axes` lists the column names.
* `frame.units` is `{column: unit}` and the only place units are recorded (there is no per-row unit column).
* `frame.metadata` is a dict (site, instrument, provenance) that travels with the frame.
* Pandas methods work on the frame and keep what still applies: `frame.resample("1h").mean()` returns an `AtmozFrame`
  (`frame["value"]` returns the plain Series).

An **`AtmozDataset`** is a named collection of frames.

* `ds[key]` is a frame, `ds.keys()`, `ds.items()`, `len(ds)`, `key in ds`. `ds.frame` is the only frame when there is exactly one.
* `ds.select("*/OZONE")` returns a new dataset of the frames whose key matches a glob pattern.
* All frames share one axis schema (the same `lat`/`lon`/`alt` column names). **Units are per frame**, so one dataset can hold ppb ozone
  and µg/m3 PM2.5.
* `ds.name`, `ds.source`, `ds.time_start`, `ds.time_end` (from the frames' time indexes) and `ds.metadata` (provenance: endpoint, files, rows, query).
* Pandas methods broadcast to every frame: `ds.resample("1h").mean()` returns a new `AtmozDataset`.

```python
ds = AirNow.download(endpoint="airnow", start_date="2025-08-01", end_date="2025-08-07", parameters=["OZONE", "PM25"])
ozone = ds.select("*/OZONE")                      # only the ozone frames
hourly = ozone.resample("1h").mean()              # every frame, same keys, units kept
frame = ds["840240053001/OZONE"]
frame.df["value"], frame.units, frame.metadata["site_name"]
```

Frames are built by the sources; to make one yourself, give a DataFrame with a UTC `DatetimeIndex`:

```python
af = atmoz.AtmozFrame(df, units={"value": "ppb"}, lat="lat", lon="lon", metadata={"site": "mine"})
```

Frame keys follow one scheme per source:

| Source | Key | One frame is | Value column |
|--------|-----|--------------|--------------|
| AirNow API | `"{IntlAQSCode}/{PARAMETER}"`, e.g. `"840240053001/OZONE"` | one site and parameter; columns `value`, `lat`, `lon` and the extras the API reports (`raw_concentration`, `aqi`, ...) | `value` |
| EPA AQS | `"{state}{county}{site}/{parameter code}"`, e.g. `"240230002/44201"` | one site and parameter (hourly files only); `value`, `lat`, `lon`, `poc`, `local_time`, ... | `value` |
| TOLNet | the file name | one file, long format: `alt` (km), `value`, `lat`, `lon` | `value` |
| UCN | `"CEILO-{n}_{YYYYMMDD}_{level}"`, e.g. `"CEILO-9_20261101_L2"` | one ceilometer, day and level, long format: `alt` (km), `value` (L2: attenuated backscatter), `lat`, `lon` | `value` |
| GEOS-CF | the variable name, e.g. `"o3"` | one variable at the requested point: `value`, `lat`, `lon`; `alt` (km) and `lev` on model-level collections (`chm`, `met3d`) | `value` |

## Units

Units are declared in `frame.units` and are whatever the source reports; atmoz does not silently convert them (except GEOS-CF's
mol/mol, see below). Spellings are normalised: `ppbv` is stored as `ppb` (likewise `ppmv`, `pptv`).

| Source | `value` unit | Notes |
|--------|--------------|-------|
| TOLNet | read from the file (ozone is normally ppb) | `alt` is converted to km from the unit the file declares. |
| GEOS-CF | `ppb` for gas species, otherwise as served (`µg/m3` for PM2.5, K, m/s, ...) | `units="mol/mol"` keeps the server's raw values. |
| AirNow API | the unit the API reports for the parameter, typically `ppb` for the gases and `µg/m3` for PM2.5 and PM10 (`ppm` where the API reports it) | One dataset can mix units; check `frame.units["value"]`. |
| EPA AQS | as in the file; ozone stays in `ppm` | Declared in `frame.units`, never converted for you. |
| UCN | `km^-1.sr^-1` (attenuated backscatter) | Kept as the file spells it. |

To convert, use `convert_units`, which returns a new frame with the column and its declared unit changed:

```python
aqs_ppb = aqs_frame.convert_units("value", "ppb")       # ppm -> ppb: values times 1000, units["value"] == "ppb"
aqs_frame.units["value"]                                # still "ppm" on the original
```

Conversion needs the column to have a declared unit (`frame.with_units(value="ppb")` declares one) and works between units pint knows,
such as mixing ratios (ppm, ppb, ppt) with each other. It does not turn a mixing ratio into a mass concentration (µg/m3).
`atmoz.collocate` converts the model to the observation unit for you.

## Site metadata

```python
ds.inventory()          # a DataFrame, one row per frame
frame.location          # (lat, lon): the median of the lat/lon columns, or None
```

`inventory()` has the same columns for every source: `key`, `lat`, `lon`, `alt_min`, `alt_max` (km, profiles only), `value_unit`,
`units` (all declared units), `start`, `end` and `rows`. Use it to compare sites and units at a glance, find the sites in a box, or take a site's
coordinates to ask another source for the same place:

```python
inv = ds.inventory()
inv[(inv.lat.between(38.5, 39.5)) & (inv.value_unit == "ppb")]
lat, lon = ds.frame.location                      # e.g. a GEOS_CF.download(lat, lon, ...) at the same site
```

## Collocation

`atmoz.collocate(obs, model, obs_col="value", model_col="value", max_gap="1h")` puts model output on the times and altitudes of an
observation and returns an `AtmozFrame` with `obs` and `model` columns (plus `alt`, `lat`, `lon` from the observation).

* **Time**: for every observation time the model is interpolated linearly between its two nearest times; if either neighbour is more than
  `max_gap` away the result is NaN.
* **Altitude** (profiles): each model profile is interpolated onto the observation altitudes with the model's own `alt` column, which may differ
  per time (GEOS-CF layer heights do). Outside the model's altitude range the result is NaN.
* **Units**: the model column is converted to the observation unit when both are declared and differ.
* Profile against profile is level by level; surface against surface in time only. A surface series against a profile is allowed: the profile side is first reduced to its lowest valid level at each time (or to the altitude given as `alt=` in km).
* `result.metadata` keeps both inputs' metadata, `max_gap` and the columns used.

See [Collocation](collocation.md) for a TOLNet versus GEOS-CF example.
