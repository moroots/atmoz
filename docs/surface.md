# atmoz.surface

EPA air-quality data: bulk pre-generated AQS files and the live AirNow API.

```python
from atmoz.surface import AirNow, AirNowAPI, EPA_PreGen, read_aqs_zip   # core install is enough
from atmoz.surface import utilities as su                    # needs: pip install "atmoz[surface]"
```

---

## Quick Start

```python
from atmoz.surface import AirNow

# Hourly ozone for one Maryland county in 2025 -> an AtmozDataset, one frame per site and parameter
ds = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025,
                     state="Maryland", county="Prince George's")
ds.keys()                                  # "{state}{county}{site}/{parameter code}", e.g. "240338003/44201"
frame = ds["240338003/44201"]              # an AtmozFrame
frame.df.head()                            # UTC time index; value, lat, lon, poc, local_time, ...
frame.units                                # {"value": "ppm", ...}
frame.metadata["state_name"]

# Live AirNow API (needs a free key, saved once, see "API key" below)
AirNow.set_api_key()                      # prompts, then stores it for every future session
ds = AirNow.download(endpoint="airnow", start_date="2025-08-01", end_date="2025-08-07", parameters=["OZONE", "PM25"])
ds["840240053001/OZONE"]                   # "{IntlAQSCode}/{PARAMETER}"
ds.select("*/OZONE").inventory()           # one row per ozone site: lat, lon, unit, time span
ds.metadata["sites"]                       # DataFrame, one row per monitoring station
```

Both sources return an `AtmozDataset` of `AtmozFrame`s (see [Conventions](conventions.md#containers)): time is the UTC index, `lat` / `lon` are
columns, units are in `frame.units`.

Everything is cached on disk (see [Caching](caching.md)), so re-running a script is instant, and every
download is **all-or-nothing**: if one request or file fails, `atmoz.exceptions.DownloadError` is raised
and no partial result is returned. Finished pieces stay in the cache, so a retry only repeats the failed ones.

---

## `AirNow.download()` — Unified Entry Point

```python
AirNow.download(endpoint="aqs" | "airnow", **kwargs)
```

| `endpoint` | Backend | Returns | Use when |
|------------|---------|---------|----------|
| `"aqs"` | `EPA_PreGen.download()` | `AtmozDataset`, keys `"{state}{county}{site}/{parameter code}"` | Historical bulk files for whole years |
| `"airnow"` (default) | `AirNowAPI.download()` | `AtmozDataset`, keys `"{IntlAQSCode}/{PARAMETER}"` | Recent / near-real-time observations from the live API |

Both endpoints return an `AtmozDataset` with one `AtmozFrame` per site and parameter. `ds.metadata` is a dict (`endpoint`, `files`, `rows`;
AQS adds `filters`, the AirNow API adds `start`, `end`, `query` and `sites`). `ds.frame` gives the frame when there is only one; an empty
result is an empty dataset. `AirNow.fetch(endpoint=...)` returns raw files `{name: path}` instead.

### `endpoint="aqs"` kwargs

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `parameters` | `str \| List[str]` | required | Names (`"ozone"`) or AQS codes (`"44201"`, `"SPEC"`). See [EPA Parameters](#epa-parameters). Unknown names raise `ValueError` with suggestions. |
| `resolutions` | `str \| List[str]` | required | `"hourly"`, `"daily"`, `"8hour"`, `"annual"`. |
| `years` | `int \| List[int]` | required | Calendar year(s). |
| `max_workers` | `int` | `5` | Download threads. |
| `bbox` | `(west, south, east, north)` | `None` | Keep only sites inside the box. |
| `start_date`, `end_date` | `str \| date` | `None` | Inclusive local (standard time) dates; the files are still chosen by `years`, these trim the rows. |
| `site` | `str \| (state, county, site) \| list` | `None` | An AQS id (`"240230002"`) or a `(state, county, site)` tuple. |
| `state`, `county` | `str \| int \| list` | `None` | Codes (`"24"`, `"031"`, zero-padded for you) or names (`"Maryland"`). |
| `parameter`, `poc` | `str \| int \| list` | `None` | Rows of one AQS parameter code (`44201`) or POC (monitor number) inside the file. |
| `where` | `dict` | `None` | `{column: value-or-list}` for any column of the file, e.g. `{"Method Code": "087"}`. |
| `use_cache` | `bool` | `True` | `False` keeps nothing on disk. |
| `refresh` | `bool` | `False` | Force a fresh download. |
| `max_age` | `"auto" \| timedelta \| None` | `"auto"` | `"auto"`: current and previous year are re-checked daily (cheap `ETag` check), older years are frozen. `None`: never re-check. |
| `cache_dir` | path | `atmoz.cache_dir("aqs")` | Where the zips live. |
| `silent` | `bool` | `False` | No progress bar / warnings. |
| `show_traceback` | `bool` | `False` | Full tracebacks for failures. |

**Returns** an `AtmozDataset` (`source="EPA AQS"`). Only **hourly** files can be downloaded as frames (other resolutions raise `ValueError`; use
`fetch()` and `read_aqs_zip()` for them). Each zip is parsed once and cached next to it as parquet, and the filters read that parquet, so a
second query on the same year does not touch the network or the big CSV.

* Key: `"{state}{county}{site}/{parameter code}"`, e.g. `"240230002/44201"`.
* Frame: UTC `time` index (hour-beginning, from `Date GMT` + `Time GMT`), columns `value`, `lat`, `lon`, `poc`, `local_time`
  (naive local standard time, hour-beginning; MDA8 uses it) and, when present, `mdl`, `qualifier`, `method_type`, `method_code`.
  Several POCs at a site share the frame; the `poc` column tells them apart.
* Units: `frame.units["value"]` is the unit of the file; **ozone stays in ppm**. Convert with `frame.convert_units("value", "ppb")`.
* `frame.metadata`: `site_id`, `state_code`, `county_code`, `site_num`, `state_name`, `county_name`, `parameter_code`, `parameter_name`,
  `datum`, `units_raw`, `lat`, `lon`, `pocs`.

```python
# Several parameters in one call (every valid combination), trimmed to a box and a few weeks
ds = AirNow.download(
    endpoint="aqs",
    parameters=["ozone", "no2"],
    resolutions="hourly",
    years=[2024, 2025],
    bbox=(-77.5, 38.5, -76.5, 39.5),
    start_date="2025-06-01", end_date="2025-08-31",
)
ds.select("*/44201").inventory()          # the ozone sites in the box
```

**Do it your way.** `EPA_PreGen.fetch(...)` (or `AirNow.fetch(endpoint="aqs", ...)`) only downloads (and caches) the zips and returns their paths, and
`read_aqs_zip(path, columns=..., bbox=...)` parses one zip, so you can mix and match or bring your own parser:

```python
from atmoz.surface import EPA_PreGen, read_aqs_zip
paths = EPA_PreGen.fetch("ozone", "hourly", [2023, 2024])     # {"hourly_44201_2023.csv": Path(...zip), ...}
df = read_aqs_zip(paths["hourly_44201_2024.csv"], columns=["Latitude", "Longitude", "Date GMT", "Time GMT", "Sample Measurement"])
```

`read_aqs_zip` and `fetch` give the plain AQS DataFrame (column schema below); `download()` is the one that gives frames.

### `endpoint="airnow"` kwargs

Forwarded to `AirNowAPI.download()`; see [`download`](#downloadstart_date-end_date-kwargs).
`api_key=` may be passed here too.

---

## `EPA_PreGen` — EPA Pre-Generated Bulk Files (`atmoz/surface/AQS.py`)

Downloads annual ZIP files from `https://aqs.epa.gov/aqsweb/airdata/`. Each ZIP holds one CSV with every US
monitoring site for that parameter / resolution / year.

How it stays fast and light:

* **Streams to disk**, never into memory, and checks the size, so a dropped connection cannot leave a corrupt file.
* **Cached**; the current/previous year is revalidated with the server's `ETag`, so you only re-download what EPA changed.
* **Parsed with Apache Arrow's multi-threaded CSV reader** straight from the zip, about 4-5x faster and 3-4x less
  memory than `pandas.read_csv` on the national hourly files. Downloads use a thread pool (they wait on the
  network); a process pool would only add the cost of copying large DataFrames between processes.
* If Arrow cannot infer a column consistently, the reader quietly falls back to pandas' whole-column inference.

### EPA Parameters

The table below is `EPA_PARAMETERS` in `atmoz/surface/AQS.py`. Combinations marked empty are skipped with a
warning. It has not been re-verified against EPA's current file list; you can always pass an AQS code directly.

| Name | Code | hourly | daily | 8hour | annual |
|------|------|--------|-------|-------|--------|
| `ozone` | 44201 | ✓ | ✓ | ✓ | |
| `so2` | 42401 | ✓ | ✓ | | |
| `co` | 42101 | ✓ | ✓ | ✓ | |
| `no2` | 42602 | ✓ | ✓ | | |
| `pm25` | 88101 | ✓ | ✓ | | |
| `pm25_frm` | 88101 | ✓ | ✓ | | |
| `pm25_nonfrm` | 88502 | ✓ | ✓ | | |
| `pm10` | 81102 | ✓ | ✓ | | |
| `pmc` | 86101 | ✓ | ✓ | | |
| `pm25_spec` | SPEC | ✓ | ✓ | | |
| `pm10_spec` | PM10SPEC | ✓ | ✓ | | |
| `wind` | WIND | ✓ | ✓ | | |
| `temp` | TEMP | ✓ | ✓ | | |
| `pressure` | PRESS | ✓ | ✓ | | |
| `rh_dp` | RH_DP | ✓ | ✓ | | |
| `haps` | HAPS | ✓ | ✓ | | |
| `vocs` | VOCS | ✓ | ✓ | | |
| `nonoxnoy` | NONOxNOy | ✓ | ✓ | | |
| `lead` | LEAD | ✓ | ✓ | | |
| `aqi_by_cbsa` | aqi_by_cbsa | | ✓ | | ✓ |
| `aqi_by_county` | aqi_by_county | | ✓ | | ✓ |
| `conc_by_monitor` | conc_by_monitor | | | | ✓ |

### AQS DataFrame schema (`read_aqs_zip`)

This is the raw file as `read_aqs_zip()` returns it; `download()` renames and reduces the columns into frames (see above). Column sets differ between hourly, daily and annual files; every column in the file is loaded unless you pass
`columns=`. Types applied when the column is present:

| Column(s) | dtype | Notes |
|-----------|-------|-------|
| `Latitude`, `Longitude` | `float64` | Exactly as published: full precision, negative signs kept. |
| `Date Local`, `Date GMT`, `Date of Last Change`, `Date` | `datetime64` | |
| `State Code`, `County Code`, `Site Num`, `Parameter Code`, `POC`, `Method Code`, `Time Local`, `Time GMT`, `Datum`, `Parameter Name`, `Units of Measure`, `Qualifier`, `Method Type`, `Method Name`, `State Name`, `County Name`, ... | `category` | Low-cardinality text; identifiers keep leading zeros (`"087"`). |
| measurements (`Sample Measurement`, `MDL`, `Uncertainty`, `Arithmetic Mean`, ...) | numeric | Inferred. |

**Units note:** ozone is reported in **ppm**; multiply by 1000 for ppb (on a frame: `frame.convert_units("value", "ppb")`).

---

## `AirNowAPI` — Live AirNow API (`atmoz/surface/AirNow_API.py`)

Fetches current and recent observations from the AirNow API. Requires a free key from
[docs.airnowapi.org](https://docs.airnowapi.org/).

### API key

The key is looked up in this order, and **never prompted for** (so scripts, CI and notebooks never hang):

1. `AirNow.download(api_key="...")` / `AirNowAPI(api_key="...")`
2. the `ATMOZ_AIRNOW_API_KEY` environment variable
3. the nearest `.atmoz.env` file, walking up from the current directory
4. `~/.atmoz/.atmoz.env`

Save it once, without hunting for the file:

```python
AirNow.set_api_key()                         # hidden prompt; writes ~/.atmoz/.atmoz.env (mode 0600)
AirNow.set_api_key("abc123", scope="project")  # writes ./.atmoz.env for this project only
```

If none is found, `MissingCredentialError` explains how to set it. The key is never included in exceptions,
reprs or the saved Parquet metadata. See [Configuration](configuration.md) for details and the
one limitation (DEBUG-level logging from `urllib3`).

### `download(start_date, end_date, **kwargs)`

```python
ds = AirNowAPI.download(
    start_date="2025-08-01",
    end_date="2025-08-07",                      # a bare date means "through 23:00 UTC that day"
    parameters=["OZONE", "PM25"],
    BBOX=["-77.5", "38.5", "-76.5", "39.5"],    # DC metro area
)
ds.keys()                          # "{IntlAQSCode}/{PARAMETER}", e.g. "840240053001/OZONE"
ds["840240053001/OZONE"].df        # UTC time index; value, lat, lon, raw_concentration, aqi, category, site_name, agency, aqs_code
ds.metadata["sites"]               # pd.DataFrame, one row per monitoring station
```

Returns an `AtmozDataset` (`source="AirNow API"`), one frame per site and parameter. The unit is the one the API reports, normalised
(`ppb`, `ppm`, `µg/m3`) and kept in `frame.units["value"]`; frame metadata has `site`, `parameter`, `site_name`, `agency`, `aqs_code`
and `intl_aqs_code`. The dataset metadata has `endpoint`, `files`, `rows`, `start`, `end`, `query` and `sites`.

* **One request per window, you choose the size.** The default window is one day (`chunk="1D"`) because a
  request for the whole US, all pollutants and many days at once gets blocked by AirNow. Narrow the box or
  pollutant list, or shrink `chunk` (any `pandas.Timedelta` string, e.g. `"6h"`) if you hit that.
* Windows run on a thread pool (`max_workers=3`), with retries and timeouts from the shared session pool.
* Windows never overlap, and `start_date == end_date` returns that single day.
* All-or-nothing: any failure raises `DownloadError` listing the failed windows.
* Windows that ended more than `AirNowAPI.settle_hours` (72) ago are cached; recent windows are always re-fetched
  because AirNow can still revise them. `use_cache=False` / `refresh=True` override.

| Argument | Default | Description |
|----------|---------|-------------|
| `start_date`, `end_date` | last 24 h | Inclusive, UTC. Strings, `date` or `datetime`. Give both or neither. |
| `max_workers` | `3` | Concurrent requests. |
| `chunk` | `"1D"` | Request window size. |
| `use_cache`, `refresh` | `True`, `False` | File-cache controls. |
| `api_key` | env / `.atmoz.env` | See *API key* above. |
| `silent` | `False` | Hide the progress bar. |
| `parameters` | all 6 | `["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"]` |
| `BBOX` | East US | `[minLon, minLat, maxLon, maxLat]`; numbers or strings, precision and sign preserved. |
| `dataType` | `"B"` | `"A"` AQI only, `"C"` concentration only, `"B"` both. |
| `monitorType` | `"0"` | `"0"` all, `"2"` mobile only. |
| `verbose` | `"1"` | Include station metadata. |
| `includerawconcentrations` | `"1"` | Include raw values. |

> **To verify with a real key:** the window logic assumes AirNow returns the `endDate` hour (inclusive), so windows
> end one hour before the next begins. If that is wrong, set `AirNowAPI.window_end_inclusive = False`.
> A final de-duplication makes either setting safe from double-counting.

### `fetch(start_date, end_date, output_dir, **kwargs)`

Same arguments as `download`, but saves one Parquet file and returns `{filename: Path}`.

```python
files = AirNowAPI.fetch("2025-08-01", "2025-08-31", output_dir="./data")
# -> ./data/airnow_api_query_2025-08-01T00_2025-08-31T23.parquet   (query in df.attrs["query_params"])
```

---

## `mda8` — Ozone 8-hour averages and the daily maximum (MDA8)

Implements the EPA's rules (40 CFR Part 50, Appendix U; the spec is in `docs/references/epa-40cfr50-appendix-u/`). Core install, no extras.

```python
from atmoz.surface import mda8, mda8_aqs

# Every site of an EPA AQS hourly ozone download; one row per site and day
ds = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2024, state="Maryland")
daily = mda8_aqs(ds)                       # an AtmozDataset, or one AtmozFrame such as ds["240338003/44201"]
daily[daily["valid"]].groupby(["State Code", "County Code", "Site Num"])["mda8"].nlargest(4)   # for a design value

# One site, any hourly Series (AirNow API data is in ppb; a UTC index needs the site's standard-time offset)
frame = AirNow.download(endpoint="airnow", start_date="2025-08-01", end_date="2025-08-07", parameters="OZONE",
                        BBOX=["-77.5", "38.5", "-76.5", "39.5"])["840240053001/OZONE"]
r = mda8(frame.df["value"], units=frame.units["value"], utc_offset=-5)
```

`mda8(series, *, units="ppb", utc_offset=None, exceptional_events=None, level_ppb=70)` returns a DataFrame indexed by local `date`:

| Column | Meaning |
|--------|---------|
| `mda8` | Highest of the 17 eight-hour averages starting 07:00 to 23:00 local standard time, truncated, in the units of the input (NaN if no average is valid). |
| `n_averages` | How many of the 17 averages are valid (exceptional-event hours count as present). |
| `valid` | The day meets the completeness rule: at least 13 of 17 averages, **or** `mda8` above 70 ppb. Filter on this before computing design values. |
| `start_hour` | Start hour (7-23) of the 8-hour period that gave the maximum. |

Rules applied: hourly values truncated to whole ppb (3 decimals in ppm); an 8-hour average needs 6 of 8 hours (divisor 6, 7 or 8); with 5 or
fewer it is valid only if the sum of the available hours exceeds 567 ppb; averages are truncated and stored at their start hour. Hours are
**hour-beginning local standard time**, never daylight saving: AQS's `Date Local` / `Time Local` already are, and for a timezone-aware index you pass
`utc_offset` (e.g. `-5` for US Eastern all year). Collocated monitors (POCs) are combined per site as EPA 2(b) says
(`combine_collocated`: the primary POC, default the lowest number, with gaps filled by the mean of the others; pass `primary={site_tuple: poc}`
to `mda8_aqs` if you know it). `mda8_aqs` accepts a raw AQS DataFrame (`read_aqs_zip` output), an `AtmozDataset` or an `AtmozFrame` from `AirNow.download(endpoint="aqs", ...)`.
For frames it uses the `local_time`, `poc` and `value` columns and each frame's own `units["value"]`, and skips frames of other parameters;
the result has the `State Code`, `County Code`, `Site Num` columns and the MDA8 comes back in the frame's unit. For a raw DataFrame it takes
column names (`value`, `date`, `hour`, `site`, `poc`) and `units` (default `"ppm"`).

Not included: exceptional-event flags are not read from AQS qualifiers (pass `exceptional_events=` yourself), and the design value
(annual fourth-highest MDA8, 3-year average) is not computed.

---

## `utilities` — Data Processing Helpers

```python
from atmoz.surface import utilities as su
```

### `make_datetime(df, time_cols, **kwargs) → pd.Series`

Combine separate date and time columns into a single datetime Series. The date column can be datetimes (as the AQS
downloader returns it) or text; the time column is text such as `"05:00"`.

```python
df["UTC"] = su.make_datetime(df, time_cols={
    "date": "Date GMT",
    "time": "Time GMT",
    "format": {"date": "%Y-%m-%d", "time": "%H:%M"},  # "date" only matters for text dates
})
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | Input DataFrame |
| `time_cols` | `dict` | see above | Keys: `"date"`, `"time"`, `"format"` |

---

### `extract_metadata(df, data_columns, ID_cols) → pd.DataFrame`

Drop measurement columns and return one row per unique monitoring station.

```python
meta = su.extract_metadata(df)
# One row per (State Code, County Code, Site Num) — drops Date/Time/Measurement cols
```

| Argument | Default | Description |
|----------|---------|-------------|
| `data_columns` | `["Date Local", "Time Local", "Date GMT", "Time GMT", "Sample Measurement"]` | Columns to drop |
| `ID_cols` | `["State Code", "County Code", "Site Num"]` | Columns that uniquely identify a station |

---

### `to_gdf(df, Latitude, Longitude, crs) → gpd.GeoDataFrame`

Convert a DataFrame with lat/lon columns to a GeoDataFrame.

```python
gdf = su.to_gdf(meta)
gdf = su.to_gdf(meta, Latitude="lat", Longitude="lon", crs="EPSG:4326")
```

---

### `slice_bbox(gdf, bbox) → gpd.GeoDataFrame`

Return points within a geographic bounding box.

```python
bbox = {"lon": [-77.5, -76.5], "lat": [38.5, 39.5]}
gdf_dc = su.slice_bbox(gdf, bbox)
```

`bbox` dict keys: `"lon"` → `[west, east]`, `"lat"` → `[south, north]`.

---

### `df_geo_slice(df, gdf, lat, lon) → pd.DataFrame`

Filter a raw DataFrame to only rows whose (lat, lon) coordinates appear in a GeoDataFrame. Use after `slice_bbox` to apply a geographic filter to the full measurement DataFrame.

```python
df_dc = su.df_geo_slice(df, gdf_dc)
```

---

### `split_by_parameter(df, parameter_col) → Dict[str, pd.DataFrame]`

Split a DataFrame into a dict keyed by parameter name.

```python
by_param = su.split_by_parameter(df)
ozone_df = by_param["Ozone"]
```

---

### `split_by_station(df, ID_cols) → Dict[tuple, pd.DataFrame]`

Split a DataFrame into a dict keyed by `(state_code, county_code, site_num)`.

```python
by_site = su.split_by_station(df)
site_df = by_site[("24", "033", "0030")]
```

---

### `site_map(instruments, title, colors, **kwargs)`

Render an interactive Plotly map of monitoring station locations.

```python
su.site_map(
    instruments={"Ozone Sites": gdf},
    title="EPA AQS Ozone Monitors — 2025",
)

# Multiple layers
su.site_map(
    instruments={"Ozone": ozone_gdf, "PM2.5": pm25_gdf},
    title="EPA AQS Sites",
    hover_cols=["State Name", "County Name", "Site Num"],
    bbox={"lon": [-125, -66], "lat": [24, 50]},
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `instruments` | `Dict[str, gpd.GeoDataFrame]` | required | Layer name → GeoDataFrame with point geometry |
| `title` | `str` | `"Site Map"` | Map title |
| `colors` | `list` | 10 preset hex colors | Colour cycle for successive layers |
| `hover_cols` | `List[str]` (kwarg) | `None` | Columns shown in hover tooltip |
| `bbox` | `dict` (kwarg) | CONUS | `{"lon": [w, e], "lat": [s, n]}` for map center/extent |
| `layout` | `dict` (kwarg) | `{"height": 800, "width": 1600}` | Plotly layout overrides |

---

### `get_resolution(df, col1, col2, col3, time_col) → pd.DataFrame`

Return the temporal sampling resolution for each (col1, col2, col3) group (e.g., per station).

```python
res = su.get_resolution(df, "State Code", "County Code", "Site Num")
```

---

### `get_stations(df, lat_lon) → List[Tuple]`

Return unique `(latitude, longitude)` tuples from a DataFrame.

```python
coords = su.get_stations(df)
# [(39.055277, -76.878333), ...]
```

---

### `count_duplicates(df, cols) → None`

Print any duplicate row combinations across all subsets of the given columns. Useful for data quality checks.

```python
su.count_duplicates(df, ["State Code", "County Code", "Site Num", "Date GMT", "Time GMT"])
# Prints "No Duplicates" or lists each duplicate set found
```

---

## Common Patterns

### Filter national AQS file to a bounding box

```python
from atmoz.surface import EPA_PreGen, read_aqs_zip
from atmoz.surface import utilities as su   # needs: pip install "atmoz[surface]"

# The raw national file, as one DataFrame (no frames)
paths = EPA_PreGen.fetch("ozone", "hourly", 2025)
df = read_aqs_zip(paths["hourly_44201_2025.csv"])

# Latitude / Longitude already arrive as float64, so there is nothing to convert.
meta = su.extract_metadata(df)
gdf  = su.to_gdf(meta)
gdf_roi = su.slice_bbox(gdf, bbox={"lon": [-77.5, -76.5], "lat": [38.5, 39.5]})
df_roi   = su.df_geo_slice(df, gdf_roi)

# Split by station
by_site = su.split_by_station(df_roi)
```

If you want frames rather than a DataFrame, filter while reading instead; nothing outside the box is ever loaded:

```python
from atmoz.surface import AirNow

ds = AirNow.download(
    endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025,
    bbox=(-77.5, 38.5, -76.5, 39.5),          # (west, south, east, north)
)
```

### Select the primary site (most observations)

```python
inv = ds.inventory().sort_values("rows", ascending=False)
key = inv["key"].iloc[0]                       # e.g. "240338003/44201"
site = ds[key]
site_lat, site_lon = site.location

ozone_ppb = site.convert_units("value", "ppb")       # AQS ozone is ppm; the frame says so in site.units
ozone_ppb.df["value"].resample("1D").max()
```
