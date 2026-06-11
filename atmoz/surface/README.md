# atmoz.surface

EPA air quality data access — bulk pre-generated AQS files and the live AirNow API.

```python
from atmoz.surface import AirNow
from atmoz.surface import utilities as su
```

---

## Quick Start

```python
from atmoz.surface import AirNow

# Download national hourly ozone file for 2025
data = AirNow.download(endpoint="aqs", parameters=["ozone"], resolutions=["hourly"], years=[2025])
df   = data["hourly_44201_2025.csv"]   # pandas DataFrame, all US monitoring sites
```

---

## `AirNow.download()` — Unified Entry Point

```python
AirNow.download(endpoint="aqs" | "airnow", **kwargs)
```

Routes to one of two backends depending on `endpoint`.

| `endpoint` | Backend | Use When |
|------------|---------|----------|
| `"aqs"` (default for bulk) | `epa_pregen._download()` | Historical bulk files for a full year |
| `"airnow"` | `AirNow().import_data()` | Recent/real-time observations via live API |

### `endpoint="aqs"` kwargs

Forwarded to `epa_pregen._download()`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `parameters` | `str \| List[str]` | required | Parameter name(s) or numeric code(s). See [EPA Parameters](#epa-parameters) below. |
| `resolutions` | `str \| List[str]` | required | `"hourly"`, `"daily"`, `"8hour"`, or `"annual"`. |
| `years` | `int \| List[int]` | required | Calendar year(s) to download. |
| `max_workers` | `int` | `5` | Parallel download threads. |
| `show_traceback` | `bool` | `False` | Print full traceback on failed downloads. |

**Returns:** `Dict[str, pd.DataFrame]` — keys are filenames (`"hourly_44201_2025.csv"`), values are DataFrames. See [AQS DataFrame Schema](#aqs-dataframe-schema) below.

```python
# Single parameter, single year
data = AirNow.download(endpoint="aqs", parameters=["ozone"], resolutions=["hourly"], years=[2025])

# Multiple parameters
data = AirNow.download(
    endpoint="aqs",
    parameters=["ozone", "pm25", "no2"],
    resolutions=["hourly", "daily"],
    years=[2024, 2025],
    max_workers=8,
)
# Returns up to 12 DataFrames (3 params × 2 resolutions × 2 years, if valid)
```

### `endpoint="airnow"` kwargs

Forwarded to `AirNow().import_data()`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `start_date` | `str \| datetime` | 24 h ago | Start of date range. |
| `end_date` | `str \| datetime` | now | End of date range. |
| `parameters` | `List[str]` | all 6 | `["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"]` |
| `BBOX` | `List[str]` | East US | `[minLon, minLat, maxLon, maxLat]` as strings. |
| `dataType` | `str` | `"B"` | `"A"` = AQI, `"C"` = concentration, `"B"` = both. |
| `monitorType` | `str` | `"0"` | `"0"` = permanent, `"2"` = mobile, `"0"` = all. |

**Returns:** `atmoz_dataset` with `.data` (DataFrame) and `.metadata` (DataFrame) attributes.

---

## `epa_pregen` — EPA Pre-Generated Bulk Files

Downloads annual ZIP files from `https://aqs.epa.gov/aqsweb/airdata/`. Each ZIP contains one CSV with all US monitoring sites for that parameter/resolution/year combination.

### EPA Parameters

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

You can also pass numeric codes directly: `parameters=["44201"]`.

### AQS DataFrame Schema

Every downloaded CSV contains these columns:

| Column | Type | Example |
|--------|------|---------|
| `State Code` | str | `"24"` |
| `County Code` | str | `"033"` |
| `Site Num` | str | `"0030"` |
| `Parameter Code` | str | `"44201"` |
| `POC` | str | `"1"` |
| `Latitude` | str → float | `"39.055277"` |
| `Longitude` | str → float | `"-76.878333"` |
| `Datum` | str | `"WGS84"` |
| `Parameter Name` | str | `"Ozone"` |
| `Date Local` | str | `"2025-01-01"` |
| `Date GMT` | str | `"2025-01-01"` |
| `Time GMT` | str | `"05:00"` |
| `Sample Measurement` | float | `0.042` |
| `Units of Measure` | str | `"Parts per million"` |
| `MDL` | str | method detection limit |
| `Uncertainty` | float | |
| `Qualifier` | str | |
| `Method Type` | str | |
| `Method Code` | str | |
| `Method Name` | str | |
| `State Name` | str | `"Maryland"` |
| `County Name` | str | `"Prince Georges"` |
| `Date of Last Change` | str | |

**Units note:** Ozone is reported in **ppm** (parts per million). Multiply by 1000 to get ppb.

---

## `AirNow` — Live AirNow API

Fetches current and recent air quality observations from the AirNow API.
Requires a free API key from [airnowapi.org](https://docs.airnowapi.org/).

### `AirNow(api_key=None)`

```python
airnow = AirNow()                   # reads key from system keyring
airnow = AirNow(api_key="abc123")   # pass key directly
```

If no key is found, the first API call will prompt for one and save it to the system keyring under `("EPA_AirNow", "API_KEY")`.

### `import_data(start_date, end_date, **kwargs)`

Fetch observations for a date range. Issues one API request per day.

```python
dataset = AirNow().import_data(
    start_date="2025-08-01",
    end_date="2025-08-07",
    parameters=["OZONE", "PM25"],
    BBOX=["-77.5", "38.5", "-76.5", "39.5"],   # DC metro area
)

print(dataset.data)       # pd.DataFrame of all observations
print(dataset.metadata)   # pd.DataFrame, one row per monitoring station
```

AirNow API parameters:

| kwarg | Default | Description |
|-------|---------|-------------|
| `parameters` | all 6 | `["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"]` |
| `BBOX` | East US | `[minLon, minLat, maxLon, maxLat]` |
| `dataType` | `"B"` | `"A"` AQI only, `"C"` concentration only, `"B"` both |
| `monitorType` | `"0"` | `"0"` all, `"2"` mobile only |
| `verbose` | `"1"` | include station metadata |
| `includerawconcentrations` | `"1"` | include raw values |
| `silent` | `True` | show progress bar |

### `download_data(start_date, end_date, output_dir, max_workers, **kwargs)`

Same as `import_data` but saves results to a Parquet file.

```python
path = AirNow().download_data(
    start_date="2025-08-01",
    end_date="2025-08-31",
    output_dir=Path("./data"),
    max_workers=4,
)
# -> ./data/airnow_api_query_2025-08-01_2025-08-31.parquet
```

Returns the `Path` to the saved file.

---

## `utilities` — Data Processing Helpers

```python
from atmoz.surface import utilities as su
```

### `make_datetime(df, time_cols, **kwargs) → pd.Series`

Combine separate date and time columns into a single datetime Series.

```python
df["UTC"] = su.make_datetime(df, time_cols={
    "date": "Date GMT",
    "time": "Time GMT",
    "format": {"date": "%Y-%m-%d ", "time": "%H:%M"},
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
# [('39.055277', '-76.878333'), ...]
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
import pandas as pd
from atmoz.surface import AirNow
from atmoz.surface import utilities as su

# Download
data = AirNow.download(endpoint="aqs", parameters=["ozone"], resolutions=["hourly"], years=[2025])
df = data["hourly_44201_2025.csv"]

# Convert lat/lon to numeric, then slice to bbox
df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

meta = su.extract_metadata(df)
gdf  = su.to_gdf(meta)
gdf_roi = su.slice_bbox(gdf, bbox={"lon": [-77.5, -76.5], "lat": [38.5, 39.5]})
df_roi   = su.df_geo_slice(df, gdf_roi)

# Split by station
by_site = su.split_by_station(df_roi)
```

### Select primary site (most observations)

```python
site_counts = df_roi.groupby(["State Code", "County Code", "Site Num"])["Sample Measurement"].count()
state, county, site = site_counts.idxmax()

df_site = df_roi[
    (df_roi["State Code"]  == state) &
    (df_roi["County Code"] == county) &
    (df_roi["Site Num"]    == site)
].copy()

site_lat = float(df_site["Latitude"].iloc[0])
site_lon = float(df_site["Longitude"].iloc[0])

# Build UTC datetime index and convert ppm → ppb
df_site["UTC"] = pd.to_datetime(df_site["Date GMT"] + " " + df_site["Time GMT"], utc=True)
df_site.set_index("UTC", inplace=True)
df_site["O3_ppb"] = df_site["Sample Measurement"] * 1000.0
```
