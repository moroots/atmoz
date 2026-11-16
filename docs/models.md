# atmoz.models

Access NASA [GEOS-CF](https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/) (Goddard Earth Observing System Composition Forecast) chemistry-climate model output at any global site via OPeNDAP.

GEOS-CF provides hourly atmospheric chemistry, composition, and meteorology fields at 0.25° × 0.3125° resolution from 2018 onward, as both a replay (assimilation) product and a rolling 5-day forecast.

```python
from atmoz.models import GEOS_CF
```

Install the extra first: `pip install "atmoz[models]"`.

---

## Quick Start

```python
from atmoz.models import GEOS_CF

# Surface air quality at a site -> an atmoz_dataset (cached on disk, all-or-nothing)
ds = GEOS_CF.download(39.055, -76.878, "2025-01-01", "2025-01-31", variables=["o3"])
o3 = ds.data["aqc_assim"]["o3"]              # pandas Series, UTC time index, ppbv
print(f"O3 range: {o3.min():.1f} - {o3.max():.1f} ppbv")

# Chemistry profiles: one time x level DataFrame per variable
prof = GEOS_CF.download(39.055, -76.878, "2025-07-28", "2025-08-01", collection="chm", variables=["o3"])
prof.data["chm_assim_o3"]                    # index: UTC time, columns: model level
```

`GEOS_CF.fetch(...)` takes the same request but only fills the cache and returns `{file_name: parquet_path}`.
See the [conventions](conventions.md): it works like every other atmoz source.

---

## Collections

Pass the collection name as the `collection` argument to `download()` / `fetch()`. `GEOS_CF.catalog` has the full dict (URLs, variables).

| Collection | Description | Default variables | Levels | Period |
|------------|-------------|-------------------|--------|--------|
| `aqc` | Surface air quality, 1-hr average | `o3`, `no2`, `co`, `so2`, `pm25_rh35_gcc` | surface | 2018 to present (also `fcast`) |
| `chm` | Chemistry **profiles**, 1-hr average, 50+ species | `o3`, `no2`, `co`, `so2`, `hno3` | lowest **36 model levels** (`lev` 37 = highest ... 72 = surface layer) | 2020 to present |
| `chm_sfc` | Surface chemistry, 1-hr average, 50+ species | same | surface | 2018 to present |
| `met` | Surface meteorology, 1-hr average | `t2m`, `u10m`, `v10m`, `q2m`, `zpbl`, `ps` | surface | 2018 to present |
| `met3d` | Meteorology on the same 36 levels as `chm` | `zl` (layer height, m) | 36 model levels | 2020 to present |

Any variable the server has can be requested (`chm` / `chm_sfc` / `met3d` have many more): an unknown name fails the request and lists
what is available. Pair a `chm` profile with `met3d`'s `zl` to get altitudes for the `lev` columns.

### Modes

| Mode | Description | Date range |
|------|-------------|------------|
| `assim` | Replay/assimilation (historical) | see the table |
| `fcast` | The server's rolling 5-day forecast (`.latest` dataset) | the next days only |

Only `aqc` has `fcast`; asking for it on another collection raises `ValueError`.

### Variable reference (defaults)

| Variable | Collection | Units | Description |
|----------|------------|-------|-------------|
| `o3`, `no2`, `co`, `so2`, `hno3` | aqc, chm, chm_sfc | mol/mol (ppbv by default) | Gas mixing ratios |
| `pm25_rh35_gcc` | aqc | µg/m³ | PM2.5 at 35% RH |
| `t2m`, `u10m`, `v10m`, `q2m`, `zpbl`, `ps` | met | K, m/s, m/s, kg/kg, m, Pa | 2-m temperature, 10-m winds, 2-m humidity, boundary-layer height, surface pressure |
| `zl` | met3d | m | Mid-layer height of each model level |

---

## Result shape and units

`download()` returns an `atmoz_dataset` (`datatype="model"`):

| Collection | `ds.data` | Each DataFrame |
|------------|-----------|----------------|
| `aqc`, `chm_sfc`, `met` (surface) | `{"aqc_assim": DataFrame}` | UTC `time` index, one column per variable |
| `chm`, `met3d` (profiles) | `{"chm_assim_o3": DataFrame, ...}` | UTC `time` index, one column per model level (`lev`), one frame per variable |

`ds.metadata` has `lat` / `lon` (the 0.25° grid cell actually used), `lat_requested` / `lon_requested`, `units`, `start`, `end`,
`collection`, `mode`. Each DataFrame's `.attrs` repeats the grid cell and the units. A profile frame has the same shape the
TOLNet curtain plot takes (time index, level columns), so it can be plotted next to a lidar curtain.

**Times** are the middle of each hourly average (`00:30`, `01:30`, ...), UTC.

**Units.** The server gives gas species (`o3`, `no2`, `no`, `co`, `so2`, `hno3`, `nh3`, `hcho`, `ch4`, `isop`, `pan`, `h2o2`) in mol/mol. `download()` converts them to
**ppbv** by default; pass `units="mol/mol"` for the raw values. Every other variable (PM2.5 in µg/m³, the met variables, other species) is returned as served. `fetch()` files are always raw.

---

## `download(lat, lon, start_date, end_date, ...)`

```python
ds = GEOS_CF.download(
    39.055, -76.878, "2025-01-01", "2025-12-31",
    collection="aqc", mode="assim", variables=["o3", "no2"],
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `lat`, `lon` | `float` | required | Site, decimal degrees. The nearest 0.25° grid cell is used. |
| `start_date`, `end_date` | `str \| date` | required | Inclusive UTC days: `"YYYY-MM-DD"`, `"YYYYMMDD"`, `date` or `datetime`. |
| `collection` | `str` | `"aqc"` | `"aqc"`, `"chm"`, `"chm_sfc"`, `"met"` or `"met3d"`. |
| `mode` | `str` | `"assim"` | `"assim"` (historical) or `"fcast"` (forecast, `aqc` only). |
| `variables` | `list[str]` | collection defaults | Lower-case names; an unknown one fails the request. |
| `units` | `str` | `"ppbv"` | `"ppbv"` or `"mol/mol"` (see above). |
| `max_workers` | `int` | `1` | Months fetched at once. Server access is serialized anyway (the netCDF libraries are not thread-safe). |
| `use_cache`, `refresh`, `cache_dir` | | `True`, `False`, shared cache | See below. |
| `silent`, `show_traceback` | `bool` | `False` | Progress bar / full tracebacks. |

**Raises** `ValueError` for a bad request (before any network call) and `DownloadError` if any month fails; nothing is returned,
but finished months stay cached.

### Monthly requests and a flaky server

The OPeNDAP server caps the size of a time slice, so a range is split into calendar months, one request each. Measured live: about 15 s
for two days of `aqc`, about 1 min for one day of 36-level `chm`, so a year of profiles takes a long while the first time (and is instant
afterwards). The NASA server regularly answers `502 Proxy Error` / "DAP server error", so each request is retried 4 times with a growing pause
before it fails the download. Its time axis (GrADS, `days since 1-1-1`) is rebuilt from the axis' own attributes, because xarray cannot decode
it and the transferred values were seen damaged.

### Cache

On by default, in the shared atmoz cache (`~/.atmoz/cache/geos_cf`; `atmoz.cache_info()` shows its size, `atmoz.clear_cache("geos_cf")`
empties it). One parquet file per month of data, keyed by the grid cell, collection, mode, dates and variables, so sites in the same
grid cell share files. Assimilation data older than two weeks is treated as final; newer data and all forecasts are re-downloaded
after three hours. `refresh=True` forces a download, `use_cache=False` ignores the cache.

### `GEOS_CF_QUERY`

The request is validated by a Pydantic model before any network call: `lat` in [-90, 90], `lon` in [-180, 180], `start_date <= end_date`,
and `mode` must exist for the collection (`chm` and `met` are `assim` only).

---

## Changes from the old API

`GEOS_CF().fetch(...)` (which stored xarray Datasets in `self.data`), `force_download` and `add_altitude_coords` are gone. Use
`GEOS_CF.download(...)`; convert to xarray yourself with `df.to_xarray()` if you want it. To compare with TOLNet, call
`GEOS_CF.download(...)` at the instrument's coordinates (`ds.metadata["info"]` of a TOLNet dataset has them).
