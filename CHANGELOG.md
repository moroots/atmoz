# Changelog

All notable changes to atmoz are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (while below 1.0, minor versions may change APIs).

## [Unreleased]

Everything below ships in 0.1.0, which has not been published yet.

### Added
- **One data shape for every source.** `atmoz.AtmozFrame` (a pandas DataFrame with a UTC `DatetimeIndex` as time axis `t`, `lat` / `lon` / `alt` (km)
  axis columns, `units` per column and a `metadata` dict) and `atmoz.AtmozDataset` (a named collection of frames sharing one axis schema, units per frame).
  `ds[key]`, `ds.select("*/OZONE")`, `ds.frame` (the only frame), `ds.inventory()` (one row per frame: lat, lon, altitude range, units, time span),
  `frame.location`, `frame.convert_units(column, unit)`; pandas methods work on a frame and broadcast over a dataset (`ds.resample("1h").mean()`).
  `ppbv` is normalised to `ppb`. Every `download()` returns an `AtmozDataset`; see `docs/conventions.md` (Containers, Units, Site metadata).
- **Frame keys per source**: AirNow `"{IntlAQSCode}/{PARAMETER}"`; AQS `"{state}{county}{site}/{parameter code}"`; TOLNet one frame per file (long: `alt`, `value`, `lat`, `lon`,
  units read from the file); UCN one frame per ceilometer, day and level (`"CEILO-9_20261101_L2"`, `value` is the attenuated backscatter);
  GEOS-CF one frame per variable (`alt` in km and `lev` on `chm` / `met3d`; `chm` also reads `met3d` for the layer heights).
- `atmoz.collocate(obs, model, obs_col=, model_col=, max_gap=)`: interpolates a model frame onto an observation's times (linear in time, NaN beyond `max_gap`) and altitudes
  (per-time profile interpolation, so GEOS-CF layer heights may differ per time), converting the model to the observation unit. Returns an `AtmozFrame` with `obs` and `model`
  columns. New guide `docs/collocation.md` (TOLNet versus GEOS-CF).
- **EPA AQS** `download()` filters and caching: the parsed file is cached as parquet next to the zip; filters `bbox`, `start_date` / `end_date`, `site`, `state`, `county`
  (codes or names), `parameter`, `poc` and `where={column: value}` are applied before the per-site split. Hourly files only (UTC time from `Date GMT` + `Time GMT`,
  the local standard time stays in a `local_time` column); units are declared as the file gives them (ozone stays ppm).
- `atmoz.surface.mda8` / `mda8_aqs`: ozone 8-hour averages and the daily maximum 8-hour average (MDA8) per EPA 40 CFR 50 Appendix U
  (truncation, 6-of-8 and 0.567 ppm sum rules, 17 windows, 13-of-17 day validity, exceptional events, collocated monitors).
  `mda8_aqs` accepts a raw AQS DataFrame, an `AtmozDataset` or an `AtmozFrame`.
- `atmoz.lidar.UCN`: Unified Ceilometer Network (Hampton University) data, levels L2 (default) and L3, one file per
  ceilometer per UTC day. `UCN.download()` returns an `AtmozDataset`, `UCN.fetch()` the cached parquet paths, `UCN.available()` a size preview,
  `UCN.sites()` the ceilometer list, `UCN.stack()` / `UCN.backscatter()` helpers. The cache keeps one compact parquet file per day (about half the NetCDF size).
  Gentle on the server (2 workers, request pacing, `atmoz/<version>` User-Agent, never re-downloads final days) and bounded:
  requests over ~1 GB raise `RequestTooLargeError` unless `allow_large=True`. Verified against the live server (first download and conversion). L0b (raw 16 s profiles) is planned: `level="L0b"` raises a clear error and `UCN.planned_levels` lists it.
- **GEOS-CF** `GEOS_CF.download(lat, lon, start, end, ...)` / `GEOS_CF.fetch(...)` (classmethods). Gas species are in ppb by default (`units="mol/mol"` keeps the raw values). The cache is
  one parquet file per month (assimilation data older than two weeks is final; newer data and forecasts expire after three hours), and a bad variable or any failed month raises.
  Checked against the live server: `chm` is the real profile product (36 model levels), `chm_sfc` and `met3d` collections, the forecast uses the `.latest` dataset, the server's GrADS
  time axis is decoded from its attributes, and requests are retried because the server often answers 502.
- **TOLNet** (`atmoz/lidar/tolnet.py`): `TOLNET.download()` / `fetch()` are classmethods using the shared session pool and file cache, all-or-nothing (`DownloadError`; `NoDataError` when nothing
  matches). `fetch()` takes `min_date` / `max_date` / `**query` and `dest_dir=` (default: the cache). `TOLNET.group(ds)`, `TOLNET.open_hdf()` and `TOLNET.plot_curtains(ds)`;
  `TOLNET.plot_curtain` delegates to `atmoz.plotting.curtain`. Value and altitude units are read from the file (m is converted to km).
- `atmoz.plot` / `atmoz.plotting`: opinionated plots with one entry point. `az.plot("curtain", df)` and
  `az.plot.curtain(df)` are the same call; `curtain`, `profile`, `windrose` (windrose) and `site_map` (plotly) so far.
  Matplotlib plots return `(fig, ax)`; plotly plots return the `Figure`. Unknown names suggest the closest match.
- `atmoz.plotting.styles`: immutable `Style` objects (rcParams, colour-blind-safe colour cycle, figure size, artist
  params, plotly layout). Pass `style=` (object, name, dict, or `None`) and `figsize=` to any plot. Style is applied per
  call with `rc_context`, never to the global matplotlib settings.
- Ships Inter (SIL Open Font License, license included) in `atmoz/assets/fonts/`, registered automatically; DejaVu Sans fallback.
- `atmoz.resources.utilities`: the plumbing every data source shares (thread pool with all-or-nothing failure, shared HTTP sessions).
- Typed exceptions `NoDataError` and `RequestTooLargeError`; `SessionHandler(user_agent=...)`, and the shared session now
  identifies itself as `atmoz/<version>`.
- Extras: `plots` (plotly, windrose, geopandas, shapely); `lidar` includes `xarray`, `netcdf4` and `h5py`.
- `docs/conventions.md` (the package-wide process, naming, containers, units, site metadata and collocation), `docs/collocation.md`, `docs/plotting.md`, and
  `docs/references/epa-40cfr50-appendix-u/`: the EPA rules for 8-hour ozone averages and the daily maximum (the spec for `atmoz.surface.mda8`), as text and JSON.
- GitHub Actions CI (`.github/workflows/ci.yml`) next to the GitLab CI; test notebooks in `dev/notebooks/`.
- Tests for the containers, collocation, AirNow, AQS, TOLNet, UCN, plotting, unit conversion and the GEOS-CF cache.

### Changed
- **`AirNow.download(endpoint=...)` is the single entry point** and returns an `AtmozDataset` for both endpoints (one frame per site and parameter); `ds.metadata` is a dict
  (`endpoint`, `files`, `rows`; AirNow API adds `start`, `end`, `query` and the per-site table `sites`; AQS adds `filters`). `AirNow.fetch(endpoint=...)` returns raw files `{name: path}`.
- **`atmoz.surface` restructured.** One module per backend, with the same process and names:
  `AQS.py` (`EPA_PreGen`), `AirNow_API.py` (`AirNowAPI`, `AirNowParams`, `make_windows`) and `AirNow.py` (the `AirNow` facade).
  `import_data` / `download_data` are now `download` / `fetch`, and `_parameters_validator` is `EPA_PreGen._validate`.
- **Package-wide naming** (see `docs/conventions.md`): every source has `download()` (parsed, ready to use) and `fetch()` (raw files only); `TOLNET.import_data()` is now `TOLNET.download()`.
- `TOLNET.plot_curtain`, `makePlots.vertical_profile`, `plot_curtain`, `plot_wind_rose` return `(fig, ax)` and do not call `plt.show()`/`plt.close()`; pass `show=True`
  to `TOLNET.plot_curtain` in scripts. `makePlots.site_map` returns the plotly figure instead of showing it.
- Curtain theme: DejaVu/Inter-style fonts instead of `Courier New`, figure 11 x 4.4 in instead of 20 x 6, smaller labels.
- The HDF helpers moved from `atmoz.resources.importData` to `atmoz.resources.hdf`.

### Removed
- `TOLNET.data` / `meta_data` / `_errors`, `change_timezone`, `tolnet_curtains` (use `plot_curtains`), `download(GEOS_CF=True)` and the curtain options `xlims`, `time_resolution`, `use_contourf`
  (slice the frame before plotting).
- The stateful `GEOS_CF().fetch()` / `.data`, `force_download` (use `refresh=True`) and `add_altitude_coords` (the `alt` column comes from `met3d`'s `zl`).

### Fixed
- `EPA_PARAMETERS` was checked against EPA's server for 2024: all 79 parameter/resolution combinations behave as the table says and no listed
  data file is missing from it.
- UCN conversion could crash the interpreter (segfault) intermittently: the HDF5 library under netCDF4 is not thread-safe and two workers converted files at once. NetCDF reads are now serialized.
- `surface.utilities.make_datetime` raised `TypeError` on the datetime `Date` columns the AQS reader returns; it now takes datetime or text dates.
- Unit conversion raised `NameError` (the pint registry was never defined); it is now built lazily and includes ppb/ppm/ppt.
- `merge_dicts` shared nested dicts with the module defaults, so `TOLNET.plot_curtain` lost its colorbar label and ticks
  after the first call. It now deep-copies, and `apply_plot_params` no longer mutates its input.
- `makePlots.site_map` raised `TypeError` (`*params["map"]` instead of `**params["map"]`).
- `colorbars.tolnet_ozone` no longer uses the deprecated `set_under` / `set_over`.

### Known issues
- AirNow window logic assumes the API returns the `endDate` hour (inclusive); see `AirNowAPI.window_end_inclusive`. Not yet checked with a live key:
  run `ATMOZ_AIRNOW_API_KEY=... python tools/verify_airnow_live.py` (3 small requests) to confirm or to get the one-line fix.

## [0.1.0] - 2026-11-12

First tagged release.

### Added
- `atmoz.resources.credentials`: a `.atmoz.env` secret store with `set_secret` / `get_secret` / `delete_secret`
  and `AirNow.set_api_key()`; owner-only file permissions; lookup order argument > environment > project file > user file.
- `atmoz.resources.cache`: shared on-disk file cache (`atmoz.cache_dir`, `cache_info`, `clear_cache`) with streamed,
  atomic, size-checked downloads and `ETag` / `Last-Modified` revalidation. Used for EPA AQS zips and finished AirNow windows.
- `atmoz.resources.urls`: one `build_url` / `redact` used by every data source.
- `epa_pregen.fetch()` and `read_aqs_zip()` for downloading and parsing separately; `columns=` and `bbox=` options to load less.
- `chunk=` option on `AirNow.import_data` / `download_data` to choose the request window size.
- Typed exceptions: `DownloadError`, `APIError`, `MissingCredentialError`.
- Test suite (offline) and GitLab CI (Python 3.11-3.13, lint, wheel check); `py.typed`; project metadata and classifiers.
- Optional dependency groups: `surface`, `lidar`, `models`, `nasa`, `legacy`, `all`, `dev`.

### Changed
- **Requires Python 3.11+** (was declared 3.8+, but the code already needed 3.11).
- Core install is much lighter; geopandas, plotly, xarray, earthaccess, h5py, windrose etc. moved to extras.
  `import atmoz` no longer imports them; modules raise a clear message naming the extra to install.
- EPA AQS files stream to disk instead of being buffered in memory and are parsed with Arrow's multi-threaded CSV reader
  (about 4-5x faster, 3-4x less memory on national hourly files). `Latitude` / `Longitude` are `float64` (exact, signs kept),
  date columns are datetimes, low-cardinality text columns are `category`.
- AirNow requests use the shared session pool (retries, timeouts, connection reuse) on a thread pool; the previous
  `ProcessPoolExecutor` in `download_data` is gone.
- AirNow request windows no longer overlap, and `start_date == end_date` returns that day. A bare end date now includes that whole day.
- Downloads are all-or-nothing: any failure raises `DownloadError` instead of printing and returning partial results.
- `silent` now consistently means "no progress bar / no warnings" (default `False`).
- `_parameters_validator` now accepts non-numeric codes with several resolutions, scalar `int` years, and mixed-case names;
  unknown names raise `ValueError` with suggestions; skipped combinations warn instead of printing.
- Live TOLNet and GEOS-CF implementations renamed from `dev_TOLNET.py` / `dev_GEOS_CF.py` to `lidar/tolnet.py` / `models/geos_cf.py`
  (public imports `from atmoz.lidar import TOLNET`, `from atmoz.models import GEOS_CF` are unchanged).
- Module READMEs moved into `docs/`, which is now built with MkDocs.
- `path_manager` stores `atmoz.paths` under `~/.atmoz` instead of inside the installed package.

### Removed
- The `keyring` dependency and every interactive `input()` prompt for the AirNow key.
- The `atmoz-project` command-line scaffolder (kept on the `atmoz-project` branch) and its notebook dependencies.
- Unused dependencies: `httpx`, `scipy`, `nbformat`, `nbconvert`, `ipykernel`.
- Tracked build debris (`__pycache__`, a stray SQLite file, `everything.pkl.gz`) and `uv.lock` (now git-ignored).
- Superseded modules moved out of the package to `dev/legacy/` (old `TOLNet.py`, `geos_cf.py`, `temp_TOLNet.py`).

### Fixed
- API keys can no longer appear in exception messages, tracebacks, reprs, cache keys or saved Parquet metadata.
- Package data (TOLNet watermark) is now included in the wheel; dev files are not.
- `"Time Loca"` typo in the AQS dtype table, double slash in AQS URLs, `p` reassigned inside the validator loop.
