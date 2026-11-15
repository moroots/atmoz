# Changelog

All notable changes to atmoz are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (while below 1.0, minor versions may change APIs).

## [Unreleased]

### Added
- `atmoz.plot` / `atmoz.plotting`: opinionated plots with one entry point. `az.plot("curtain", df)` and
  `az.plot.curtain(df)` are the same call; `curtain`, `profile`, `windrose` (windrose) and `site_map` (plotly) so far.
  Matplotlib plots return `(fig, ax)`; plotly plots return the `Figure`. Unknown names suggest the closest match.
- `atmoz.plotting.styles`: immutable `Style` objects (rcParams, colour-blind-safe colour cycle, figure size, artist
  params, plotly layout). Pass `style=` (object, name, dict, or `None`) and `figsize=` to any plot. Style is applied per
  call with `rc_context`, never to the global matplotlib settings.
- Ships Inter (SIL Open Font License, license included) in `atmoz/assets/fonts/`, registered automatically; DejaVu Sans fallback.
- `plots` extra (plotly, windrose, geopandas, shapely).

### Changed
- `TOLNET.plot_curtain`, `makePlots.vertical_profile`, `plot_curtain`, `plot_wind_rose` now return `(fig, ax)` and no
  longer call `plt.show()`/`plt.close()`; pass `show=True` to `TOLNET.plot_curtain` for the old behaviour in scripts.
  `makePlots.site_map` returns the plotly figure instead of showing it.
- Curtain theme: DejaVu/Inter-style fonts instead of `Courier New`, figure 11 x 4.4 in instead of 20 x 6, smaller labels.

### Fixed
- `merge_dicts` shared nested dicts with the module defaults, so `TOLNET.plot_curtain` lost its colorbar label and ticks
  after the first call. It now deep-copies, and `apply_plot_params` no longer mutates its input.
- `makePlots.site_map` raised `TypeError` (`*params["map"]` instead of `**params["map"]`).
- `colorbars.tolnet_ozone` no longer uses the deprecated `set_under` / `set_over`.

## [0.1.0] - 2026-11-12

First tagged release.

### Added
- `atmoz.resources.credentials`: a `.atmoz.env` secret store with `set_secret` / `get_secret` / `delete_secret`
  and `AirNow.set_api_key()`; owner-only file permissions; lookup order argument > environment > project file > user file.
- `atmoz.resources.cache`: shared on-disk file cache (`atmoz.cache_dir`, `cache_info`, `clear_cache`) with streamed,
  atomic, size-checked downloads and `ETag` / `Last-Modified` revalidation. Used for EPA AQS zips and finished AirNow windows.
- `atmoz.resources.urls`: one `build_url` / `redact` used by every data source.
- `EPA_PreGen.fetch()` and `read_aqs_zip()` for downloading and parsing separately; `columns=` and `bbox=` options to load less.
- `chunk=` option on `AirNowAPI.download` / `fetch` to choose the request window size.
- Typed exceptions: `DownloadError`, `APIError`, `MissingCredentialError`, `NoDataError`, `RequestTooLargeError`.
- `atmoz.lidar.UCN`: Unified Ceilometer Network (Hampton University) data, levels L2 (default) and L3, one file per
  ceilometer per UTC day. `UCN.download()` returns an `atmoz_dataset` of DataFrames (one row per time step, profile variables
  as list columns), `UCN.fetch()` the cached parquet paths, `UCN.available()` a size preview, `UCN.sites()` the ceilometer list,
  `UCN.stack()` / `UCN.backscatter()` helpers. The cache keeps one compact parquet file per day (about half the NetCDF size).
  Gentle on the server (2 workers, request pacing, `atmoz/<version>` User-Agent, never re-downloads final days) and bounded:
  requests over ~1 GB raise `RequestTooLargeError` unless `allow_large=True`. L0b is not supported yet.
- `SessionHandler(user_agent=...)`; the shared session now identifies itself as `atmoz/<version>`.
- Test suite (offline) and GitLab CI (Python 3.11-3.13, lint, wheel check); `py.typed`; project metadata and classifiers.
- Optional dependency groups: `surface`, `lidar`, `models`, `nasa`, `legacy`, `all`, `dev`.

### Changed
- GEOS-CF `fetch()` caches NetCDF results in the shared atmoz cache by default (`use_cache=False` to opt out; `cache_dir=` still overrides).
- **`atmoz.surface` restructured.** One module per backend, with the same process and names:
  `AQS.py` (`EPA_PreGen`, was `epa_pregen`), `AirNow_API.py` (`AirNowAPI`, `AirNowParams`, `make_windows`) and
  `AirNow.py` (the `AirNow` facade). `AirNow.download(endpoint=...)` is the single entry point and always returns an
  `atmoz_dataset` (AirNow API: `.data` is a DataFrame; AQS: `.data` is `{filename: DataFrame}`);
  `AirNow.fetch(endpoint=...)` returns raw files `{name: path}`. `import_data` / `download_data` are now `download` / `fetch`.
  Dead state (`_data`, `_queries_strings`, `_process`, `_nest`) is removed.
- **Package-wide naming** (see `docs/conventions.md`): every source has `download()` and `fetch()`; `TOLNET.import_data()` is now
  `TOLNET.download()` and `TOLNET.download()` (to disk) is now `TOLNET.fetch()`.
- Shared download plumbing (thread pool, all-or-nothing errors, shared HTTP sessions) moved to `atmoz.resources.utilities`.
- **Requires Python 3.11+** (was declared 3.8+, but the code already needed 3.11).
- Core install is much lighter; geopandas, plotly, xarray, earthaccess, h5py, windrose etc. moved to extras.
  `import atmoz` no longer imports them; modules raise a clear message naming the extra to install.
- EPA AQS files stream to disk instead of being buffered in memory and are parsed with Arrow's multi-threaded CSV reader
  (about 4-5x faster, 3-4x less memory on national hourly files). `Latitude` / `Longitude` are `float64` (exact, signs kept),
  date columns are datetimes, low-cardinality text columns are `category`.
- AirNow requests use the shared session pool (retries, timeouts, connection reuse) on a thread pool; the previous
  `ProcessPoolExecutor` in the old `download_data` is gone.
- AirNow request windows no longer overlap, and `start_date == end_date` returns that day. A bare end date now includes that whole day.
- Downloads are all-or-nothing: any failure raises `DownloadError` instead of printing and returning partial results.
- `silent` now consistently means "no progress bar / no warnings" (default `False`).
- `EPA_PreGen._validate` (was `_parameters_validator`) now accepts non-numeric codes with several resolutions, scalar `int` years, and mixed-case names;
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
- `dataframe.convert_units()` and unit-aware arithmetic raised `NameError` (the pint registry was never defined); it is now built lazily and includes ppbv/ppmv/pptv.
- API keys can no longer appear in exception messages, tracebacks, reprs, cache keys or saved Parquet metadata.
- Package data (TOLNet watermark) is now included in the wheel; dev files are not.
- `"Time Loca"` typo in the AQS dtype table, double slash in AQS URLs, `p` reassigned inside the validator loop.

### Known issues
- GEOS-CF module (`atmoz.models.geos_cf`): the previous API is broken and the module is being redesigned (planned after 0.1.0).
- AirNow window logic assumes the API returns the `endDate` hour (inclusive); see `AirNow.window_end_inclusive`.
- `EPA_PARAMETERS` (which parameter has which resolutions) has not been re-checked against EPA's current file list.
- GEOS-CF keeps its own `cache_dir` instead of the shared cache.
