# Changelog

All notable changes to atmoz are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (while below 1.0, minor versions may change APIs).

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

### Known issues
- `AirNow.download()` returns different types for `endpoint="airnow"` (`atmoz_dataset`) and `"aqs"` (`dict` of DataFrames); to be unified.
- `atmoz.resources.atmoz_dataclasses` references a `ureg` unit registry whose definition is commented out, so its unit conversion raises `NameError`.
- AirNow window logic assumes the API returns the `endDate` hour (inclusive); see `AirNow.window_end_inclusive`.
- `EPA_PARAMETERS` (which parameter has which resolutions) has not been re-checked against EPA's current file list.
- GEOS-CF keeps its own `cache_dir` instead of the shared cache.
