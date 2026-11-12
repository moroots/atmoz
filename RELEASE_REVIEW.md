# atmoz 0.0.2: release-readiness review

**Verdict: not ready for a public release.** The core design is reasonable. The blockers below are mostly packaging and hygiene, and they're quick to fix. A wheel was built to check packaging; no tests were run, since there are none.

## Release blockers

1. **`import atmoz` fails on Python < 3.11.** `AirNow.py` does `from datetime import UTC` (3.11+). `sessionHandler.py` uses `float | tuple | None` annotations (3.10+). But `pyproject.toml` says `requires-python = ">=3.8"`. Set `>=3.11` (or `>=3.10` and swap `UTC` for `timezone.utc`). `.python-version` says 3.13.
2. **Package data is missing from the wheel.** The built wheel contains only `.py` files. `Watermark_TOLNet.png` isn't shipped, so `get_asset()` returns `None` and TOLNet plots lose their watermark. The `package-data` entry points at `atmoz.schemas`, which doesn't exist. Point it at `atmoz/assets/**` and anything else loaded at runtime.
3. **Dev and scratch files ship in the wheel.** These include `dev_TOLNET.py`, `temp_TOLNet.py` and `dev_GEOS_CF.py`. Also tracked in git:
   - committed `__pycache__/*.pyc` files (the `.gitignore` rule came later)
   - `atmoz/config/atmoz.paths`, a SQLite file that's also covered by the `config/` ignore rule
   - `resources/everything.pkl.gz`

   Remove these or move them to a `dev/` folder outside the package.
4. **No tests and no CI.** There's no `tests/`, no `.github/`, and no pytest config. Most of the code is network-bound, so mock with `responses` or `pytest-httpx`. First tests worth writing:
   - `_parameters_validator`
   - `AirNowParams`
   - `_metadata` and `_nest`
   - the session handler
5. **`uv.lock` is both tracked and gitignored.** Pick one. For a library, ignoring it is fine; run `git rm --cached uv.lock`.
6. **Dependencies are unpinned and incomplete.**
   - `python-dateutil` is imported in `TOLNet.py` but isn't declared (it comes in transitively via pandas).
   - `nbconvert`, `nbformat` and `ipykernel` are hard dependencies but look like they're only needed for the `atmoz-project` CLI. Make them an extra.
   - `earthaccess`, `geopandas`, `plotly`, `windrose` and `h5py` are heavy for a core install. Consider extras such as `atmoz[lidar]`, `atmoz[surface]` and `atmoz[nasa]`.
   - Several deps have no lower bounds (`pandas`, `numpy`, `matplotlib`).
7. **Docs are stubs.** `project/mkdocs.yml` is the template ("My Docs") and `index.md` is the default MkDocs page. The README's feature list omits AirNow, GEOS-CF and the AtmoFrame/AtmoDataset classes, and the install section is git-only. Add a quickstart for each module; the surface README is the only good one.
8. **Smaller metadata gaps.**
   - No `CHANGELOG`.
   - No `[project.urls]`, classifiers or license field.
   - Typo in the description ("Atmoshperic").
   - Version is bumped to 0.0.2 but uncommitted.
   - No `py.typed`.
   - `__init__.py` imports `AirNow` eagerly, so importing the package pulls in geopandas and plotly.

## Keyring removal

Keyring is used in only two spots in `AirNow.py`: `__init__` (the `get_password` call) and `_pull` (the `set_password` call). The `input()` prompts are the other problem: they hang in notebooks, CI and scripts. Replace all of it with this resolution order:

1. the explicit `api_key=` argument
2. the `AIRNOW_API_KEY` environment variable
3. a `.env` file or `~/.config/atmoz/config.toml` (optional)
4. otherwise raise a clear `ValueError` that says how to set the key. Never prompt.

Then:
- Delete `import keyring`.
- Drop the dependency from `pyproject.toml`.
- Update the "system keyring" text in `atmoz/surface/README.md`.
- To keep keyring support for those who like it, make it an optional extra (`atmoz[keyring]`) behind a guarded import.

Related: `_build_url` puts the key into a hand-built query string that isn't URL-encoded. Pass `params=` to `requests` instead. Also make sure the key never appears in error messages. `_pull` currently raises with `response.text`, and URLs containing the key can end up in logs.

## `AirNow.download()` performance

`download()` has two backends, with different problems.

**`endpoint="aqs"` (`epa_pregen`), the likely slow path:**
- `utils.download_zip` buffers the whole zip in memory before reading it. National hourly ozone files are hundreds of MB once unzipped.
- The `dtypes` dict reads `Latitude`, `Longitude` and `Date Local` as `str`, which is memory-heavy and slow. Use `float32` for lat/lon and `category` for repeated columns (`State Name`, `County Name`, `Units of Measure`, `Method Name`).
- Reading all 24 columns when you need a handful is wasteful. Add a `usecols=` option, and try `engine="pyarrow"`.
- There's no on-disk cache, so every call re-downloads. Cache the zips and parsed parquet under a cache dir (for example `platformdirs`). Past years' files are immutable, so only the current year ever needs a refresh.
- Typo: `"Time Loca"` should be `"Time Local"`, so that dtype is silently ignored.
- Add a `bbox`/`state` filter applied during parsing (chunked `read_csv`), so the whole US isn't held in memory when only one region is needed.
- Threads are fine for I/O, but `max_workers=5` against one host rarely helps. The parse step is CPU-bound and gets no benefit from threads.

**`endpoint="airnow"` (`import_data`):**
- It issues one request per day, sequentially. Reuse the existing `ParallelExecutor` with a small pool (3 to 4, mindful of AirNow's rate limit of about 500 requests/hour).
- It uses bare `requests.get` with no timeout, no retry and no session reuse. `SessionHandler` already exists, so use it here.
- It can request wider windows per call. AirNow allows multi-day ranges, so chunk by roughly a week instead of a day.
- A request failure aborts the whole run with no partial result.

**Correctness bugs noticed in the same code:**
- `download_data` submits `self._pull` to a `ProcessPoolExecutor`. That pickles the whole object per task, and it's pointless for I/O, so use threads.
- Consecutive day windows overlap at the boundary hour, which gives duplicated rows. Dedupe, or make the end exclusive.
- `start_date == end_date` leaves the job list empty, so `list_of_dfs[0]` raises `IndexError`.
- In `_parameters_validator`, `p` is reassigned inside the loop, and `param_code_map` is keyed by code, so `pm25` and `pm25_frm` collide.
- The URL is built as `f"{base_url_aqs}/{...}"`, which gives a double slash.
- `download()` is annotated `-> Path` but returns a dataset or dict.
- Silent `except Exception: print(...)` hides failures. Return or raise a structured failure list.

## Suggested order

1. Fix the Python version, package data and the tracked junk files (about an hour).
2. Remove keyring, add env-var auth and fix URL encoding.
3. Add the minimal tests plus a GitHub Actions matrix (3.11 to 3.13), with `ruff`.
4. AQS performance: streaming to disk, dtypes, `usecols` and the cache.
5. Docs and README, extras split, changelog, then tag 0.1.0.
