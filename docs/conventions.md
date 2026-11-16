# Conventions

Every data source in atmoz (AirNow API, EPA AQS files, TOLNet, UCN and GEOS-CF) follows the same
process with the same names, so you only learn it once.

## Public entry points

| Method | Returns | What it does |
|--------|---------|--------------|
| `download(...)` | `atmoz_dataset` | Get the data, parsed, ready to use. |
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
| 5 | `_assemble` | Results -> `atmoz_dataset`. |

The same helper runs steps 3-4 for every source: `atmoz.resources.utilities.run_jobs`. It uses a thread pool
(downloads wait on the network) and is **all-or-nothing**: if any job fails it raises `DownloadError` and returns nothing.

## Naming

* Modules: one per backend, named for the data source (`AQS.py`, `AirNow_API.py`); classes `EPA_PreGen`, `AirNowAPI`.
* Shared plumbing lives in `atmoz.resources` (`utilities`, `cache`, `credentials`, `urls`, `sessionHandler`).
* Common options keep the same names everywhere: `use_cache`, `refresh`, `cache_dir`, `max_workers`, `silent`, `show_traceback`, `session_handler`.

## Status

AirNow API, EPA AQS, TOLNet, UCN and GEOS-CF follow this process and return an `atmoz_dataset` with `.data == {name: DataFrame}`
and a dict in `.metadata`.
