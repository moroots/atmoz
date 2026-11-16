# API design: AtmozFrame / AtmozDataset (agreed decisions)

Status: agreed, not yet implemented. This is the brief for the migration; implementation work follows it.

## Containers

* `AtmozFrame` (replaces `AtmoFrame`): one table. A pandas DataFrame plus
  * time is its own axis, `t`: a reference to the DataFrame's `DatetimeIndex` (not a column). Long-format profiles repeat a timestamp once per altitude, so the index may have duplicates;
  * `t` is required: every frame has a tz-aware UTC `DatetimeIndex`. Sources convert on load; AQS local standard time is kept in a `local_time` column (MDA8 uses it);
  * the other axes are column names: `lat`, `lon`, `alt` (altitude, km). `x`/`y`/`z` are dropped. Pressure and the like are ordinary columns;
  * `units`: `{column: unit}`, the only source of truth for units (no per-row `unit` column),
  * `metadata`: a dict that is stored on the frame (today the argument is silently dropped: fix).
* `AtmozDataset` (replaces `AtmoDataset`): a named collection of `AtmozFrame`s.
  * all frames share one schema (axes and units); time range inferred from the `t` index;
  * has `name`, `source`, `metadata` (provenance: endpoint, files, rows, query).
* `atmoz_dataset` is deleted from the public API (no users yet, so no compatibility shim). Every `download()` returns an `AtmozDataset`.
* Fixes while migrating: units accept `ppbv` as an alias of `ppb` (normalised to `ppb`); the `frames={AtmoFrame}` default becomes `None`.

## Shapes per source

| Source | Frame key | Frame shape |
|--------|-----------|-------------|
| AirNow, AQS | `site/parameter` | index t, `value`, lat/lon constant, one unit (ppb for AirNow, ppm kept for AQS and declared) |
| TOLNet, UCN, GEOS-CF profiles | file or chunk | long: index t, `alt` (km), `value` |
| GEOS-CF surface series | chunk | index t, `value` |

## AQS

Cache the raw file as parquet. Filters apply to the cached parquet before the per-site split: `bbox=(W,S,E,N)`,
`start_date`/`end_date`, `site`/`state`/`county`, `parameter`/`POC`, and any column of the file.

## After the migration

1. `collocate(lidar_frame, model_frame)`: align TOLNet and GEOS-CF in time and altitude.
2. Units/site helper: returns lat, lon and units from any dataset. Convention doc.
3. Docs and a quickstart; one example notebook; demo helpers only if time.

## Out of scope for now

Publishing to PyPI and release housekeeping. Version stays 0.1.0 until then (nothing has been published to PyPI, so 0.1.0 is free). The marama_2026 analysis lives in its own repo.
