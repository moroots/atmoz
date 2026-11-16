# API design: AtmozFrame / AtmozDataset (agreed decisions)

Status: implemented. This is the design brief the migration followed.

## Containers

* `AtmozFrame` (replaces `AtmoFrame`): one table. A pandas DataFrame plus
  * time is its own axis, `t`: a reference to the DataFrame's `DatetimeIndex` (not a column). Long-format profiles repeat a timestamp once per altitude, so the index may have duplicates;
  * `t` is required: every frame has a tz-aware UTC `DatetimeIndex`. Sources convert on load; AQS local standard time is kept in a `local_time` column (MDA8 uses it);
  * the other axes are column names: `lat`, `lon`, `alt` (altitude, km). `x`/`y`/`z` are dropped. Pressure and the like are ordinary columns;
  * `units`: `{column: unit}`, the only source of truth for units (no per-row `unit` column),
  * `metadata`: a dict that is stored on the frame (today the argument is silently dropped: fix).
* `AtmozDataset` (replaces `AtmoDataset`): a named collection of `AtmozFrame`s.
  * all frames share one axis schema (lat/lon/alt column names); units are per frame (one dataset can hold ppb ozone and ug/m3 PM2.5); time range inferred from the `t` index;
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

1. `atmoz.collocate(obs, model)`: align a model to observations in time and altitude (surface against profile allowed). Done.
2. `AtmozDataset.inventory()` / `AtmozFrame.location`: lat, lon and units from any dataset, plus the conventions doc. Done.
3. Docs and a quickstart: done. An example notebook and demo helpers remain.

## Out of scope for now

Publishing to PyPI and release housekeeping. Version is 0.1.1 (nothing has been published to PyPI). The marama_2026 analysis lives in its own repo.

## Altitude references

`alt` is in km. `metadata["alt_reference"]` is `"asl"` or `"agl"`, and `metadata["site_elevation_km"]` converts between them
(`AtmozFrame.with_alt_reference`). TOLNet is asl (instrument elevation from the file), GEOS-CF is agl. `collocate` converts to agl when
they differ and raises when a reference or elevation is missing.

## Next scope (agreed 2026-11-16)

* `atmoz.collocate`: a list of frames or datasets; the first is the anchor; datasets pair by nearest site within `max_distance_km`;
  vertical `method="auto"` (interpolate when the other profile is coarser, bin-mean when it is finer); time window as well as point match.
* `Sonde.read(path)`: NASA LaRC STAQS `.ict` files to an AtmozDataset (one frame per flight, per-row lat/lon, asl).
* `atmoz.stats`: `compare`, `exceedances`, `diurnal`, `trend`. `atmoz.vertical`: `layer_mean`, `mean_profile`.
* `atmoz.met` (extra `atmoz[met]`): theta, dewpoint, PBL height, wind components, Skew-T plot.
* New plots follow the existing styled pattern. All of it is part of the stable 0.1.1 surface, so only what is solid ships.
* Bar: offline tests plus one real-data check per source.
