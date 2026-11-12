# Caching

Downloads are plain files you can inspect, copy or delete by hand. Default location: `~/.atmoz/cache/`
(`ATMOZ_CACHE_DIR` or `ATMOZ_HOME` to move it).

| Namespace | Contents | Policy |
|-----------|----------|--------|
| `aqs/` | EPA AQS zips (`hourly_44201_2025.zip` + a `.meta.json` with `ETag` / `Last-Modified`) | Years before last year are frozen. The current and previous year are re-checked at most daily with a conditional request, so only changed files are downloaded again. |
| `airnow_api/` | One Parquet file per finished AirNow request window | Windows that ended more than 72 h ago. Recent windows are never cached (AirNow still revises them). The key is not part of the cache key. |

Downloads are streamed to `<name>.part` and atomically renamed, and the byte count is checked, so an
interrupted download never leaves a file that looks complete.

## Controls

Per call: `use_cache=False` (keep nothing), `refresh=True` (download again), `max_age=` (AQS: `"auto"`,
a `timedelta`, or `None` for "never re-check"), `cache_dir=` (AQS).

```python
import atmoz
atmoz.cache_dir("aqs")      # Path
atmoz.cache_info()          # {"aqs": {"files": 4, "bytes": 96693640}}
atmoz.clear_cache("aqs")    # bytes freed; atmoz.clear_cache() clears everything
```
