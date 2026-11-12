# Configuration

atmoz keeps its per-user files in one folder, `~/.atmoz/` (override with `ATMOZ_HOME`):

```
~/.atmoz/
├── .atmoz.env     API keys (written by atmoz.set_secret / AirNow.set_api_key)
├── atmoz.paths    named paths per machine (path_manager)
└── cache/         downloaded files, see Caching
```

## API keys: `.atmoz.env`

Secrets live in a plain `KEY=value` file named **`.atmoz.env`** (deliberately not `.env`, so it never collides
with another tool). Lookup order:

1. a value you pass explicitly (`AirNow(api_key="...")`)
2. the process environment (`export ATMOZ_AIRNOW_API_KEY=...`)
3. the nearest `.atmoz.env`, walking up from the current directory (per-project key)
4. `~/.atmoz/.atmoz.env` (per-user key)

`ATMOZ_ENV_FILE=/path/to/file` forces one specific file.

### Saving a key (replaces the old keyring)

```python
from atmoz.surface import AirNow
AirNow.set_api_key()                           # hidden prompt, saved for all projects
AirNow.set_api_key("abc123", scope="project")  # saved to ./.atmoz.env for this project only
```

or generally:

```python
import atmoz
atmoz.set_secret("airnow", "abc123")      # -> Path of the file written
atmoz.get_secret("airnow")
atmoz.delete_secret("airnow")
```

The file is created with owner-only permissions (`0600`). **Do not commit it**; atmoz's `.gitignore` already
excludes `.atmoz.env`, but add it to yours.

atmoz never prompts for a missing key (that hangs notebooks and CI); it raises `MissingCredentialError`
that says exactly how to set it.

### Keeping the key out of logs

atmoz removes the key from every exception message, `repr`, and saved file. AirNow only accepts the key as a
URL query parameter, so one limit remains: if *you* turn on `DEBUG` logging for `urllib3`
(`logging.getLogger("urllib3").setLevel(logging.DEBUG)`), it logs request paths including the key. Leave that
logger at INFO or above when sharing logs.

## NASA EarthData

`atmoz.data_access.NASA.EarthData` (used for TOLNet HDF downloads) reads NASA's standard `~/.netrc`
(`machine urs.earthdata.nasa.gov login <user> password <pass>`). If you pass a username and password it writes
that file for you. This is NASA's own convention; protect the file the same way (`chmod 600 ~/.netrc`).

## Environment variables

| Variable | Effect |
|----------|--------|
| `ATMOZ_HOME` | Per-user folder (default `~/.atmoz`) |
| `ATMOZ_CACHE_DIR` | Cache location (default `$ATMOZ_HOME/cache`) |
| `ATMOZ_ENV_FILE` | Use exactly this secrets file |
| `ATMOZ_AIRNOW_API_KEY` | AirNow key |
