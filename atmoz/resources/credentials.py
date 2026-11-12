"""Tiny ``.env``-style secret store for atmoz.

Replaces the system keyring. Secrets live in a plain ``KEY=value`` file named
``.atmoz.env`` (never ``.env``, so it cannot collide with another tool).

Lookup order for :func:`get_secret`:

1. the value passed explicitly by the caller
2. the process environment, e.g. ``export ATMOZ_AIRNOW_API_KEY=...``
3. the *nearest* ``.atmoz.env`` found walking up from the current directory
   (a per-project file)
4. the user-level file ``~/.atmoz/.atmoz.env``

Set ``ATMOZ_ENV_FILE`` to force one specific file. Write secrets with
:func:`set_secret` (or ``AirNow.set_api_key()``) so you never have to hunt for
the file; it is created with owner-only permissions (``0600``).
"""

from __future__ import annotations

import getpass
import os
import stat
from pathlib import Path
from typing import Dict, Literal, Optional

from atmoz.exceptions import MissingCredentialError
from atmoz.resources.cache import atmoz_home

ENV_FILENAME = ".atmoz.env"
ENV_FILE_OVERRIDE = "ATMOZ_ENV_FILE"

#: Friendly service name -> variable name stored in the env file.
SECRET_NAMES: Dict[str, str] = {
    "airnow": "ATMOZ_AIRNOW_API_KEY",
}


def _var(name: str) -> str:
    """Resolve a friendly name (``"airnow"``) or a raw variable name."""
    return SECRET_NAMES.get(name.lower(), name)


def user_env_file() -> Path:
    """Path of the user-level secrets file (may not exist yet)."""
    return atmoz_home() / ENV_FILENAME


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """Return the env file that :func:`get_secret` would read, if any."""
    override = os.environ.get(ENV_FILE_OVERRIDE)
    if override:
        return Path(override).expanduser()
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / ENV_FILENAME
        if candidate.is_file():
            return candidate
    user = user_env_file()
    return user if user.is_file() else None


def _parse(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def get_secret(name: str, explicit: Optional[str] = None, *, required: bool = True) -> Optional[str]:
    """Look up a secret (see module docstring for the search order)."""
    if explicit:
        return explicit
    var = _var(name)
    value = os.environ.get(var)
    if value:
        return value
    path = find_env_file()
    if path is not None:
        value = _parse(path).get(var)
        if value:
            return value
    if required:
        raise MissingCredentialError(
            f"No value found for {var}. Pass it explicitly, set the {var} environment "
            f"variable, or save it once with atmoz.set_secret({name!r}) "
            f"(stored in {user_env_file()})."
        )
    return None


def set_secret(
    name: str,
    value: Optional[str] = None,
    *,
    scope: Literal["user", "project"] = "user",
    path: Optional[os.PathLike] = None,
) -> Path:
    """Save a secret so future sessions find it automatically.

    Parameters
    ----------
    name:
        Friendly name (``"airnow"``) or a raw variable name.
    value:
        The secret. If omitted you are prompted without echo (works in
        terminals and Jupyter).
    scope:
        ``"user"`` writes ``~/.atmoz/.atmoz.env`` (shared by all projects);
        ``"project"`` writes ``./.atmoz.env`` in the current directory.
    path:
        Write to this exact file instead of using *scope*.

    Returns
    -------
    Path
        The file that was written. Its mode is ``0600``.
    """
    var = _var(name)
    if value is None:
        value = getpass.getpass(f"{var}: ")
    value = (value or "").strip()
    if not value:
        raise ValueError("Refusing to store an empty secret.")
    if "\n" in value or "\r" in value:
        raise ValueError("Secrets cannot contain newlines.")

    if path is not None:
        target = Path(path).expanduser()
    elif scope == "project":
        target = Path.cwd() / ENV_FILENAME
    elif scope == "user":
        target = user_env_file()
    else:
        raise ValueError("scope must be 'user' or 'project'.")

    target.parent.mkdir(parents=True, exist_ok=True)
    values = _parse(target) if target.exists() else {}
    values[var] = value
    body = "# Written by atmoz. Keep this file private; do not commit it.\n"
    body += "".join(f"{k}={v}\n" for k, v in values.items())

    # Create with 0600 from the start so the secret is never world-readable.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    try:
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # e.g. Windows
        pass
    return target


def delete_secret(name: str, *, path: Optional[os.PathLike] = None) -> bool:
    """Remove a secret from the env file. Returns ``True`` if it was present."""
    var = _var(name)
    target = Path(path).expanduser() if path else find_env_file()
    if target is None or not target.exists():
        return False
    values = _parse(target)
    if var not in values:
        return False
    del values[var]
    body = "# Written by atmoz. Keep this file private; do not commit it.\n"
    body += "".join(f"{k}={v}\n" for k, v in values.items())
    target.write_text(body, encoding="utf-8")
    return True
