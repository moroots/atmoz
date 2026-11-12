"""URL helpers shared by every atmoz data source.

All modules build URLs with :func:`build_url` so that quoting, trailing
slashes and query-string encoding behave the same everywhere, and log or
error output can be scrubbed with :func:`redact`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import quote, urlencode

# Query-parameter names whose values must never be printed or logged.
SECRET_PARAMS = ("api_key", "apikey", "key", "token", "access_token", "password")

_SECRET_RE = re.compile(
    r"(?i)([?&;]?(?:%s)=)[^&#\s'\"]+" % "|".join(re.escape(p) for p in SECRET_PARAMS)
)


def build_url(
    base: str,
    *segments: Any,
    params: Optional[Mapping[str, Any]] = None,
) -> str:
    """Join *base* and path *segments*, then append an encoded query string.

    Parameters
    ----------
    base:
        Scheme + host (+ optional base path), e.g. ``"https://example.com/api"``.
    *segments:
        Path pieces. Each is converted with ``str`` and percent-encoded, so a
        piece never injects extra ``/`` or ``?``. Empty / ``None`` pieces are
        skipped.
    params:
        Query parameters. ``None`` values are dropped; lists and tuples are
        joined with commas (the convention used by AirNow and TOLNet); bools
        become ``"true"`` / ``"false"``.

    Examples
    --------
    >>> build_url("https://x.org/api/", "data", 1, params={"a": [1, 2], "b": None})
    'https://x.org/api/data/1?a=1%2C2'
    """
    url = base.rstrip("/")
    for seg in segments:
        if seg is None or seg == "":
            continue
        url += "/" + quote(str(seg), safe="")
    if params:
        query = encode_params(params)
        if query:
            url += "?" + query
    return url


def encode_params(params: Mapping[str, Any]) -> str:
    """Encode *params* the same way :func:`build_url` does."""
    clean = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            value = ",".join(str(v) for v in value)
        clean[key] = value
    return urlencode(clean, quote_via=quote)


def redact(text: object, *secrets: Optional[str]) -> str:
    """Return *text* with secret query parameters and literal *secrets* masked."""
    out = _SECRET_RE.sub(r"\1***", str(text))
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
            out = out.replace(quote(secret, safe=""), "***")
    return out
