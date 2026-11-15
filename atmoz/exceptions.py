"""Exceptions raised by atmoz."""

from __future__ import annotations

from typing import Any, List, Optional


class AtmozError(Exception):
    """Base class for every atmoz-specific exception."""


class MissingCredentialError(AtmozError, ValueError):
    """A required API key / secret could not be found."""


class DownloadError(AtmozError, RuntimeError):
    """One or more downloads failed.

    Downloads are all-or-nothing: no partial result is returned. Anything that
    had already been fetched is in the file cache, so simply re-running the
    call only repeats the failed pieces.

    Attributes
    ----------
    failed:
        One ``(job, exception)`` pair per failed download. ``exception``
        messages never contain API keys.
    """

    def __init__(self, message: str, failed: Optional[List[Any]] = None) -> None:
        super().__init__(message)
        self.failed = failed or []


class NoDataError(AtmozError, LookupError):
    """The source answered normally but has no data for the request (e.g. no file for that day).

    Distinct from :class:`DownloadError` (something *failed*): here nothing is wrong, the data just isn't there.
    """


class RequestTooLargeError(AtmozError, ValueError):
    """A request would download or hold more than the safety thresholds allow.

    Nothing was downloaded. Narrow the request, or pass ``allow_large=True`` if you really want it.
    """


class APIError(AtmozError):
    """A web API answered with an error (or could not be reached).

    Messages are scrubbed of API keys before the exception is created.
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
