"""Plumbing shared by every data source (AirNow API, EPA AQS files, ...).

Each source follows the same process, with the same names (see ``docs/conventions.md``):

    _validate -> _url -> _fetch_one -> _parse_one -> (_load_one) -> _assemble

and the same two public entry points: ``download()`` (returns an ``atmoz_dataset``) and
``fetch()`` (raw files on disk, no parsing). The parts that are identical for every
source (thread pool, all-or-nothing failure, shared HTTP sessions) live here.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from atmoz.exceptions import DownloadError
from atmoz.resources.parallelExecutor import JobResult, ParallelExecutor
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.urls import redact

PathLike = Union[str, Path]
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
Job = Mapping[str, Any]

_lock = threading.Lock()
_default_handler: Optional[SessionHandler] = None


def as_list(value: Any) -> list:
    """``"a"`` -> ``["a"]``, ``2024`` -> ``[2024]``, any iterable -> ``list(iterable)``."""
    if isinstance(value, (str, bytes, int, float)) or value is None:
        return [value]
    return list(value)


def default_session_handler() -> SessionHandler:
    """One HTTP session pool shared by every source (each thread gets its own ``requests.Session``).

    Connection pooling, retries on 429/5xx and a default timeout come with it.
    """
    global _default_handler
    with _lock:
        if _default_handler is None:
            from atmoz import __version__
            _default_handler = SessionHandler(
                pool_connections=10, pool_maxsize=10, max_retries=3, backoff_factor=0.5, timeout=(5.0, 60.0),
                user_agent=f"atmoz/{__version__}",
            )
        return _default_handler


def run_jobs(fn: Callable[[Job], Any],
             jobs: Sequence[Job],
             *,
             key: Callable[[Job], Hashable],
             describe: Callable[[Job], str],
             noun: str = "download",
             desc: str = "Downloading",
             max_workers: int = 5,
             silent: bool = False,
             show_traceback: bool = False,
             secrets: Sequence[Optional[str]] = ()) -> Dict[Hashable, Any]:
    """Run ``fn(job)`` for every job on a thread pool. **All-or-nothing.**

    Returns ``{key(job): result}`` in the order the jobs were given (not completion order).
    If any job fails, raises :class:`~atmoz.exceptions.DownloadError` naming every failure
    (``describe(job)``; *secrets* are scrubbed from the messages) and returns nothing.
    Work that finished before the failure is whatever ``fn`` itself saved (e.g. the file cache).
    """
    def work(**job):
        return {"key": key(job), "value": fn(job)}

    executor = ParallelExecutor(
        max_workers=max_workers, desc=desc, mode="thread",
        show_traceback=show_traceback, progress=not silent, print_failures=False,
    )
    result: JobResult = executor(work)(list(jobs))
    if result.failed:
        lines = [f"  {describe(f.job)}: {redact(f.exc, *secrets)}" for f in result.failed]
        raise DownloadError(
            f"{len(result.failed)} of {len(jobs)} {noun}(s) failed (nothing returned):\n" + "\n".join(lines),
            failed=[(f.job, f.exc) for f in result.failed],
        )
    return {key(job): result.succeeded[key(job)] for job in jobs}
