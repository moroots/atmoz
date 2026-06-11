# -*- coding: utf-8 -*-
"""
Created on 2026-06-06 22:05:07

@author: Maurice Roots

Description:
     - A short module for handling Parallel Processing (Threads and Pools) with simple decorators. Keep clean code...
"""
# %%

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypedDict
import pickle
import traceback
import warnings


class JobReturn(TypedDict):
    """The required return type for functions decorated with ParallelExecutor.

    Note: This TypedDict is used for documentation and static analysis only.
    Runtime enforcement is handled explicitly inside the executor.

    Example::

        def fetch(url: str) -> JobReturn:
            response = requests.get(url)
            return {"value": response.json(), "key": url}
    """
    value: Any
    key: Any


try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    # Warn once at import time, not on every decorated-function call.
    warnings.warn(
        "tqdm is not installed; ParallelExecutor progress bars are disabled. "
        "Install it with: pip install tqdm",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FailedJob:
    """Structured record of a single failed job.

    Attributes
    ----------
    job:
        The original job dict that was submitted.
    exc:
        The exception that was raised.
    exc_type:
        ``type(exc)`` — exposed directly so callers can filter by error
        category without an ``isinstance`` chain on ``exc``.

    Example::

        for failure in result.failed:
            if failure.exc_type is TimeoutError:
                retry_queue.append(failure.job)
    """
    job: dict
    exc: Exception
    exc_type: type

    def __repr__(self) -> str:
        return f"FailedJob(job={self.job!r}, exc_type={self.exc_type.__name__})"


class JobResult:
    """Container for the outcome of a parallel execution."""

    def __init__(self):
        self.succeeded: dict[Any, Any] = {}
        self.failed: list[FailedJob] = []

    def __repr__(self):
        return (
            f"JobResult(succeeded={len(self.succeeded)}, failed={len(self.failed)})"
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """
    A decorator that executes a function over a list of jobs in parallel,
    using either threads or processes.

    The decorated function must accept keyword arguments matching each job dict
    and return a ``JobReturn`` dict with exactly two keys: ``"value"`` (the
    result to store) and ``"key"`` (the key under which to store it).

    If the decorated function returns a dict that does not satisfy the
    ``JobReturn`` contract (missing ``"value"`` or ``"key"``), the job is
    recorded in ``JobResult.failed`` as a ``FailedJob`` rather than raising
    to the caller.

    Args:
        max_workers:     Maximum number of parallel workers.
        desc:            Label shown in the progress bar.
        mode:            ``'thread'`` (default) or ``'process'``.
                         Process mode requires the decorated function to be
                         picklable (no lambdas or closures). A picklability
                         check is performed eagerly at decoration time.
                         Note: ``ProcessPoolExecutor`` is re-created on every
                         call; for hot loops prefer ``'thread'`` or manage
                         your own process pool externally.
        show_traceback:  If ``True``, print full tracebacks on failure instead
                         of a one-line summary.
        on_error:        Optional callback invoked on each failed job with
                         signature ``on_error(job: dict, exc: Exception)``.
        warn_collisions: If ``True`` (default), emit a ``UserWarning`` when
                         two jobs produce the same key and the earlier result
                         is overwritten.

    Example::

        @ParallelExecutor(max_workers=4, desc="Fetching")
        def fetch(url: str, timeout: int = 10) -> JobReturn:
            response = requests.get(url, timeout=timeout)
            return {"value": response.json(), "key": url}

        result = fetch([
            {"url": "https://example.com", "timeout": 5},
            {"url": "https://other.com"},
        ])
        print(result.succeeded)

        # Filter failures by type:
        timeouts = [f for f in result.failed if f.exc_type is TimeoutError]
    """

    def __init__(
        self,
        max_workers: int = 5,
        desc: str = "Processing",
        mode: str = "thread",
        show_traceback: bool = False,
        on_error: Optional[Callable[[dict, Exception], None]] = None,
        warn_collisions: bool = True,
    ):
        if mode not in {"thread", "process"}:
            raise ValueError("mode must be 'thread' or 'process'")

        self.max_workers = max_workers
        self.desc = desc
        self.mode = mode
        self.show_traceback = show_traceback
        self.on_error = on_error
        self.warn_collisions = warn_collisions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_executor(self) -> ThreadPoolExecutor | ProcessPoolExecutor:
        if self.mode == "process":
            return ProcessPoolExecutor(max_workers=self.max_workers)
        return ThreadPoolExecutor(max_workers=self.max_workers)

    def _check_picklable(self, func: Callable) -> None:
        """Raise early if *func* cannot be sent to worker processes."""
        try:
            pickle.dumps(func)
        except Exception as exc:
            raise ValueError(
                f"'{func.__name__}' is not picklable and cannot be used in "
                "process mode. Use a module-level function instead of a "
                "lambda or closure."
            ) from exc

    @staticmethod
    def _iterate(futures, total: int, desc: str):
        """Wrap ``as_completed`` with tqdm when available."""
        completed = as_completed(futures)
        if _TQDM_AVAILABLE:
            return tqdm(completed, total=total, desc=desc)
        return completed

    # ------------------------------------------------------------------
    # Decorator entry point
    # ------------------------------------------------------------------

    def __call__(self, func: Callable) -> Callable:
        if self.mode == "process":
            self._check_picklable(func)

        @wraps(func)
        def wrapper(jobs: Iterable[dict]) -> JobResult:
            # Materialise generators so we can safely call len() and iterate
            # twice (once to submit, once to collect).
            jobs = list(jobs)
            outcome = JobResult()

            # Do NOT use `with executor:` when KeyboardInterrupt handling is
            # needed — the context manager calls shutdown(wait=True) on exit,
            # which would block in __exit__ before our except clause runs,
            # and then shutdown() would be called a second time in the handler.
            # Instead, manage the executor lifetime explicitly so we can call
            # shutdown(wait=False, cancel_futures=True) on interrupt.
            executor = self._make_executor()
            try:
                future_to_job: dict = {
                    executor.submit(func, **job): job
                    for job in jobs
                }

                for future in self._iterate(
                    future_to_job, len(future_to_job), self.desc
                ):
                    job = future_to_job[future]
                    try:
                        result = future.result()

                        if not isinstance(result, dict) or not {"value", "key"} <= result.keys():
                            raise TypeError(
                                f"'{func.__name__}' must return a JobReturn dict "
                                f"with 'value' and 'key' keys, got: {type(result).__name__}."
                            )

                        value, key = result["value"], result["key"]

                        if self.warn_collisions and key in outcome.succeeded:
                            # stacklevel=1 points at the line inside this loop
                            # body, which is the closest meaningful frame
                            # available from within a future-collection loop.
                            # Validate with warnings.filterwarnings("error")
                            # in tests if the exact call site matters.
                            warnings.warn(
                                f"Duplicate key {key!r} from job {job}. "
                                "The earlier result will be overwritten.",
                                UserWarning,
                                stacklevel=1,
                            )

                        outcome.succeeded[key] = value

                    except Exception as exc:
                        failed = FailedJob(job=job, exc=exc, exc_type=type(exc))
                        outcome.failed.append(failed)

                        if self.on_error is not None:
                            try:
                                self.on_error(job, exc)
                            except Exception as cb_exc:
                                warnings.warn(
                                    f"on_error callback raised an exception: {cb_exc}",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )

                        if self.show_traceback:
                            print(
                                "".join(
                                    traceback.format_exception(
                                        type(exc), exc, exc.__traceback__
                                    )
                                )
                            )
                        else:
                            print(f"[FAILED JOB] {job} -> {exc}")

            except KeyboardInterrupt:
                # cancel_futures=True drops pending (not yet running) futures.
                # Already-running futures in ProcessPoolExecutor workers cannot
                # be killed here — they will run to completion or until the OS
                # reclaims the worker processes.
                executor.shutdown(wait=False, cancel_futures=True)
                raise

            else:
                # Normal exit: wait for any stragglers and release resources.
                executor.shutdown(wait=True)

            return outcome

        return wrapper