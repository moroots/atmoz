# -*- coding: utf-8 -*-
"""
Created on 2026-06-06 21:27:40

@author: Maurice Roots

Description:
     - A short Module for handling requests.Sessions() when doing ThreadPoolExecutor or ProcessPoolExecutor 
"""
# %%

import functools
import threading
import weakref

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SessionHandler:
    """
    Thread-safe HTTP session manager with connection pooling and automatic retries.

    Each thread gets its own `requests.Session` (via threading.local), so sessions
    are never shared across threads. All sessions are tracked via a WeakSet so they
    are closed automatically when a thread exits, and can also be closed explicitly
    via `close()` or the context-manager protocol.

    Parameters
    ----------
    pool_connections : int
        Number of connection pools to cache (default 100; requests default is 10).
        Each unique host gets its own pool, so raise this if you talk to many hosts.
    pool_maxsize : int
        Maximum number of connections to keep in each pool (default 100).
        Connections beyond this limit are discarded after use.
    max_retries : int
        Total number of retry attempts per request (default 3).
    backoff_factor : float
        Seconds to wait between retries using exponential back-off:
        ``{backoff_factor} * (2 ** (retry_number - 1))``.
        With the default of 0.5 the waits are 0.5 s, 1 s, 2 s (≈ 3.5 s total).
    timeout : float or tuple or None
        Default (connect, read) timeout in seconds applied to every request.
        Pass a tuple ``(connect_timeout, read_timeout)`` or a single float for
        both. ``None`` disables the default (callers must then set it themselves).

    Usage
    -----
    As a context manager::

        with SessionHandler() as handler:
            session = handler.session()
            session.get("https://example.com")

    As a decorator (plain functions only — see note below)::

        handler = SessionHandler()

        @handler
        def fetch(session, url):
            return session.get(url).json()

        fetch("https://example.com")

    .. note::
        The ``__call__`` decorator injects a ``session`` as the **first positional
        argument** of the wrapped function. It is designed for plain functions, not
        instance methods (where ``self`` occupies that slot). Use ``handler.session()``
        directly inside methods instead.
    """

    def __init__(
        self,
        pool_connections: int = 100,
        pool_maxsize: int = 100,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: float | tuple | None = 10.0,
    ):
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

        self._local = threading.local()
        self._sessions: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Session access
    # ------------------------------------------------------------------

    def session(self) -> requests.Session:
        """
        Return the session for the current thread, creating one if necessary.

        Raises
        ------
        RuntimeError
            If ``close()`` has already been called on this handler.
        """
        if self._closed:
            raise RuntimeError(
                "SessionHandler has been closed and cannot create new sessions."
            )

        if not hasattr(self._local, "session"):
            retry = Retry(
                total=self.max_retries,
                backoff_factor=self.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                # None means retry on all HTTP methods, including POST.
                allowed_methods=None,
            )

            adapter = HTTPAdapter(
                pool_connections=self.pool_connections,
                pool_maxsize=self.pool_maxsize,
                max_retries=retry,
                # Block instead of silently dropping connections when the pool
                # is exhausted, so callers see back-pressure rather than errors.
                pool_block=True,
            )

            sess = requests.Session()
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)

            # Inject a default timeout so requests never hang indefinitely.
            if self.timeout is not None:
                sess.request = functools.partial(  # type: ignore[method-assign]
                    sess.request, timeout=self.timeout
                )

            self._local.session = sess

            with self._lock:
                self._sessions.add(sess)

        return self._local.session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close every tracked session and mark this handler as closed.

        After this call, ``session()`` will raise ``RuntimeError``. Threads that
        still hold a local reference to a session object should stop using it.
        """
        self._closed = True

        with self._lock:
            for sess in list(self._sessions):
                try:
                    sess.close()
                except Exception:
                    pass
            # Do NOT call self._sessions.clear() — WeakSet manages its own
            # entries when referents are garbage-collected, and clear() is not
            # reliable across all Python builds.

    def __enter__(self) -> "SessionHandler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Decorator interface
    # ------------------------------------------------------------------

    def __call__(self, func):
        """
        Decorate a **plain function** so that a thread-local session is injected
        as its first positional argument.

        .. warning::
            Do **not** use this on instance methods. ``session`` will be injected
            into the ``self`` slot. Call ``handler.session()`` directly instead.
        """
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(self.session(), *args, **kwargs)

        return wrapped