# -*- coding: utf-8 -*-
"""
Created on 2026-02-24 21:09:28

@author: Maurice Roots

Description:
     - A simple module for storing paths in research evironments
"""
#%% 

from pathlib import Path
import sqlite3
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Optional
import socket
import warnings

from importlib.resources import files


SCHEMA = {
    "machines": """
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
""",
    "paths": """
CREATE TABLE IF NOT EXISTS paths (
    id INTEGER PRIMARY KEY,
    machine_id INTEGER NOT NULL,
    path_name TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (machine_id) REFERENCES machines(id),
    UNIQUE (machine_id, path_name)
);
"""
}


@dataclass
class PathEntry:
    name: str
    path: Path
    machine: str
    created_at: str
    id: Optional[int] = None

class PathManager:
    """SQLite-backed manager for named paths per machine.

    - Opens a persistent sqlite3 connection at init (fast for repeated ops).
    - Loads all paths for the provided `machine` into `self.paths` (dict).
    - Raises ValueError if `machine` not found in `machines` table.
    """

    def __init__(self, paths_file: str = None, machine: str = None):
        # determine machine name: prefer explicit, otherwise use OS hostname
        if machine is None:
            self.machine = socket.gethostname()
        else:
            self.machine = machine

        # prefer an explicit Path if provided; otherwise resolve the package resource
        if paths_file:
            self.path_file = Path(paths_file)
        else:
            # convert resource to a filesystem Path string and wrap with Path
            self.path_file = Path(str(files("atmoz.config").joinpath("atmoz.paths")))

        # ensure parent dirs exist
        self.path_file.parent.mkdir(parents=True, exist_ok=True)

        # create DB file and tables if missing
        if not self.path_file.exists():
            self._set_paths_file(self.path_file)

        # open long-lived connection for performance
        self._conn = sqlite3.connect(self.path_file, detect_types=sqlite3.PARSE_DECLTYPES)
        self._conn.row_factory = sqlite3.Row

        # make sure schema exists (safe to run repeatedly)
        with self.connection() as conn:
            for sql in SCHEMA.values():
                conn.executescript(sql)

        # verify machine exists and cache its id (raise if not present)
        self._resolve_machine()

    def _resolve_machine(self) -> None:
        """Lookup `self.machine` in `machines` table and set `self.machine_id`.

        If the machine is not registered, set `self.machine_id = None` and
        allow the manager to be used; the machine will be created when the
        user first adds a path.
        """
        with self.connection() as conn:
            cur = conn.execute("SELECT id FROM machines WHERE name = ?", (self.machine,))
            row = cur.fetchone()
            if not row:
                # machine not registered yet; defer creation until first write
                warnings.warn(
                    f"Machine '{self.machine}' not found in machines table; "
                    "it will be created on first path addition.",
                    UserWarning,
                )
                self.machine_id = None
            else:
                self.machine_id = int(row["id"])

        # load paths into memory as dict[str, PathEntry]
        self.get_paths()

    @contextmanager
    def connection(self):
        """Yield an sqlite3.Connection. Uses the persistent connection when present,
        otherwise creates a short-lived connection.
        """
        if getattr(self, "_conn", None):
            yield self._conn
        else:
            conn = sqlite3.connect(self.path_file)
            try:
                conn.row_factory = sqlite3.Row
                yield conn
            finally:
                conn.close()

    def _set_paths_file(self, path_file: Path):
        # initialize sqlite DB and create tables
        conn = sqlite3.connect(path_file)
        try:
            for sql in SCHEMA.values():
                conn.executescript(sql)
        finally:
            conn.close()

    def get_paths(self) -> Dict[str, PathEntry]:
        """Load and cache all paths for the current machine.

        Returns dict mapping `path_name` -> `PathEntry`.
        """
        # if machine isn't registered yet, return empty cache
        if getattr(self, "machine_id", None) is None:
            self.paths = {}
            return self.paths

        rows = []
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT p.id, p.path_name, p.path, p.created_at, m.name as machine
                FROM paths p
                JOIN machines m ON p.machine_id = m.id
                WHERE m.id = ?
                """,
                (self.machine_id,)
            )
            rows = cur.fetchall()

        self.paths: Dict[str, PathEntry] = {
            r["path_name"]: PathEntry(
                name=r["path_name"],
                path=Path(r["path"]),
                machine=r["machine"],
                created_at=r["created_at"],
                id=int(r["id"]),
            )
            for r in rows
        }
        return self.paths

    def new_path(self, name: str, path: str) -> PathEntry:
        """Insert or update a single path for this machine, then reload cache."""
        with self.connection() as conn:
            # ensure machine exists in machines table; create if missing
            if getattr(self, "machine_id", None) is None:
                conn.execute("INSERT OR IGNORE INTO machines (name) VALUES (?)", (self.machine,))
                cur = conn.execute("SELECT id FROM machines WHERE name = ?", (self.machine,))
                row = cur.fetchone()
                self.machine_id = int(row["id"])

            conn.execute(
                """
                INSERT INTO paths (machine_id, path_name, path)
                VALUES (?, ?, ?)
                ON CONFLICT(machine_id, path_name) DO UPDATE SET
                    path=excluded.path,
                    created_at=CURRENT_TIMESTAMP
                """,
                (self.machine_id, name, str(path)),
            )
            conn.commit()

        self.get_paths()
        return self.paths[name]

    def new_paths(self, paths: Dict[str, str]) -> Dict[str, PathEntry]:
        """Bulk insert/update multiple paths efficiently."""
        with self.connection() as conn:
            cur = conn.cursor()
            # ensure machine exists
            if getattr(self, "machine_id", None) is None:
                conn.execute("INSERT OR IGNORE INTO machines (name) VALUES (?)", (self.machine,))
                cur2 = conn.execute("SELECT id FROM machines WHERE name = ?", (self.machine,))
                row = cur2.fetchone()
                self.machine_id = int(row["id"])

            for name, p in paths.items():
                cur.execute(
                    """
                    INSERT INTO paths (machine_id, path_name, path)
                    VALUES (?, ?, ?)
                    ON CONFLICT(machine_id, path_name) DO UPDATE SET
                        path=excluded.path,
                        created_at=CURRENT_TIMESTAMP
                    """,
                    (self.machine_id, name, str(p)),
                )
            conn.commit()

        return self.get_paths()

    def get_path(self, name: str) -> Optional[PathEntry]:
        return self.paths.get(name)

    def remove_path(self, name: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM paths WHERE machine_id = ? AND path_name = ?",
                (self.machine_id, name),
            )
            conn.commit()

        self.get_paths()

    def clear_paths(self) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM paths WHERE machine_id = ?", (self.machine_id,))
            conn.commit()
        self.get_paths()

    def close(self) -> None:
        if getattr(self, "_conn", None):
            self._conn.close()
            self._conn = None
