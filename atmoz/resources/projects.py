# -*- coding: utf-8 -*-
"""
Created on 2026-02-11 11:24:23

@author: Maurice Roots

Description:
     - A module to create uv projects: Templating
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_FOLDERS = [
     "src",
     "figures",
     "presentations",
     "notebooks",
     "data",
     "lookups",
]


def create_project(project_path: Path, folders: list[str], use_uv: bool = True) -> None:
     project_path = project_path.resolve()

     if use_uv:
          uv_command = ["uv", "init", str(project_path)]
          try:
               subprocess.run(uv_command, check=True)
          except FileNotFoundError as exc:
               raise RuntimeError("uv is not available on PATH.") from exc
          except subprocess.CalledProcessError as exc:
               raise RuntimeError("uv init failed.") from exc
     else:
          project_path.mkdir(parents=True, exist_ok=True)

     for folder in folders:
          (project_path / folder).mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
     parser = argparse.ArgumentParser(
          description="Create a uv project and standard folders."
     )
     parser.add_argument(
          "name_or_path",
          help="Project name or path to create.",
     )
     parser.add_argument(
          "--no-uv",
          action="store_true",
          help="Skip running uv init and only create folders.",
     )
     parser.add_argument(
          "--folders",
          nargs="*",
          default=DEFAULT_FOLDERS,
          help="Folder names to create under the project.",
     )
     return parser


def main(argv: list[str] | None = None) -> int:
     parser = build_parser()
     args = parser.parse_args(argv)

     try:
          create_project(Path(args.name_or_path), args.folders, use_uv=not args.no_uv)
     except RuntimeError as exc:
          parser.error(str(exc))
          return 2

     return 0


if __name__ == "__main__":
     raise SystemExit(main())
