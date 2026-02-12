# -*- coding: utf-8 -*-
"""
Created on 2026-02-11 11:24:23

@author: Maurice Roots

Description:
     - A module to create uv projects: Templating
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date
from pathlib import Path


DEFAULT_FOLDERS = [
     "src",
     "figures",
     "presentations",
     "notebooks",
     "data",
     "lookups",
]

DEFAULT_README = """# {project_name}
This is the {project_name} project.
This is a standardized atmospheric science data analysis project created with atmoz.
"""

def _new_cell_id() -> str:
     return uuid.uuid4().hex


def _build_notebook(project_name: str, date_str: str) -> dict:
     return {
          "cells": [
               {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": [
                         f"## {project_name}\n",
                         "\n",
                         f"This is the {project_name} project.\n",
                         "\n",
                         "This is a standardized atmospheric science data analysis project created with atmoz.\n",
                         "\n",
                         "Description:\n",
                         "\n",
                         "Author(s):\n",
                         "\n",
                         f"Last updated: {date_str}\n",
                         "\n",
                         "License:\n",
                         "\n",
                         "Acknowledgements:\n",
                         "\n",
                         "References:\n",
                         "\n",
                         "TODO:\n",
                         "\n",
                         "- [ ] Task 1\n",
                         "- [ ] Task 2\n",
                         "- [ ] Task 3\n",
                         "---\n",
                         "\n",
                    ],
                    "id": _new_cell_id(),
               },
               {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": [
                         "---\n",
                         "\n",
                         "### Introduction\n",
                         "\n",
                    ],
                    "id": _new_cell_id(),
               },
               {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": [
                         "---\n",
                         "\n",
                         "### Code\n",
                         "\n",
                    ],
                    "id": _new_cell_id(),
               },
               {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"language": "python"},
                    "outputs": [],
                    "source": [
                         "import atmoz\n",
                         "import numpy as np\n",
                         "import pandas as pd\n",
                         "import matplotlib.pyplot as plt\n",
                    ],
                    "id": _new_cell_id(),
               },
          ],
          "metadata": {
               "language_info": {"name": "python"},
          },
          "nbformat": 4,
          "nbformat_minor": 5,
     }


def _build_project_paths(project_name: str, base_path: str | Path) -> dict[str, Path]:
     base_dir = Path(base_path).expanduser().resolve() / project_name
     return {
          "base": base_dir,
          "readme": base_dir / "README.md",
          "notebooks_dir": base_dir / "notebooks",
          "notebook": base_dir / "notebooks" / f"{project_name}.ipynb",
     }


def _write_text_file(path: Path, content: str) -> None:
     path.parent.mkdir(parents=True, exist_ok=True)
     path.write_text(content, encoding="utf-8")


def create_project(project_name: str, base_path: str | Path) -> Path:
     paths = _build_project_paths(project_name, base_path)
     paths["base"].mkdir(parents=True, exist_ok=True)

     for folder in DEFAULT_FOLDERS:
          (paths["base"] / folder).mkdir(parents=True, exist_ok=True)

     readme_content = DEFAULT_README.format(project_name=project_name)
     _write_text_file(paths["readme"], readme_content)

     notebook_payload = _build_notebook(project_name, date.today().isoformat())
     notebook_content = json.dumps(notebook_payload, indent=4, ensure_ascii=True)
     _write_text_file(paths["notebook"], f"{notebook_content}\n")

     return paths["base"]


def _parse_project_args(argv: list[str] | None = None) -> argparse.Namespace:
     parser = argparse.ArgumentParser(
          prog="atmoz project",
          description="Create a standardized atmoz project.",
     )
     parser.add_argument("name", nargs="?", help="Project name")
     parser.add_argument("path", nargs="?", help="Base path for the project")
     parser.add_argument("-n", "--name", dest="name_flag", help="Project name")
     parser.add_argument("-p", "--path", dest="path_flag", help="Base path for the project")
     return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
     args = _parse_project_args(argv)
     project_name = args.name_flag or args.name
     base_path = args.path_flag or args.path

     if not project_name or not base_path:
          raise SystemExit("Project name and path are required.")

     create_project(project_name, base_path)
     return 0


if __name__ == "__main__":
     raise SystemExit(main(sys.argv[1:]))

