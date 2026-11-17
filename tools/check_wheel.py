"""Fail if the built wheel is missing package data or contains dev files. Used by CI."""

import glob
import sys
import zipfile

wheel = sorted(glob.glob("dist/*.whl"))[-1]
names = zipfile.ZipFile(wheel).namelist()
problems = []
if not any(n.endswith("Watermark_TOLNet.png") for n in names):
    problems.append("package data (watermark) missing from wheel")
if not any(n.endswith("py.typed") for n in names):
    problems.append("py.typed missing")
if any(n.startswith("extras/") or n.endswith("h4toh5convert") or "executables" in n for n in names):
    problems.append("HDF converter binary must not be in the wheel (it is pulled from GitHub on request)")
bad = [n for n in names if "legacy" in n or "/dev_" in n or n.endswith(".pyc") or "temp_" in n]
if bad:
    problems.append(f"dev files in wheel: {bad}")
if problems:
    sys.exit("\n".join(problems))
print(f"{wheel}: OK ({len(names)} files)")
