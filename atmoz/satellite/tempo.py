"""NASA TEMPO gridded (L3) products at points and boxes, streamed from the NASA archive (no full download needed).

Follows the package conventions (``docs/conventions.md``): ``download()`` returns an ``AtmozDataset``, ``fetch()`` returns
``{name: Path}`` (whole granules, cached, resumable), downloads are all-or-nothing. ``TEMPO.read()`` parses local granules.

>>> tempo = TEMPO("NO2_L3")                                    # needs an Earthdata login, see docs/satellite.md
>>> ds = tempo.download("2025-07-28", points={"GSFC": (38.98, -76.83)})
>>> ds["GSFC/vertical_column_troposphere"]                     # AtmozFrame: index t (UTC), value, lat, lon, quality_flag
>>> o3 = TEMPO("O3PROF_L3").download("2025-07-28", points={"GSFC": (38.98, -76.83)})
>>> o3["GSFC/ozone_profile"]                                   # long: index t, alt (km asl), value (ppb), partial_column (DU)
>>> integrate_layers(o3, 0, 2)                                 # 0-2 km partial column (DU) and mean mixing ratio (ppb)

Only the standard library, numpy and pandas are needed to import this module. ``h5py`` (reading) and ``earthaccess``
(search, streaming, downloads) are imported when first used: ``pip install "atmoz[satellite]"``.
"""

from __future__ import annotations

import gc
import re
import time
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses import AtmozDataset, AtmozFrame
from atmoz.exceptions import DownloadError, NoDataError
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.utilities import PathLike, run_jobs

# --------------------------------------------------------------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------------------------------------------------------------- #
# The grid is regular lat/lon with 1-D "latitude" / "longitude" axes and one scan per file, so every variable is shaped
# (1, lat, lon), profile variables (1, lat, lon, layer).
_PRODUCTS: Dict[str, dict] = {
    "NO2_L3": {
        "short_name": "TEMPO_NO2_L3",
        "profile": False,
        "default_variables": ["vertical_column_troposphere", "vertical_column_troposphere_uncertainty"],
        "description": "NO2 vertical columns (tropospheric, total, stratospheric) and their uncertainties",
    },
    "O3PROF_L3": {
        "short_name": "TEMPO_O3PROF_L3",
        "profile": True,
        "default_variables": ["ozone_profile"],
        "description": "Retrieved ozone profile: 24 layers, partial columns in DU (BETA product)",
    },
}
_BY_SHORT_NAME = {p["short_name"]: k for k, p in _PRODUCTS.items()}

_VARIABLE_GROUPS = ("product", "support_data")
_QUALITY_VARIABLE = "product/main_data_quality_flag"
_TOTAL_COLUMN = "product/total_ozone_column"
_PRESSURE_BOUNDS = "support_data/ozone_profile_pressure_bounds"
_ALTITUDE_BOUNDS = "support_data/ozone_profile_altitude_bounds"
_ALTITUDE = "support_data/ozone_profile_altitude"
_CLOUD = "support_data/eff_cloud_fraction"
_GRANULE_TIME = re.compile(r"_(\d{8}T\d{6}Z)_")
_PRODUCT_KEY = re.compile(r"^(TEMPO_.+?)_(V\d+)_")

#: What the altitude of a TEMPO profile is measured from. The layer bounds of ``O3PROF_L3`` are altitudes above sea level and
#: the lower bound of the bottom layer is the terrain height (it is kept as ``site_elevation_km`` for point frames).
ALT_REFERENCE = "asl"

_DU_MOLEC_CM2 = 2.6867e16
_G = 9.80665          # m s-2
_M_AIR = 0.0289647    # kg mol-1
_N_A = 6.02214e23     # mol-1

_RETRY_DELAY = 5.0    # seconds; the pause before retry n is n * _RETRY_DELAY (reads) or _RETRY_DELAY * 2**n (downloads)
_BATCH_FACTOR = 6     # process mode: granules per worker per batch before the worker pool is recreated
_FILL_LIMIT = 1e29    # |value| beyond this is a fill value even when the file does not declare one

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]   # (west, south, east, north)


# --------------------------------------------------------------------------------------------------------------------------------- #
# Ozone profile arithmetic
# --------------------------------------------------------------------------------------------------------------------------------- #
def du_to_ppbv(du, dp_hpa):
    """Ozone partial column (DU) in a layer -> volume mixing ratio (ppb), from the layer's pressure thickness (hPa).

    Hydrostatic balance: the air column of the layer is ``dp / (g * M_air)``. Accepts scalars or arrays.
    """
    air_molec_cm2 = np.asarray(dp_hpa, dtype=float) * 100.0 / (_G * _M_AIR) * _N_A * 1e-4
    return np.asarray(du, dtype=float) * _DU_MOLEC_CM2 / air_molec_cm2 * 1e9


def _integrate_frame(af: AtmozFrame, z_min: float, z_max: float, reference: str) -> AtmozFrame:
    need = ["partial_column", "pressure_thickness", "alt_lower", "alt_upper"]
    missing = [c for c in need if c not in af.df.columns]
    if missing:
        raise KeyError(f"integrate_layers needs a TEMPO ozone profile frame with the columns {need}; missing {missing}.")
    if "error" in str(af.metadata.get("variable", "")):
        raise ValueError("integrate_layers sums partial columns; the errors of a profile do not add linearly. "
                         "Integrate 'ozone_profile' instead.")
    d = af.df.reset_index()
    tname = d.columns[0]
    keys = [tname, "lat", "lon"]
    lo, hi = d["alt_lower"].to_numpy(float), d["alt_upper"].to_numpy(float)
    if reference == "agl":      # the bottom of the lowest layer of each pixel and scan is the ground
        ground = d.groupby(keys)["alt_lower"].transform("min").to_numpy(float)
        lo, hi = lo - ground, hi - ground
    thick = np.where(hi - lo > 0, hi - lo, np.nan)
    frac = np.clip(np.minimum(hi, z_max) - np.maximum(lo, z_min), 0, None) / thick
    d["_du"] = d["partial_column"].to_numpy(float) * frac
    d["_dp"] = d["pressure_thickness"].to_numpy(float) * frac
    g = d.groupby(keys, sort=True)[["_du", "_dp"]].sum(min_count=1).reset_index()
    out = pd.DataFrame({"lat": g["lat"], "lon": g["lon"], "column": g["_du"],
                        "value": du_to_ppbv(g["_du"], g["_dp"].where(g["_dp"] > 0))})
    out.index = pd.DatetimeIndex(g[tname], name="time")
    meta = {k: v for k, v in af.metadata.items() if k not in ("alt_reference", "site_elevation_km")}
    meta.update({"layer_km": (float(z_min), float(z_max)), "layer_reference": reference,
                 "layer_column": "partial_column"})
    return AtmozFrame(out, units={"column": "DU", "value": "ppb", "lat": "deg", "lon": "deg"}, lat="lat", lon="lon",
                      metadata=meta)


def integrate_layers(data: Union[AtmozFrame, AtmozDataset], z_min: float = 0.0, z_max: float = 2.0, *,
                     reference: str = "asl") -> Union[AtmozFrame, AtmozDataset]:
    """Partial ozone column between two altitudes, from a TEMPO ``O3PROF_L3`` profile frame (or dataset of them).

    Each layer contributes in proportion to its vertical overlap with ``[z_min, z_max]`` (a layer is assumed uniform).
    Returns, per scan and pixel, ``column`` (DU) and ``value``, the layer-mean mixing ratio (ppb: the column divided by the
    hydrostatic air column of the same slice). The result has no ``alt`` axis: it is one number per time, like a surface
    series; ``metadata["layer_km"]`` and ``["layer_reference"]`` say which slab it is.

    Parameters
    ----------
    z_min, z_max:
        Limits in km.
    reference:
        ``"asl"`` (default, TEMPO's own reference) or ``"agl"``, which measures the limits from each pixel's ground (the
        lower bound of its bottom layer).

    TEMPO's layers are 2-3 km thick in the lower troposphere, so a 0-2 km value is essentially the lowest layer and carries
    little independent vertical information.
    """
    if reference not in ("asl", "agl"):
        raise ValueError("reference must be 'asl' or 'agl'.")
    if not z_min < z_max:
        raise ValueError("z_min must be below z_max.")
    if isinstance(data, AtmozFrame):
        return _integrate_frame(data, z_min, z_max, reference)
    if isinstance(data, AtmozDataset):
        frames = {k: _integrate_frame(f, z_min, z_max, reference) for k, f in data.items()}
        return AtmozDataset(name=data.name, source=data.source, frames=frames,
                            metadata={**data.metadata, "layer_km": (float(z_min), float(z_max)),
                                      "layer_reference": reference})
    raise TypeError(f"expected an AtmozFrame or AtmozDataset, got {type(data).__name__}.")


# --------------------------------------------------------------------------------------------------------------------------------- #
# Reading one granule (called by the executor; module level so process workers can pickle it)
# --------------------------------------------------------------------------------------------------------------------------------- #
_logged_in = False
_UNITS_TO_KM = {"km": 1.0, "m": 1e-3, "meter": 1e-3, "meters": 1e-3}


def _earthdata():
    """Log in to Earthdata (``~/.netrc``) and return the ``EarthData`` wrapper. Needs the ``satellite`` extra."""
    from atmoz.data_access.NASA import EarthData

    return EarthData()


def _ensure_login() -> None:
    """Log in once per process (worker processes do not inherit a usable session)."""
    global _logged_in
    if not _logged_in:
        _earthdata()
        _logged_in = True


def _open(source):
    """A local path -> the path; a remote granule -> a file-like object streamed with ``earthaccess.open``."""
    if isinstance(source, (str, Path)):
        return source
    import earthaccess

    _ensure_login()
    return earthaccess.open([source])[0]


def _link(granule) -> str:
    links = list(granule.data_links())
    nc = [u for u in links if u.lower().endswith((".nc", ".nc4", ".h5", ".he5"))]
    return (nc or links)[0]


def _granule_name(source) -> str:
    return Path(source).name if isinstance(source, (str, Path)) else _link(source).rsplit("/", 1)[-1]


def _text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.size:
        value = value.reshape(-1)[0]
    return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, np.bytes_)) else str(value)


def _units_of(dset, default: str = "") -> str:
    return _text(dset.attrs["units"]).strip() if "units" in dset.attrs else default


def _read(dset, key=()) -> np.ndarray:
    """A window of a dataset as floats, fill values (the file's own, or any |value| > 1e29) replaced by NaN."""
    arr = np.asarray(dset[key], dtype=float)
    for name in ("_FillValue", "VAR_FILL_VALUE", "missing_value"):
        if name in dset.attrs:
            arr = np.where(arr == np.asarray(dset.attrs[name]).reshape(-1)[0], np.nan, arr)
            break
    return np.where(np.abs(arr) > _FILL_LIMIT, np.nan, arr)


_TIME_UNITS = {"second": "s", "seconds": "s", "minute": "min", "minutes": "min", "hour": "h", "hours": "h",
               "day": "D", "days": "D"}


def _scan_time(dset) -> pd.Timestamp:
    """The scan time of a granule: its ``time`` variable (CF ``<unit> since <reference>``) as a UTC timestamp."""
    units = _units_of(dset)
    match = re.match(r"\s*(\w+)\s+since\s+(\S+)", units, re.I)
    if not match or match.group(1).lower() not in _TIME_UNITS:
        raise ValueError(f"Cannot read the granule time: units {units!r} are not '<seconds|minutes|hours|days> since <date>'.")
    ref = pd.Timestamp(match.group(2))
    ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref.tz_convert("UTC")
    value = float(np.asarray(dset[()]).reshape(-1)[0])
    return ref + pd.to_timedelta(value, unit=_TIME_UNITS[match.group(1).lower()])


def _find(f, name: str) -> str:
    for group in _VARIABLE_GROUPS:
        path = f"{group}/{name}"
        if path in f:
            return path
    available = [f"{g}/{k}" for g in _VARIABLE_GROUPS if g in f for k in f[g].keys()]
    raise KeyError(f"Variable {name!r} not found in the granule. Available: {available}")


def _nearest(axis: np.ndarray, value: float, tol: float) -> Optional[int]:
    i = int(np.argmin(np.abs(axis - value)))
    return i if abs(axis[i] - value) <= tol else None


def _spacing(axis: np.ndarray) -> float:
    return float(np.median(np.abs(np.diff(axis)))) if axis.size > 1 else 0.0


def _windows(lat: np.ndarray, lon: np.ndarray, spec: dict):
    """``(region, lat window, lon window)`` for every requested point and the box; regions outside the grid go to ``skipped``."""
    out, skipped = [], []
    d_lat, d_lon = _spacing(lat), _spacing(lon)
    for region, (plat, plon) in spec["points"].items():
        i, j = _nearest(lat, plat, d_lat), _nearest(lon, plon, d_lon)
        if i is None or j is None:
            skipped.append(region)
        else:
            out.append((region, slice(i, i + 1), slice(j, j + 1), (plat, plon)))
    if spec["bbox"] is not None:
        west, south, east, north = spec["bbox"]
        ii = np.where((lat >= south) & (lat <= north))[0]
        jj = np.where((lon >= west) & (lon <= east))[0]
        if ii.size and jj.size:
            out.append(("bbox", slice(int(ii.min()), int(ii.max()) + 1), slice(int(jj.min()), int(jj.max()) + 1), None))
        else:
            skipped.append("bbox")
    return out, skipped


def _grid(lat: np.ndarray, lon: np.ndarray, wi: slice, wj: slice):
    la, lo = np.meshgrid(lat[wi], lon[wj], indexing="ij")
    return la, lo


def _column_rows(f, region, wi, wj, requested, ctx) -> Optional[pd.DataFrame]:
    lat, lon, variables, max_quality, paths = ctx["lat"], ctx["lon"], ctx["variables"], ctx["max_quality"], ctx["paths"]
    win = (0, wi, wj)
    flag = _read(f[_QUALITY_VARIABLE], win)
    values = {v: _read(f[paths[v]], win) for v in variables}
    if max_quality is not None:
        bad = ~(flag <= max_quality)       # also masks NaN flags
        values = {k: np.where(bad, np.nan, v) for k, v in values.items()}
    la, lo = _grid(lat, lon, wi, wj)
    frame = pd.DataFrame({"time": ctx["scan"], "region": region, "lat": la.ravel(), "lon": lo.ravel(),
                          **{v: a.ravel() for v, a in values.items()}, "quality_flag": flag.ravel()})
    if requested is not None:
        frame["lat_requested"], frame["lon_requested"] = requested
        return frame                          # a point keeps its row even when masked: coverage stays visible
    frame = frame.dropna(subset=variables, how="all")
    return frame if len(frame) else None


def _profile_rows(f, region, wi, wj, requested, ctx) -> Optional[pd.DataFrame]:
    lat, lon, variables, paths = ctx["lat"], ctx["lon"], ctx["variables"], ctx["paths"]
    total = _read(f[_TOTAL_COLUMN], (0, wi, wj))
    if not np.isfinite(total).any():
        return None       # this scan did not observe the pixel(s): one cheap read before any 24-layer array is pulled
    n_layer = f[paths[variables[0]]].shape[-1]
    ks = slice(n_layer - 1, n_layer) if ctx["lowest_layer"] else slice(0, n_layer)
    w4, w5 = (0, wi, wj, ks), (0, wi, wj, ks, slice(None))
    pb = _read(f[_PRESSURE_BOUNDS], w5)
    ab = _read(f[_ALTITUDE_BOUNDS], w5)
    ab = ab * _UNITS_TO_KM.get(_units_of(f[_ALTITUDE_BOUNDS], "km").lower(), 1.0)
    pb = pb * (0.01 if _units_of(f[_PRESSURE_BOUNDS], "hPa") == "Pa" else 1.0)
    alt = _read(f[_ALTITUDE], w4) * _UNITS_TO_KM.get(_units_of(f[_ALTITUDE], "km").lower(), 1.0)
    cols = {v: _read(f[paths[v]], w4) for v in variables}
    ni, nj, nl = alt.shape
    with warnings.catch_warnings():        # pixels without bounds are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        upper, lower = np.nanmax(ab, axis=-1), np.nanmin(ab, axis=-1)
    la, lo = _grid(lat, lon, wi, wj)
    frame = pd.DataFrame({
        "time": ctx["scan"], "region": region,
        "lat": np.repeat(la.ravel(), nl), "lon": np.repeat(lo.ravel(), nl),
        "layer": np.tile(np.arange(n_layer)[ks], ni * nj),
        "alt": alt.ravel(), "alt_lower": lower.ravel(), "alt_upper": upper.ravel(),
        "pressure_thickness": np.abs(pb[..., 0] - pb[..., 1]).ravel(),
        **{v: a.ravel() for v, a in cols.items()},
        "total_ozone_column": np.repeat(total.ravel(), nl),
    })
    if _CLOUD in f:
        frame["eff_cloud_fraction"] = np.repeat(_read(f[_CLOUD], (0, wi, wj)).ravel(), nl)
    if requested is not None:
        frame["lat_requested"], frame["lon_requested"] = requested
    frame = frame[np.isfinite(frame["total_ozone_column"]) & frame[variables].notna().any(axis=1)]
    return frame if len(frame) else None


def _subset(f, spec: dict) -> dict:
    """Subset one open granule: the pixels of the requested points / box as long tables, plus units and skipped regions."""
    lat, lon = _read(f["latitude"]), _read(f["longitude"])
    variables = spec["variables"]
    paths = {v: _find(f, v) for v in variables}
    ctx = {"lat": lat, "lon": lon, "variables": variables, "paths": paths, "max_quality": spec["max_quality"],
           "lowest_layer": spec["lowest_layer"], "scan": _scan_time(f["time"])}
    windows, skipped = _windows(lat, lon, spec)
    rows = []
    for region, wi, wj, requested in windows:
        make = _profile_rows if spec["profile"] else _column_rows
        part = make(f, region, wi, wj, requested, ctx)
        if part is not None:
            rows.append(part)
    return {"scan": ctx["scan"], "rows": pd.concat(rows, ignore_index=True) if rows else None,
            "units": {v: _units_of(f[paths[v]]) for v in variables}, "skipped": skipped}


def _read_once(source, spec: dict) -> dict:
    import h5py

    handle = _open(source)
    try:
        with h5py.File(handle, "r", rdcc_nbytes=64 * 1024 ** 2) as f:   # tiles are ~10 MB: a big cache avoids re-reading them
            return _subset(f, spec)
    finally:
        if hasattr(handle, "close"):
            handle.close()
        gc.collect()


def _read_granule(source, spec: dict, retries: int = 3) -> dict:
    """Read one granule; remote reads retry dropped connections (reset / timeout / failed open) with a growing pause."""
    if isinstance(source, (str, Path)):
        retries = 1
    for attempt in range(1, retries + 1):
        try:
            return _read_once(source, spec)
        except OSError:
            if attempt == retries:
                raise
            time.sleep(_RETRY_DELAY * attempt)


def _read_keyed(name: str, source, spec: dict, retries: int = 3) -> dict:
    """The form ``ParallelExecutor`` wants (process mode)."""
    return {"key": name, "value": _read_granule(source, spec, retries)}


def _run_reads(jobs: List[dict], *, max_workers: int, mode: str, silent: bool, show_traceback: bool) -> Dict[str, dict]:
    """Granule reads on a thread or process pool. All-or-nothing: any failure raises :class:`DownloadError`."""
    if mode not in ("thread", "process"):
        raise ValueError("mode must be 'thread' or 'process'.")
    desc = "Reading TEMPO granules"
    if mode == "thread":
        return run_jobs(lambda job: _read_granule(job["source"], job["spec"], job["retries"]), jobs,
                        key=lambda j: j["name"], describe=lambda j: j["name"], noun="granule read", desc=desc,
                        max_workers=max_workers, silent=silent, show_traceback=show_traceback)
    # h5py holds a global lock during remote reads, so threads do not overlap network waits; processes do.
    from atmoz.resources.parallelExecutor import ParallelExecutor

    executor = ParallelExecutor(max_workers=max_workers, desc=desc, mode="process", show_traceback=show_traceback,
                                progress=not silent, print_failures=False)
    read = executor(_read_keyed)
    succeeded: Dict[str, dict] = {}
    failed = []
    batch = max(max_workers, 1) * _BATCH_FACTOR    # a new pool per batch recycles the workers and bounds their memory
    for k in range(0, len(jobs), batch):
        part = read(jobs[k:k + batch])
        succeeded.update(part.succeeded)
        failed.extend(part.failed)
    if failed:
        lines = [f"  {f.job['name']}: {f.exc}" for f in failed]
        raise DownloadError(f"{len(failed)} of {len(jobs)} granule read(s) failed (nothing returned):\n" + "\n".join(lines),
                            failed=[(f.job, f.exc) for f in failed])
    return {j["name"]: succeeded[j["name"]] for j in jobs}


# --------------------------------------------------------------------------------------------------------------------------------- #
# Whole-granule download (resumable)
# --------------------------------------------------------------------------------------------------------------------------------- #
def _session():
    """An authenticated ``requests`` session for the NASA archive (Earthdata login)."""
    import earthaccess

    _ensure_login()
    return earthaccess.get_requests_https_session()


def _download_file(url: str, dest: Path, retries: int = 6) -> Path:
    """Download *url* to *dest* through ``<dest>.part``, resuming with an HTTP ``Range`` request after a dropped connection.

    A finished file is verified by size and then renamed, so an interrupted download never looks complete.
    """
    import requests

    session = _session()
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(url, headers=headers, stream=True, timeout=(15, 90)) as r:
                if r.status_code == 416:                        # nothing left to fetch
                    total = have
                else:
                    r.raise_for_status()
                    if r.status_code == 206:
                        total = int(r.headers["Content-Range"].split("/")[-1])
                        mode = "ab"
                    else:                                       # the server ignored Range: start over
                        total = int(r.headers["Content-Length"])
                        mode = "wb"
                    with open(part, mode) as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
            if part.stat().st_size == total:
                part.replace(dest)
                return dest
            last = OSError(f"size {part.stat().st_size} != expected {total}")
            if part.stat().st_size > total:                     # cannot be resumed
                part.unlink()
        except (requests.RequestException, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(min(60.0, _RETRY_DELAY * 2 ** attempt))
    raise OSError(f"download failed after {retries} attempts: {url}: {last}")


# --------------------------------------------------------------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------------------------------------------------------------- #
def _normalise_points(points) -> Dict[str, Point]:
    if points is None:
        return {}
    if isinstance(points, dict):
        items = {str(k): v for k, v in points.items()}
    else:
        pts = list(points)
        if len(pts) == 2 and all(np.isscalar(x) for x in pts):
            pts = [tuple(pts)]
        items = {}
        for p in pts:
            try:
                items[f"{float(p[0]):.3f},{float(p[1]):.3f}"] = p
            except (TypeError, ValueError, IndexError):
                raise ValueError(f"a point must be (lat, lon), got {p!r}.") from None
    out = {}
    for name, p in items.items():
        try:
            lat, lon = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"point {name!r} must be (lat, lon), got {p!r}.") from None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"point {name!r} = ({lat}, {lon}) is outside lat -90..90 / lon -180..180.")
        out[name] = (lat, lon)
    if "bbox" in out:
        raise ValueError("The point name 'bbox' is reserved for the box.")
    return out


def _check_bbox(bbox) -> Optional[BBox]:
    if bbox is None:
        return None
    try:
        west, south, east, north = (float(x) for x in bbox)
    except (TypeError, ValueError):
        raise ValueError("bbox must be (west, south, east, north) in degrees.") from None
    if not (west < east and south < north):
        raise ValueError(f"bbox {tuple(bbox)} must have west < east and south < north.")
    if not (-90 <= south and north <= 90 and -180 <= west and east <= 180):
        raise ValueError(f"bbox {tuple(bbox)} is outside lat -90..90 / lon -180..180.")
    return (west, south, east, north)


def _day(value, name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} {value!r} is not a date (use YYYY-MM-DD).") from None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


# --------------------------------------------------------------------------------------------------------------------------------- #
# TEMPO
# --------------------------------------------------------------------------------------------------------------------------------- #
class TEMPO:
    """NASA TEMPO L3 client for one product and version.

    * :meth:`download` -> :class:`~atmoz.dataclasses.AtmozDataset` (streamed and subset: no granule is downloaded)
    * :meth:`fetch`    -> ``{granule file name: Path}`` (whole granules in the cache, resumable)
    * :meth:`read`     -> :class:`~atmoz.dataclasses.AtmozDataset` from granules already on disk

    Products (``TEMPO.products``): ``"NO2_L3"`` (columns) and ``"O3PROF_L3"`` (ozone profiles, 24 layers). Searches are pinned to one
    product *version*: a short name searched without a version returns every version together (V03 and V04), which double counts.

    A dataset holds one frame per region and variable, keyed ``"{region}/{variable}"`` (``region`` is the name you gave the point,
    ``"lat,lon"`` when you gave none, or ``"bbox"``). Frames have the UTC scan time as index (a scan lasts ~10 minutes; its timestamp
    is the scan time stored in the file) and the columns ``value``, ``lat`` and ``lon`` (the pixel centre). Column products add
    ``quality_flag`` (0 normal, 1 suspicious, 2 bad). Profile products are long (one row per time and layer) and add ``alt`` (km
    above sea level, the layer middle), ``alt_lower`` / ``alt_upper``, ``layer`` (0 = top), ``pressure_thickness`` (hPa),
    ``partial_column`` (DU), ``total_ozone_column`` (DU) and ``eff_cloud_fraction``; their ``value`` is ozone in **ppb**.
    ``metadata["alt_reference"]`` is ``"asl"``; point frames also carry ``site_elevation_km`` (the ground, from the layer bounds).
    """

    products = _PRODUCTS
    alt_reference = ALT_REFERENCE

    def __init__(self, product: str = "NO2_L3", version: str = "V04", *, earthdata=None) -> None:
        if product not in _PRODUCTS:
            raise ValueError(f"Unsupported product {product!r}. Options: {list(_PRODUCTS)}")
        if not re.fullmatch(r"V\d+", str(version)):
            raise ValueError(f"version must look like 'V04', got {version!r}.")
        self.product = product
        self.version = str(version)
        self.short_name = _PRODUCTS[product]["short_name"]
        self.profile = bool(_PRODUCTS[product]["profile"])
        self._earthdata_client = earthdata       # created (and logged in) on first use; pass a stand-in to test offline

    # ---- search ----------------------------------------------------------------------------------------------------------
    @property
    def earthdata(self):
        if self._earthdata_client is None:
            self._earthdata_client = _earthdata()
        return self._earthdata_client

    @staticmethod
    def _window(start_date, end_date) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start = _day(start_date, "start_date")
        end = _day(end_date if end_date is not None else start_date, "end_date")
        if end < start:
            raise ValueError("end_date is before start_date.")
        return start, end + pd.Timedelta(days=1)    # [start, end + 1 day) in UTC

    @staticmethod
    def _scan_start(granule) -> Optional[pd.Timestamp]:
        match = _GRANULE_TIME.search(_link(granule).rsplit("/", 1)[-1])
        return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%SZ") if match else None

    def search(self, start_date, end_date=None) -> list:
        """Granules whose scan starts within ``[start_date, end_date + 1 day)`` UTC, oldest first.

        CMR matches on overlap, so a scan that began at 23:58 the previous evening is returned by the service; it is dropped
        here because its scan start is not on the requested day.
        """
        start, stop = self._window(start_date, end_date)
        found = self.earthdata.search_data(
            short_name=self.short_name, version=self.version,
            temporal=(start.strftime("%Y-%m-%dT%H:%M:%SZ"), stop.strftime("%Y-%m-%dT%H:%M:%SZ")))
        kept = []
        for granule in found:
            t = self._scan_start(granule)
            if t is not None and start <= t < stop:
                kept.append((t, granule))
        return [g for _, g in sorted(kept, key=lambda x: x[0])]

    def _search_or_raise(self, start_date, end_date, only_scans=None) -> list:
        granules = self.search(start_date, end_date)
        if only_scans is not None and granules:
            wanted = np.sort(_naive_utc(only_scans).values.astype("datetime64[ns]"))
            if not len(wanted):
                granules = []
            else:
                starts = np.array([self._scan_start(g) for g in granules], dtype="datetime64[ns]")
                pos = np.searchsorted(wanted, starts, side="left")
                near = np.minimum(pos, len(wanted) - 1)
                keep = (pos < len(wanted)) & ((wanted[near] - starts) <= np.timedelta64(120, "s"))
                granules = [g for g, k in zip(granules, keep) if k]
        if not granules:
            span = f"{start_date}" + (f" to {end_date}" if end_date else "")
            extra = " matching only_scans" if only_scans is not None else ""
            raise NoDataError(f"No {self.short_name} {self.version} granules found for {span}{extra}.")
        return granules

    # ---- process: validate -> read -> assemble -------------------------------------------------------------------------------
    def _validate(self, points, bbox, variables, max_quality, lowest_layer) -> dict:
        """Check the request and return the spec every granule read uses."""
        pts, box = _normalise_points(points), _check_bbox(bbox)
        if not pts and box is None:
            raise ValueError("Provide points and/or bbox.")
        if lowest_layer and not self.profile:
            raise ValueError(f"lowest_layer applies to profile products only ({self.product} is a column product).")
        if max_quality is not None and (not isinstance(max_quality, (int, np.integer)) or max_quality < 0):
            raise ValueError("max_quality must be a non-negative integer or None.")
        names = [variables] if isinstance(variables, str) else list(variables or _PRODUCTS[self.product]["default_variables"])
        return {"points": pts, "bbox": box, "variables": names, "max_quality": None if self.profile else max_quality,
                "profile": self.profile, "lowest_layer": bool(lowest_layer)}

    def _assemble(self, results: Dict[str, dict], spec: dict, window: Tuple[Any, Any]) -> AtmozDataset:
        """All granules -> the dataset: one frame per region and variable."""
        skipped = sorted({s for r in results.values() for s in r["skipped"]})
        tables = [r["rows"] for r in results.values() if r["rows"] is not None]
        if skipped:
            warnings.warn(f"Outside the TEMPO grid, no data returned for: {skipped}", UserWarning, stacklevel=3)
        if not tables:
            raise NoDataError(
                f"TEMPO {self.product} {self.version}: {len(results)} granule(s) were read but none holds a pixel for the "
                f"request" + (f" ({skipped} lie outside the grid)" if skipped else "") + ". Each scan covers only a strip of "
                "the grid, so a point is not observed in every granule; widen the dates or check the location.")
        data = pd.concat(tables, ignore_index=True).sort_values(
            ["time", "region", "lat", "lon"] + (["layer"] if self.profile else [])).reset_index(drop=True)
        units = next((r["units"] for r in results.values()), {})
        frames: Dict[str, AtmozFrame] = {}
        for region, rows in data.groupby("region", sort=False):
            for v in spec["variables"]:
                frames[f"{region}/{v}"] = self._frame(rows, region, v, units.get(v, ""), spec)
        meta = {"endpoint": "tempo", "product": self.product, "short_name": self.short_name, "version": self.version,
                "start": str(window[0]), "end": str(window[1]), "variables": list(spec["variables"]),
                "points": dict(spec["points"]), "bbox": spec["bbox"], "max_quality": spec["max_quality"],
                "lowest_layer": spec["lowest_layer"], "units": {v: units.get(v, "") for v in spec["variables"]},
                "granules": sorted(results), "skipped": skipped, "rows": {k: len(f) for k, f in frames.items()},
                **({"alt_reference": ALT_REFERENCE} if self.profile else {})}
        return AtmozDataset(name=f"TEMPO {self.product} {self.version}", source="TEMPO", frames=frames, metadata=meta)

    def _frame(self, rows: pd.DataFrame, region: str, variable: str, unit: str, spec: dict) -> AtmozFrame:
        base = {"product": self.product, "short_name": self.short_name, "version": self.version, "region": region,
                "variable": variable, "file_units": unit}
        if region in spec["points"]:
            base["lat_requested"], base["lon_requested"] = spec["points"][region]
        else:
            base["bbox"] = spec["bbox"]
        t = pd.DatetimeIndex(rows["time"], name="time")
        if not self.profile:
            df = pd.DataFrame({"value": rows[variable].to_numpy(), "lat": rows["lat"].to_numpy(),
                               "lon": rows["lon"].to_numpy(),
                               "quality_flag": pd.array(rows["quality_flag"].astype("Int8"))}, index=t)
            units = {"value": unit} if unit else {}
            return AtmozFrame(df, units=units, lat="lat", lon="lon", metadata=base)

        raw = rows[variable].to_numpy(float)
        in_du = unit == "DU"
        value = du_to_ppbv(raw, rows["pressure_thickness"].to_numpy(float)) if in_du else raw
        cols = {"value": value, "lat": rows["lat"].to_numpy(), "lon": rows["lon"].to_numpy(), "alt": rows["alt"].to_numpy(),
                "alt_lower": rows["alt_lower"].to_numpy(), "alt_upper": rows["alt_upper"].to_numpy(),
                "layer": rows["layer"].to_numpy(), "pressure_thickness": rows["pressure_thickness"].to_numpy()}
        if in_du:
            cols["partial_column"] = raw
        cols["total_ozone_column"] = rows["total_ozone_column"].to_numpy()
        if "eff_cloud_fraction" in rows:
            cols["eff_cloud_fraction"] = rows["eff_cloud_fraction"].to_numpy()
        df = pd.DataFrame(cols, index=t)
        units = {"value": "ppb" if in_du else unit, "alt": "km", "alt_lower": "km", "alt_upper": "km",
                 "pressure_thickness": "hPa", "total_ozone_column": "DU", **({"partial_column": "DU"} if in_du else {})}
        meta = {**base, "alt_reference": ALT_REFERENCE, "lowest_layer": spec["lowest_layer"]}
        if region in spec["points"]:
            ground = df["alt_lower"].groupby([df.index, df["lat"], df["lon"]]).min().dropna()
            if len(ground):
                meta["site_elevation_km"] = float(ground.median())
        return AtmozFrame(df, units={c: u for c, u in units.items() if u}, lat="lat", lon="lon", alt="alt", metadata=meta)

    # ---- public ------------------------------------------------------------------------------------------------------------
    def download(self,
                 start_date: Union[str, date, datetime],
                 end_date: Optional[Union[str, date, datetime]] = None,
                 *,
                 points: Optional[Union[Dict[str, Point], Sequence[Point], Point]] = None,
                 bbox: Optional[BBox] = None,
                 variables: Optional[Union[str, Sequence[str]]] = None,
                 max_quality: Optional[int] = 0,
                 lowest_layer: bool = False,
                 only_scans: Optional[Iterable] = None,
                 max_workers: int = 4,
                 mode: str = "thread",
                 retries: int = 3,
                 silent: bool = False,
                 show_traceback: bool = False) -> AtmozDataset:
        """Stream and subset TEMPO granules for one or more days and return them as an ``AtmozDataset``.

        Only the requested pixels are transferred; granules are never saved. The result is not cached: for repeated work on
        the same granules use :meth:`fetch` and then :meth:`read`.

        Parameters
        ----------
        start_date, end_date:
            Inclusive UTC days. ``end_date`` defaults to ``start_date``. Granules are selected by scan start time.
        points:
            ``{name: (lat, lon)}``, a list of ``(lat, lon)`` or one ``(lat, lon)``. The nearest pixel is returned; a point more
            than one pixel outside the grid is skipped with a warning.
        bbox:
            ``(west, south, east, north)`` in degrees: every pixel inside (``"bbox"`` region). Pixels that are all NaN are dropped.
        variables:
            Names under ``product/`` or ``support_data/``. Default: the product's main column and its uncertainty
            (``NO2_L3``), or ``ozone_profile`` (``O3PROF_L3``).
        max_quality:
            Column products: keep pixels whose ``main_data_quality_flag`` is <= this (0 normal, 1 suspicious, 2 bad); the others
            become NaN. ``None`` disables the masking. Profile products have no quality flag.
        lowest_layer:
            Profile products: read only the bottom retrieval layer (much cheaper for a large box). In ``O3PROF_L3`` V04 that
            layer spans the ground to about 2 km above it; ``alt_lower`` / ``alt_upper`` give the actual limits.
        only_scans:
            Scan start times (UTC), e.g. ``frame.t`` of an earlier download: granules whose scan does not start within two minutes
            before one of them are skipped, which avoids re-probing scans known not to cover the site.
        max_workers, mode:
            Parallel granule readers. ``"thread"`` (default) or ``"process"``: h5py holds a global lock during remote reads, so
            threads do not overlap network waits and ``"process"`` is much faster for many granules (each worker logs in itself).
        retries:
            Attempts per granule for dropped connections (the pause grows with each).

        Raises
        ------
        NoDataError: no granule for those days, or none holds a pixel for the request.
        DownloadError: a granule could not be read (nothing is returned).
        """
        spec = self._validate(points, bbox, variables, max_quality, lowest_layer)
        window = self._window(start_date, end_date)
        granules = self._search_or_raise(start_date, end_date, only_scans)
        jobs = [{"name": _granule_name(g), "source": g, "spec": spec, "retries": retries} for g in granules]
        results = _run_reads(jobs, max_workers=max_workers, mode=mode, silent=silent, show_traceback=show_traceback)
        return self._assemble(results, spec, (window[0].date(), (window[1] - pd.Timedelta(days=1)).date()))

    @classmethod
    def read(cls,
             paths: Union[PathLike, Iterable[PathLike]],
             *,
             points: Optional[Union[Dict[str, Point], Sequence[Point], Point]] = None,
             bbox: Optional[BBox] = None,
             variables: Optional[Union[str, Sequence[str]]] = None,
             max_quality: Optional[int] = 0,
             lowest_layer: bool = False,
             product: Optional[str] = None,
             max_workers: int = 1,
             mode: str = "thread",
             silent: bool = True,
             show_traceback: bool = False) -> AtmozDataset:
        """Subset TEMPO granules that are already on disk (a path, a directory or a list, e.g. from :meth:`fetch`).

        The same subsetting and the same result as :meth:`download`; no login or network is needed. The product and version are
        taken from the file names (``TEMPO_NO2_L3_V04_...nc``) unless *product* is given.
        """
        files = _expand(paths)
        if not files:
            raise NoDataError(f"No granule files found in {paths!r}.")
        match = _PRODUCT_KEY.match(files[0].name)
        key = product or (_BY_SHORT_NAME.get(match.group(1)) if match else None)
        if key is None:
            raise ValueError(f"Cannot tell the TEMPO product from {files[0].name!r}; pass product=" + repr(list(_PRODUCTS)) + ".")
        client = cls(key, match.group(2) if match else "V04")
        for f in files:
            m = _PRODUCT_KEY.match(f.name)
            if m and (m.group(1), m.group(2)) != (client.short_name, client.version):
                raise ValueError(f"{f.name} is {m.group(1)} {m.group(2)}, the other files are {client.short_name} {client.version}.")
        spec = client._validate(points, bbox, variables, max_quality, lowest_layer)
        jobs = [{"name": f.name, "source": f, "spec": spec, "retries": 1} for f in files]
        results = _run_reads(jobs, max_workers=max_workers, mode=mode, silent=silent, show_traceback=show_traceback)
        starts = sorted(t for t in (pd.Timestamp(r["scan"]) for r in results.values()))
        return client._assemble(results, spec, (starts[0].date(), starts[-1].date()))

    def fetch(self,
              start_date: Union[str, date, datetime],
              end_date: Optional[Union[str, date, datetime]] = None,
              *,
              max_workers: int = 3,
              retries: int = 6,
              refresh: bool = False,
              cache_dir: Optional[PathLike] = None,
              silent: bool = False,
              show_traceback: bool = False) -> Dict[str, Path]:
        """Download whole granules (every variable, the full grid) for the days and return ``{file name: Path}``.

        Granules are 400-550 MB for ``O3PROF_L3``, so few connections and resuming matter more than speed: a dropped
        connection resumes with an HTTP Range request, and a complete file is kept and skipped next time (``refresh=True``
        downloads again). Files go to ``cache_dir`` (default: the atmoz cache, namespace ``tempo``). Parse them with :meth:`read`.

        Raises
        ------
        NoDataError: no granule for those days.
        DownloadError: a download failed after *retries* attempts (finished files stay in the cache).
        """
        granules = self._search_or_raise(start_date, end_date)
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("tempo")
        jobs = [{"name": _link(g).rsplit("/", 1)[-1], "url": _link(g)} for g in granules]

        def one(job):
            dest = dest_dir / job["name"]
            if dest.exists() and not refresh:
                return dest
            return _download_file(job["url"], dest, retries)

        return run_jobs(one, jobs, key=lambda j: j["name"], describe=lambda j: j["name"], noun="granule download",
                        desc="Downloading TEMPO granules", max_workers=max_workers, silent=silent, show_traceback=show_traceback)


def _naive_utc(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(list(values))
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def _expand(paths) -> List[Path]:
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files: List[Path] = []
    for p in items:
        p = Path(p)
        files.extend(sorted(p.glob("TEMPO_*.nc")) if p.is_dir() else [p])
    return files


if __name__ == "__main__":
    ds = TEMPO("NO2_L3").download("2025-07-28", points={"GSFC": (38.98, -76.83)})
    print(ds.inventory())
