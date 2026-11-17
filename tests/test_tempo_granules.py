"""TEMPO: reading granules. Tiny synthetic HDF5 files built with h5py stand in for the L3 products; Earthdata is faked. No network."""
import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

import atmoz  # noqa: E402
from atmoz.dataclasses import AtmozDataset, AtmozFrame  # noqa: E402
from atmoz.exceptions import DownloadError, NoDataError  # noqa: E402
from atmoz.satellite import tempo as tp  # noqa: E402
from atmoz.satellite.tempo import TEMPO, du_to_ppbv, integrate_layers  # noqa: E402

GPS_EPOCH = pd.Timestamp("1980-01-06", tz="UTC")
LAT = np.array([38.0, 38.25, 38.5, 38.75, 39.0])
LON = np.array([-77.0, -76.75, -76.5, -76.25, -76.0])
FILL = -1e30


def _stamp(scan: str) -> str:
    return pd.Timestamp(scan, tz="UTC").strftime("%Y%m%dT%H%M%SZ")


def _common(f, scan: str):
    f.create_dataset("latitude", data=LAT)
    f.create_dataset("longitude", data=LON)
    seconds = (pd.Timestamp(scan, tz="UTC") - GPS_EPOCH).total_seconds()
    f.create_dataset("time", data=np.array([seconds])).attrs["units"] = "seconds since 1980-01-06T00:00:00Z"


def make_no2(path, scan="2025-07-28 12:00:00"):
    """Column of pixel (i, j) = 1e15 * (10 i + j + 1); flag 0 except pixel (1, 1) = 2 and a fill value at (2, 2)."""
    with h5py.File(path, "w") as f:
        _common(f, scan)
        ii, jj = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
        col = (1e15 * (10 * ii + jj + 1))[None]
        col[0, 2, 2] = FILL
        flag = np.zeros((1, 5, 5), dtype="i2")
        flag[0, 1, 1] = 2
        p = f.create_group("product")
        d = p.create_dataset("vertical_column_troposphere", data=col)
        d.attrs["units"], d.attrs["_FillValue"] = "molecules/cm^2", FILL
        p.create_dataset("vertical_column_troposphere_uncertainty", data=col * 0.1).attrs["units"] = "molecules/cm^2"
        p.create_dataset("main_data_quality_flag", data=flag)
    return path


def make_o3prof(path, scan="2025-07-28 12:00:00", unobserved=((0, 0),)):
    """Four layers, top -> bottom, over a ground at 0.2 + 0.1 i km: partial columns 20, 10, 5, 3 DU (x (1 + 0.1 j))."""
    nl = 4
    pb = np.zeros((1, 5, 5, nl, 2))
    ab = np.zeros((1, 5, 5, nl, 2))
    alt = np.zeros((1, 5, 5, nl))
    prof = np.zeros((1, 5, 5, nl))
    total = np.full((1, 5, 5), 300.0)
    p_edges = [0.0, 100.0, 300.0, 700.0, 1000.0]          # hPa: top ... ground
    for i in range(5):
        ground = 0.2 + 0.1 * i
        a_edges = [60.0, 20.0, 8.0, ground + 2.0, ground]  # km: top ... ground (the bottom layer is ground .. +2 km)
        for k in range(nl):
            pb[0, i, :, k] = (p_edges[k], p_edges[k + 1])
            ab[0, i, :, k] = (a_edges[k], a_edges[k + 1])
            alt[0, i, :, k] = (a_edges[k] + a_edges[k + 1]) / 2
        for j in range(5):
            prof[0, i, j] = np.array([20.0, 10.0, 5.0, 3.0]) * (1 + 0.1 * j)
    for i, j in unobserved:
        total[0, i, j] = np.nan
    with h5py.File(path, "w") as f:
        _common(f, scan)
        p = f.create_group("product")
        p.create_dataset("ozone_profile", data=prof).attrs["units"] = "DU"
        p.create_dataset("ozone_profile_error", data=prof * 0.05).attrs["units"] = "DU"
        p.create_dataset("total_ozone_column", data=total)
        s = f.create_group("support_data")
        s.create_dataset("ozone_profile_pressure_bounds", data=pb).attrs["units"] = "hPa"
        s.create_dataset("ozone_profile_altitude_bounds", data=ab).attrs["units"] = "km"
        s.create_dataset("ozone_profile_altitude", data=alt).attrs["units"] = "km"
        s.create_dataset("eff_cloud_fraction", data=np.full((1, 5, 5), 0.25))
    return path


class Granule:
    """A fake earthaccess granule backed by a local file: ``_open`` hands the file back instead of streaming it."""

    def __init__(self, path):
        self.path = path

    def data_links(self):
        return [f"https://data.example.nasa/{self.path.name}"]


class EarthData:
    def __init__(self, granules):
        self.granules = granules

    def search_data(self, **params):
        return list(self.granules)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tp, "_RETRY_DELAY", 0.0)
    monkeypatch.setattr(tp, "_open", lambda g: g.path if isinstance(g, Granule) else g)


def stream(tmp_path, product, scans, maker):
    """A client whose search finds one fake granule per scan time."""
    granules = []
    for k, scan in enumerate(scans):
        name = f"TEMPO_{product}_V04_{_stamp(scan)}_S{k:03d}G01.nc"
        granules.append(Granule(maker(tmp_path / name, scan)))
    return TEMPO(product, earthdata=EarthData(granules))


# ---- columns -------------------------------------------------------------------------------------------------------------------
def test_point_download_returns_one_frame_per_variable_with_utc_time_and_units(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00", "2025-07-28 13:00"], make_no2)
    ds = t.download("2025-07-28", points={"site": (38.8, -76.3)}, silent=True)
    assert isinstance(ds, AtmozDataset) and ds.source == "TEMPO"
    assert sorted(ds.keys()) == ["site/vertical_column_troposphere", "site/vertical_column_troposphere_uncertainty"]
    f = ds["site/vertical_column_troposphere"]
    assert isinstance(f, AtmozFrame) and (f.axes["lat"], f.axes["lon"], f.axes["alt"]) == ("lat", "lon", None)
    assert str(f.t.tz) == "UTC" and list(f.t) == [pd.Timestamp("2025-07-28 12:00", tz="UTC"), pd.Timestamp("2025-07-28 13:00", tz="UTC")]
    assert f.units["value"] == "molecules/cm2"
    assert f.df["lat"].tolist() == [38.75, 38.75] and f.df["lon"].tolist() == [-76.25, -76.25]     # the nearest pixel centre
    assert f.df["value"].tolist() == [1e15 * 34, 1e15 * 34] and f.df["quality_flag"].dtype == "Int8"
    assert f.metadata["product"] == "NO2_L3" and f.metadata["version"] == "V04" and f.metadata["lat_requested"] == 38.8
    assert "alt_reference" not in f.metadata
    assert ds.metadata["granules"] == sorted(ds.metadata["granules"]) and len(ds.metadata["granules"]) == 2
    assert ds.metadata["start"] == "2025-07-28" and ds.metadata["units"]["vertical_column_troposphere"] == "molecules/cm^2"


def test_quality_mask_and_fill_values_become_nan(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    v = "vertical_column_troposphere"
    bad = t.download("2025-07-28", points={"bad": (38.25, -76.75)}, silent=True)[f"bad/{v}"]       # flag 2
    assert np.isnan(bad.df["value"].iloc[0]) and bad.df["quality_flag"].iloc[0] == 2                  # row kept: coverage visible
    keep = t.download("2025-07-28", points={"bad": (38.25, -76.75)}, max_quality=None, silent=True)[f"bad/{v}"]
    assert keep.df["value"].iloc[0] == pytest.approx(1e15 * 12)
    fill = t.download("2025-07-28", points={"fill": (38.5, -76.5)}, max_quality=None, silent=True)[f"fill/{v}"]
    assert np.isnan(fill.df["value"].iloc[0])


def test_bbox_returns_every_pixel_inside_and_drops_all_nan_pixels(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    f = t.download("2025-07-28", bbox=(-76.8, 38.2, -76.2, 38.8), silent=True)["bbox/vertical_column_troposphere"]
    # lat 38.25 .. 38.75 x lon -76.75 .. -76.25: 3 x 3 pixels, the flagged (1,1) and the filled (2,2) are all-NaN -> dropped
    assert len(f) == 7 and set(f.df["lat"]) <= {38.25, 38.5, 38.75}
    assert f.metadata["bbox"] == (-76.8, 38.2, -76.2, 38.8) and "lat_requested" not in f.metadata
    assert (f.df["quality_flag"] == 0).all()


def test_a_point_outside_the_grid_warns_and_the_others_are_returned(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    with pytest.warns(UserWarning, match=r"Outside the TEMPO grid.*'far'"):
        ds = t.download("2025-07-28", points={"near": (38.5, -76.5), "far": (10.0, 10.0)}, variables="vertical_column_troposphere",
                        silent=True)
    assert list(ds.keys()) == ["near/vertical_column_troposphere"] and ds.metadata["skipped"] == ["far"]


def test_nothing_in_the_grid_is_a_no_data_error(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    with pytest.warns(UserWarning), pytest.raises(NoDataError, match="none holds a pixel"):
        t.download("2025-07-28", points={"far": (10.0, 10.0)}, silent=True)


def test_unknown_variable_names_what_is_available(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    with pytest.raises(DownloadError, match=r"Variable 'nope' not found.*product/vertical_column_troposphere"):
        t.download("2025-07-28", points=(38.5, -76.5), variables=["nope"], silent=True)


def test_points_default_names_and_sorting_over_scans(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 13:00", "2025-07-28 12:00", "2025-07-28 14:00"], make_no2)
    ds = t.download("2025-07-28", points=[(38.5, -76.5), (39.0, -76.0)], variables="vertical_column_troposphere", silent=True)
    assert sorted(ds.keys()) == ["38.500,-76.500/vertical_column_troposphere", "39.000,-76.000/vertical_column_troposphere"]
    assert all(f.t.is_monotonic_increasing and len(f) == 3 for f in ds.values())


def test_scan_time_units_other_than_seconds_are_read():
    class Dset:
        attrs = {"units": b"hours since 2025-07-28T00:00:00Z"}

        def __getitem__(self, key):
            return np.array([12.5])

    assert tp._scan_time(Dset()) == pd.Timestamp("2025-07-28 12:30", tz="UTC")
    Dset.attrs = {"units": "furlongs since never"}
    with pytest.raises(ValueError, match="Cannot read the granule time"):
        tp._scan_time(Dset())


# ---- retries -------------------------------------------------------------------------------------------------------------------
def test_dropped_connections_are_retried_then_succeed(tmp_path, monkeypatch):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    state = {"n": 0}

    def flaky(g):
        state["n"] += 1
        if state["n"] <= 2:
            raise ConnectionResetError("reset by peer")
        return g.path

    monkeypatch.setattr(tp, "_open", flaky)
    ds = t.download("2025-07-28", points=(38.5, -76.5), silent=True)
    assert state["n"] == 3 and len(ds) == 2


def test_persistent_failures_raise_download_error_naming_the_granule(tmp_path, monkeypatch):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00", "2025-07-28 13:00"], make_no2)
    state = {"n": 0}

    def down(g):
        state["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(tp, "_open", down)
    with pytest.raises(DownloadError, match=r"(?s)2 of 2 granule read\(s\) failed.*timed out") as err:
        t.download("2025-07-28", points=(38.5, -76.5), retries=2, silent=True)
    assert state["n"] == 4 and "TEMPO_NO2_L3_V04_20250728T120000Z" in str(err.value)


def test_a_bad_mode_is_rejected(tmp_path):
    t = stream(tmp_path, "NO2_L3", ["2025-07-28 12:00"], make_no2)
    with pytest.raises(ValueError, match="mode must be"):
        t.download("2025-07-28", points=(38.5, -76.5), mode="gpu", silent=True)


# ---- ozone profiles ------------------------------------------------------------------------------------------------------------
def test_profile_point_is_long_in_ppb_with_km_altitudes_and_a_declared_reference(tmp_path):
    t = stream(tmp_path, "O3PROF_L3", ["2025-07-28 12:00", "2025-07-28 13:00"], make_o3prof)
    ds = t.download("2025-07-28", points={"p": (38.25, -76.5)}, silent=True)
    assert list(ds.keys()) == ["p/ozone_profile"]
    f = ds["p/ozone_profile"]
    assert len(f) == 8 and (f.axes["lat"], f.axes["lon"], f.axes["alt"]) == ("lat", "lon", "alt") and str(f.t.tz) == "UTC"
    assert f.units["value"] == "ppb" and f.units["alt"] == "km" and f.units["partial_column"] == "DU"
    assert f.alt_reference == "asl" and ds.metadata["alt_reference"] == "asl"
    assert f.metadata["site_elevation_km"] == pytest.approx(0.3)             # the bottom of the lowest layer
    first = f.df[f.df.index == f.df.index[0]]
    assert first["layer"].tolist() == [0, 1, 2, 3]                            # top -> bottom as in the file
    assert first["partial_column"].tolist() == pytest.approx([24.0, 12.0, 6.0, 3.6])    # pixel j = 2 -> x 1.2
    assert first["value"].tolist() == pytest.approx(du_to_ppbv(first["partial_column"], [100, 200, 400, 300]).tolist())
    assert first["alt"].iloc[3] == pytest.approx((2.3 + 0.3) / 2) and first["alt_lower"].iloc[3] == pytest.approx(0.3)
    assert first["total_ozone_column"].eq(300.0).all() and first["eff_cloud_fraction"].eq(0.25).all()


def test_unobserved_pixels_are_left_out_not_filled_with_nan(tmp_path):
    t = stream(tmp_path, "O3PROF_L3", ["2025-07-28 12:00"], make_o3prof)
    with pytest.raises(NoDataError, match="none holds a pixel"):
        t.download("2025-07-28", points={"unseen": (38.0, -77.0)}, silent=True)       # pixel (0, 0) has no total column


def test_lowest_layer_is_the_bottom_layer_of_the_full_profile(tmp_path):
    t = stream(tmp_path, "O3PROF_L3", ["2025-07-28 12:00"], make_o3prof)
    box = (-76.6, 38.4, -76.4, 38.6)
    full = t.download("2025-07-28", bbox=box, silent=True)["bbox/ozone_profile"]
    low = t.download("2025-07-28", bbox=box, lowest_layer=True, silent=True)["bbox/ozone_profile"]
    assert len(full) == 4 * len(low) and (low["layer"] == 3).all() and low.metadata["lowest_layer"] is True
    bottom = full.df[full.df["layer"] == 3].sort_values(["lat", "lon"])
    assert low.df.sort_values(["lat", "lon"])["value"].tolist() == pytest.approx(bottom["value"].tolist())
    assert (low.df["alt_upper"] - low.df["alt_lower"]).tolist() == pytest.approx([2.0] * len(low))
    assert "site_elevation_km" not in low.metadata                            # a box has no single ground height


def test_integrated_ozone_matches_the_lowest_layer_when_the_slab_is_that_layer(tmp_path):
    t = stream(tmp_path, "O3PROF_L3", ["2025-07-28 12:00"], make_o3prof)
    pt = {"p": (38.5, -76.5)}
    full = t.download("2025-07-28", points=pt, silent=True)
    low = t.download("2025-07-28", points=pt, lowest_layer=True, silent=True)
    ground = full["p/ozone_profile"].metadata["site_elevation_km"]
    agl = integrate_layers(full, 0.0, 2.0, reference="agl")["p/ozone_profile"]
    assert agl.df["column"].iloc[0] == pytest.approx(low["p/ozone_profile"].df["partial_column"].iloc[0])
    assert agl.df["value"].iloc[0] == pytest.approx(low["p/ozone_profile"].df["value"].iloc[0])
    asl = integrate_layers(full, 0.0, 2.0)["p/ozone_profile"]                 # ground .. 2 km asl is only part of the bottom layer
    share = (2.0 - ground) / 2.0
    assert asl.df["column"].iloc[0] == pytest.approx(low["p/ozone_profile"].df["partial_column"].iloc[0] * share)


def test_error_variable_is_a_separate_frame_and_lowest_layer_needs_a_profile_product(tmp_path):
    t = stream(tmp_path, "O3PROF_L3", ["2025-07-28 12:00"], make_o3prof)
    ds = t.download("2025-07-28", points=(38.5, -76.5), variables=["ozone_profile", "ozone_profile_error"], silent=True)
    key = "38.500,-76.500"
    assert ds[f"{key}/ozone_profile_error"].df["partial_column"].iloc[0] == pytest.approx(
        0.05 * ds[f"{key}/ozone_profile"].df["partial_column"].iloc[0])


def test_profile_frames_work_with_atmoz_vertical_and_collocate(tmp_path):
    t = stream(tmp_path, "O3PROF_L3", ["2025-07-28 12:00", "2025-07-28 13:00"], make_o3prof)
    f = t.download("2025-07-28", points={"p": (38.5, -76.5)}, silent=True)["p/ozone_profile"]
    lm = atmoz.vertical.layer_mean(f, 0.0, 3.0, min_points=1)              # same asl reference
    assert len(lm) == 2 and lm.units["value"] == "ppb"
    layer = integrate_layers(f, 0.0, 10.0)
    out = atmoz.collocate([layer, lm], names=["tempo", "other"], max_gap="3h")
    assert {"tempo", "other"} <= set(out.df.columns) and len(out) == 2


# ---- local files: fetch() then read() ------------------------------------------------------------------------------------------
def test_read_subsets_local_granules_and_infers_the_product(tmp_path):
    make_no2(tmp_path / "TEMPO_NO2_L3_V04_20250728T120000Z_S001G01.nc")
    make_no2(tmp_path / "TEMPO_NO2_L3_V04_20250728T130000Z_S002G01.nc", "2025-07-28 13:00")
    ds = TEMPO.read(tmp_path, points={"a": (38.5, -76.5)})
    f = ds["a/vertical_column_troposphere"]
    assert len(f) == 2 and ds.name == "TEMPO NO2_L3 V04" and ds.metadata["start"] == "2025-07-28"
    one = TEMPO.read([tmp_path / "TEMPO_NO2_L3_V04_20250728T120000Z_S001G01.nc"], bbox=(-77, 38, -76, 39))
    assert len(one["bbox/vertical_column_troposphere"]) == 23                  # 25 pixels minus the flagged and the filled


def test_read_in_process_mode_gives_the_same_result(tmp_path):
    make_o3prof(tmp_path / "TEMPO_O3PROF_L3_V04_20250728T120000Z_S001G01.nc")
    make_o3prof(tmp_path / "TEMPO_O3PROF_L3_V04_20250728T130000Z_S002G01.nc", "2025-07-28 13:00")
    kw = {"points": {"p": (38.5, -76.5)}, "max_workers": 2}
    thread = TEMPO.read(tmp_path, mode="thread", **kw)["p/ozone_profile"]
    process = TEMPO.read(tmp_path, mode="process", **kw)["p/ozone_profile"]
    pd.testing.assert_frame_equal(thread.df, process.df)


def test_read_errors(tmp_path):
    with pytest.raises(NoDataError, match="No granule files"):
        TEMPO.read(tmp_path, points=(38.5, -76.5))
    make_no2(tmp_path / "mystery.nc")
    with pytest.raises(ValueError, match=r"Cannot tell the TEMPO product.*NO2_L3"):
        TEMPO.read(tmp_path / "mystery.nc", points=(38.5, -76.5))
    assert len(TEMPO.read(tmp_path / "mystery.nc", product="NO2_L3", points=(38.5, -76.5))) == 2
    make_no2(tmp_path / "TEMPO_NO2_L3_V04_20250728T120000Z_S001G01.nc")
    make_o3prof(tmp_path / "TEMPO_O3PROF_L3_V04_20250728T130000Z_S002G01.nc")
    with pytest.raises(ValueError, match="the other files are"):
        TEMPO.read([tmp_path / "TEMPO_NO2_L3_V04_20250728T120000Z_S001G01.nc",
                    tmp_path / "TEMPO_O3PROF_L3_V04_20250728T130000Z_S002G01.nc"], points=(38.5, -76.5))
    (tmp_path / "TEMPO_NO2_L3_V04_20250728T140000Z_S003G01.nc").write_bytes(b"not hdf5")
    with pytest.raises(DownloadError, match="TEMPO_NO2_L3_V04_20250728T140000Z"):
        TEMPO.read(tmp_path / "TEMPO_NO2_L3_V04_20250728T140000Z_S003G01.nc", product="NO2_L3", points=(38.5, -76.5))
