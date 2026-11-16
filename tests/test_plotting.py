"""Offline tests for atmoz.plotting: dispatch, styles, return values and no global side effects."""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import atmoz
from atmoz.plotting import Style, plot, styles
from atmoz.resources import default_plot_params, useful_functions


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def curtain_df():
    times = pd.date_range("2025-07-28 12:00", periods=24, freq="10min", tz="UTC")
    altitude_km = np.linspace(0.2, 10, 40)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(20, 90, (len(times), len(altitude_km))), index=times, columns=altitude_km)


# ---- entry point ------------------------------------------------------------------------------------

def test_plot_by_name_and_by_attribute_are_the_same(curtain_df):
    fig1, ax1 = plot("curtain", curtain_df)
    fig2, ax2 = plot.curtain(curtain_df)
    assert fig1 is not fig2
    assert ax1.get_ylabel() == ax2.get_ylabel() == "Altitude (km ASL)"


def test_top_level_names_resolve():
    assert atmoz.plot is plot
    assert atmoz.plotting.styles is styles
    assert isinstance(atmoz.plotting.styles.curtain, Style)


def test_unknown_plot_suggests_the_right_name():
    with pytest.raises(AttributeError, match="Did you mean 'curtain'"):
        plot("curtatin", None)
    with pytest.raises(AttributeError, match="Did you mean 'windrose'"):
        plot.windrse


def test_unknown_style_suggests_the_right_name():
    with pytest.raises(KeyError, match="Did you mean 'curtain'"):
        styles["curtian"]


def test_plot_lists_itself():
    text = repr(plot)
    assert all(name in text for name in ("curtain", "profile", "windrose", "site_map"))
    assert set(dir(plot)) == {"curtain", "profile", "windrose", "site_map", "skewt", "comparison_profile", "exceedances"}


def test_import_is_light():
    code = (
        "import sys, atmoz, atmoz.plotting\n"
        "heavy = {'matplotlib', 'plotly', 'windrose', 'geopandas', 'xarray'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- styles -----------------------------------------------------------------------------------------

def test_plotting_never_changes_global_rcparams(curtain_df):
    before = dict(matplotlib.rcParams)
    plot.curtain(curtain_df)
    plot.curtain(curtain_df, style={"font.size": 18})
    assert dict(matplotlib.rcParams) == before


def test_style_none_uses_callers_settings(curtain_df):
    with matplotlib.rc_context({"axes.spines.top": True}):
        _, ax = plot.curtain(curtain_df, style=None)
    assert ax.spines["top"].get_visible()


def test_style_object_name_and_dict_all_work(curtain_df):
    _, ax_obj = plot.curtain(curtain_df, style=atmoz.plotting.styles.curtain)
    _, ax_name = plot.curtain(curtain_df, style="curtain")
    _, ax_dict = plot.curtain(curtain_df, style={"axes.titlesize": 20})
    assert ax_obj.title.get_fontsize() == ax_name.title.get_fontsize() == styles.curtain.rc["axes.titlesize"]
    assert ax_dict.title.get_fontsize() == 20


def test_style_is_immutable_and_replace_makes_a_copy():
    small = styles.curtain.replace(figsize=(6, 2))
    assert small.figsize == (6, 2) and styles.curtain.figsize == (11, 4.4)
    with pytest.raises(Exception):
        styles.curtain.figsize = (1, 1)


def test_figsize_comes_from_the_style_and_can_be_overridden(curtain_df):
    fig, _ = plot.curtain(curtain_df)
    assert tuple(fig.get_size_inches()) == (11, 4.4)
    fig, _ = plot.curtain(curtain_df, figsize=(8, 3))
    assert tuple(fig.get_size_inches()) == (8, 3)


def test_font_stack_ends_in_a_font_that_always_exists():
    assert styles.curtain.rc["font.family"][-1] == "DejaVu Sans"
    assert "Courier New" not in str(styles.curtain.rc)


# ---- plots ------------------------------------------------------------------------------------------

def test_curtain_returns_fig_and_ax_and_stays_open(curtain_df):
    fig, ax = plot.curtain(curtain_df, title="Test")
    assert fig.axes and ax.get_title(loc="left") == "Test"
    assert plt.fignum_exists(fig.number)          # not closed: notebooks show it, fig.savefig works
    assert len(fig.axes) == 2                      # axes + colorbar


def test_curtain_accepts_a_dict_of_frames_and_meters(curtain_df):
    in_m = curtain_df.copy()
    in_m.columns = in_m.columns * 1000
    fig, ax = plot.curtain({"2025-07-28": in_m}, altitude_units="m")
    assert ax.get_ylim() == (0, 15)


def test_curtain_into_existing_axes(curtain_df):
    fig, axes = plt.subplots(1, 2)
    out_fig, out_ax = plot.curtain(curtain_df, ax=axes[0])
    assert out_fig is fig and out_ax is axes[0]


def test_curtain_rejects_bad_input():
    with pytest.raises(TypeError):
        plot.curtain([1, 2, 3])
    with pytest.raises(ValueError, match="Unknown variable"):
        plot.curtain(pd.DataFrame(), variable="unobtainium")


def test_profile_returns_fig_and_ax():
    y = np.linspace(0, 5, 30)
    fig, ax = plot.profile({"sonde": {"X": 40 + 5 * y, "Y": y, "uncertainty": np.full(30, 3.0)}})
    assert ax.get_xlabel().startswith("Ozone") and ax.get_ylim() == (0, 5)
    assert tuple(fig.get_size_inches()) == (4.2, 5.6)


def test_profile_needs_x_and_y():
    with pytest.raises(KeyError):
        plot.profile({"bad": {"X": [1, 2]}})


def test_windrose_returns_fig_and_ax():
    pytest.importorskip("windrose")
    rng = np.random.default_rng(1)
    fig, ax = plot.windrose(rng.uniform(0, 360, 500), rng.gamma(2, 2, 500))
    assert ax.get_title(loc="left") == "Wind rose"


def test_site_map_returns_a_plotly_figure():
    pytest.importorskip("plotly")
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    sites = {
        "Hampton": gpd.GeoDataFrame({"id": [1]}, geometry=[Point(-76.34, 37.02)]),
        "Baltimore": gpd.GeoDataFrame({"id": [2]}, geometry=[Point(-76.61, 39.29)]),
    }
    fig = plot.site_map(sites, title="Sites", hover_cols=["id"])
    assert [t.name for t in fig.data] == ["Hampton", "Baltimore"]
    assert fig.layout.title.text == "Sites"


# ---- regression: shared default dicts ---------------------------------------------------------------

def test_merge_dicts_does_not_share_or_mutate_defaults():
    defaults = default_plot_params.tolnet_plot_params
    merged = useful_functions.merge_dicts(defaults, {"ax.set_title": {"label": "x"}})
    assert merged["fig.colorbar"] is not defaults["fig.colorbar"]
    merged["fig.colorbar"]["mappable"] = object()
    merged["fig.colorbar"].pop("sub_functions")
    assert "mappable" not in defaults["fig.colorbar"] and "sub_functions" in defaults["fig.colorbar"]


# ---- bundled font -----------------------------------------------------------------------------------

def test_inter_is_bundled_and_used(curtain_df):
    from atmoz.plotting.fonts import font_dir

    assert (font_dir() / "Inter-Regular.ttf").is_file() and (font_dir() / "Inter-Bold.ttf").is_file()
    assert (font_dir() / "Inter-LICENSE-OFL.txt").is_file()   # the OFL requires the license to travel with the font
    _, ax = plot.curtain(curtain_df, title="Inter")
    assert ax.title.get_fontname() == "Inter"
    assert ax.yaxis.label.get_fontname() == "Inter"


def test_curtain_accepts_a_long_atmoz_frame():
    import pandas as pd
    import matplotlib.pyplot as plt
    from atmoz import AtmozFrame
    from atmoz.plotting.curtain import curtain
    times = pd.date_range("2024-07-01", periods=3, freq="1h", tz="UTC")
    rows = [(t, a, 10.0 + a) for t in times for a in (0.5, 1.0, 1.5)]
    df = pd.DataFrame(rows, columns=["time", "alt", "value"]).set_index("time")
    af = AtmozFrame(df, units={"value": "ppb", "alt": "km"}, alt="alt")
    fig, ax = curtain(af)
    fig2, _ = curtain({"day": af})
    assert fig is not None and fig2 is not None
    plt.close("all")


def test_comparison_profile_and_exceedances():
    import matplotlib.pyplot as plt
    import pandas as pd
    from atmoz import AtmozDataset, AtmozFrame
    from atmoz import stats

    t = pd.DatetimeIndex(["2024-07-01 00:00", "2024-07-01 01:00"], tz="UTC", name="time")
    rows = [(tt, a, 10.0 + 5 * a, 12.0 + 5 * a) for tt in t for a in (0.5, 1.5, 2.5)]
    df = pd.DataFrame(rows, columns=["time", "alt", "lidar", "model"]).set_index("time")
    af = AtmozFrame(df, units={"lidar": "ppb", "alt": "km"}, alt="alt",
                    metadata={"alt_reference": "agl", "columns": {"lidar": "value", "model": "value"}})
    fig, ax = plot.comparison_profile(af, bins=1.0)
    assert [ln.get_label() for ln in ax.get_lines()] == ["lidar", "model"] and "AGL" in ax.get_ylabel()
    fig2, _ = plot.comparison_profile(AtmozDataset(frames={"a": af, "b": af}), columns=["model"])
    with pytest.raises(KeyError):
        plot.comparison_profile(af, columns=["nope"])
    with pytest.raises(ValueError):
        plot.comparison_profile(AtmozFrame(df.drop(columns="alt")))

    days = pd.date_range("2024-07-01", periods=3)
    table = pd.DataFrame({"State Code": ["24", "24"], "Site Num": ["1004", "1007"], "date": days[:2],
                          "mda8": [0.08, 0.06], "n_averages": 17, "valid": True, "start_hour": 10})
    summary = stats.exceedances(table, units="ppm")
    fig3, ax3 = plot.exceedances(summary)
    assert len(ax3.patches) == 2 and "70" in ax3.get_title("left")
    with pytest.raises(KeyError):
        plot.exceedances(table)
    plt.close("all")


def test_comparison_profile_ylim_clips_the_data():
    import matplotlib.pyplot as plt
    import pandas as pd
    from atmoz import AtmozFrame

    t = pd.DatetimeIndex(["2024-07-01 00:00"], tz="UTC", name="time")
    df = pd.DataFrame({"alt": [0.5, 1.5, 15.5], "obs": [40.0, 50.0, 900.0], "model": [41.0, 51.0, 950.0]}, index=t.repeat(3))
    af = AtmozFrame(df, units={"obs": "ppb"}, alt="alt", metadata={"alt_reference": "agl"})
    fig, ax = plot.comparison_profile(af, ylim=(0, 5))
    assert ax.get_xlim()[1] < 200
    plt.close("all")
