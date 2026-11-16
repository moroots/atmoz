import pandas as pd

from atmoz.surface import utilities as su


def _expected():
    return pd.Series(pd.to_datetime(["2024-01-01 05:00", "2024-01-02 23:30"]))


def test_make_datetime_with_datetime_date_column():
    df = pd.DataFrame({"Date GMT": pd.to_datetime(["2024-01-01", "2024-01-02"]), "Time GMT": ["05:00", "23:30"]})
    pd.testing.assert_series_equal(su.make_datetime(df), _expected(), check_names=False)


def test_make_datetime_with_text_and_category_columns():
    df = pd.DataFrame({"Date GMT": ["2024-01-01", "2024-01-02"], "Time GMT": pd.Categorical(["05:00", "23:30"])})
    pd.testing.assert_series_equal(su.make_datetime(df), _expected(), check_names=False)


def test_make_datetime_old_format_with_trailing_space_still_works():
    df = pd.DataFrame({"Date GMT": ["2024-01-01", "2024-01-02"], "Time GMT": ["05:00", "23:30"]})
    cols = {"date": "Date GMT", "time": "Time GMT", "format": {"date": "%Y-%m-%d ", "time": "%H:%M"}}
    pd.testing.assert_series_equal(su.make_datetime(df, cols), _expected(), check_names=False)
