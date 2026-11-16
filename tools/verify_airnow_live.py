"""Check, against the live AirNow API, the assumption behind atmoz's request windows.

atmoz splits a date range into windows and assumes AirNow returns the ``endDate`` hour (inclusive), so a window ends one hour
before the next begins (``AirNowAPI.window_end_inclusive = True``). If that is wrong, hours are duplicated or lost.

Usage (needs a free key from https://docs.airnowapi.org/):

    ATMOZ_AIRNOW_API_KEY=... python tools/verify_airnow_live.py

Makes 3 small requests (a one-day, small-box window and its neighbours) and prints a verdict. The key is never printed.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from atmoz.surface.AirNow_API import AirNowAPI, make_windows


def hours_returned(api: AirNowAPI, start: str, end: str) -> list[str]:
    """The distinct UTC hours the API returns for one request window."""
    job = {"startDate": start, "endDate": end, "parameters": "OZONE", "BBOX": "-77.5,38.5,-76.5,39.5",
           "dataType": "B", "format": "application/json", "verbose": 1, "monitorType": 0, "includerawconcentrations": 0}
    response = requests.get(api._url(job, with_key=True), timeout=60)
    if response.status_code != 200:
        sys.exit(f"AirNow answered HTTP {response.status_code} (check the key and the query); not printing the URL.")
    frame = pd.DataFrame(response.json())
    return sorted(frame["UTC"].unique()) if not frame.empty else []


def main() -> int:
    key = os.environ.get("ATMOZ_AIRNOW_API_KEY")
    if not key:
        print("Set ATMOZ_AIRNOW_API_KEY first (free key: https://docs.airnowapi.org/).")
        return 2
    api = AirNowAPI(api_key=key)
    day = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    start, end = f"{day}T00", f"{day}T23"
    got = hours_returned(api, start, end)
    if not got:
        print(f"No ozone observations came back for {day} in the test box; try again later.")
        return 2
    print(f"Requested {start} .. {end}: AirNow returned {len(got)} distinct hours, first {got[0]}, last {got[-1]}.")
    inclusive = got[-1].endswith("T23:00") or got[-1].endswith("T23")
    print("endDate hour IS returned (inclusive)." if inclusive else "endDate hour is NOT returned (exclusive).")

    windows = make_windows(f"{day}T00", f"{day}T23", "12h", end_inclusive=True)
    first, second = (hours_returned(api, *w) for w in windows)
    overlap, union = sorted(set(first) & set(second)), sorted(set(first) | set(second))
    print(f"Two 12 h windows {windows}: overlap {len(overlap)} hour(s), together {len(union)} of {len(got)} hours.")

    ok = inclusive and not overlap and len(union) == len(got)
    if ok:
        print("RESULT: OK. window_end_inclusive = True is correct; no hour is duplicated or lost.")
        return 0
    print("RESULT: MISMATCH. Set AirNowAPI.window_end_inclusive = False (and re-run this script), then update the changelog.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
