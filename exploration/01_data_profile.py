"""Download the Vadimkin air-raid sirens dataset and print a coverage profile.

Usage: uv run --with pandas python exploration/01_data_profile.py
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        urlretrieve(f"{BASE}/{name}", path)
    return path


def main() -> None:
    o = pd.read_csv(fetch("official_data_en.csv"), parse_dates=["started_at", "finished_at"])
    v = pd.read_csv(fetch("volunteer_data_en.csv"), parse_dates=["started_at", "finished_at"])

    print("official :", o.started_at.min(), "->", o.started_at.max(), f"({len(o)} rows)")
    print("volunteer:", v.started_at.min(), "->", v.started_at.max(), f"({len(v)} rows)")
    print("volunteer naive share:", round(v.naive.mean(), 3))

    # Oblast -> raion regime shift (2025)
    q = o.started_at.dt.tz_localize(None).dt.to_period("Q")
    print(pd.crosstab(q, o.level).tail(8))

    dur = (o.finished_at - o.started_at).dt.total_seconds() / 60
    print("official median duration, min:", round(dur.median(), 1))


if __name__ == "__main__":
    main()
