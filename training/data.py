"""Loading the volunteer (eTryvoga) alert dataset at oblast level.

All timestamps are UTC. An alert with ``finished_at = NaT`` is treated as still active.
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
VOLUNTEER_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/volunteer_data_en.csv"
)
VOLUNTEER_CSV = RAW_DIR / "volunteer_data_en.csv"

# Start of the training data. Window length barely matters on validation (3-24 months are within noise,
# exploration/06_train_window.py); 24 months + recency weights (start_model.HALF_LIFE_DAYS) won in both windows.
TRAIN_START = pd.Timestamp("2024-09-01", tz="UTC")
# The duration model and its feature study (exploration/05) were validated on data from this date; kept as is.
DURATION_TRAIN_START = pd.Timestamp("2025-10-01", tz="UTC")
TEST_DAYS = 28
KYIV_TZ = "Europe/Kyiv"
# Permanent alert since 2022-04-04; its rows are only from before that and carry no signal.
EXCLUDED_REGIONS = {"Luhanska oblast"}


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not VOLUNTEER_CSV.exists():
        urlretrieve(VOLUNTEER_URL, VOLUNTEER_CSV)
    return VOLUNTEER_CSV


def load_alerts(path: Path | None = None) -> pd.DataFrame:
    """Return alerts sorted by (region, started_at) with a ``duration_min`` column."""
    df = pd.read_csv(path or download(), parse_dates=["started_at", "finished_at"])
    df = df.drop_duplicates(["region", "started_at"]).sort_values(["region", "started_at"])
    df["duration_min"] = (df.finished_at - df.started_at).dt.total_seconds() / 60
    return df.reset_index(drop=True)


def regions(alerts: pd.DataFrame) -> list[str]:
    return sorted(set(alerts.region.unique()) - EXCLUDED_REGIONS)


def truncate_at(alerts: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """What was knowable at time ``t``: no alert starting at/after t, unknown end for ongoing ones."""
    known = alerts[alerts.started_at < t].copy()
    ongoing = known.finished_at > t
    known.loc[ongoing, "finished_at"] = pd.NaT
    known.loc[ongoing, "duration_min"] = float("nan")
    return known
