"""Leakage-safe features. Every feature at time ``t`` uses only alerts that started before ``t``
and end times that were already in the past at ``t`` (an ongoing alert is known to be active,
but not when it will end)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.data import KYIV_TZ

# oblast adjacency derived from the map polygons (training/build_neighbors.py)
NEIGHBORS: dict[str, list[str]] = json.loads((Path(__file__).resolve().parent / "neighbors.json").read_text())

HOUR = np.int64(3600 * 10**9)
MAX_GAP_MIN = 7 * 24 * 60  # cap for "minutes since last alert ended"

# active_now is computed but not a model input: the start model is only used when the oblast is calm.
START_FEATURES = [
    "region", "hour_kyiv", "dow_kyiv", "min_since_last_end",
    "starts_24h", "starts_7d", "n_active_oblasts", "national_starts_1h",
    "nb_active_frac", "nb_starts_1h", "prev_duration_log",
]
# Additions over the first version (region, hour, n_active_oblasts, prev_duration_min) were chosen on two
# validation windows in exploration/05_feature_ideas.py.
DURATION_FEATURES = ["region", "hour_sin", "hour_cos", "n_active_oblasts", "prev_duration_min",
                     "national_starts_1h", "starts_24h", "min_since_last_end", "nb_active_frac"]


def _ns(ts) -> np.ndarray:
    return pd.DatetimeIndex(ts).as_unit("ns").asi8


def _intervals(alerts: pd.DataFrame, region: str) -> tuple[np.ndarray, np.ndarray]:
    a = alerts[alerts.region == region].sort_values("started_at")
    starts = _ns(a.started_at)
    ends = np.where(a.finished_at.isna(), np.iinfo(np.int64).max, _ns(a.finished_at.fillna(a.started_at)))
    return starts, ends


def _active_at(starts, ends, t) -> np.ndarray:
    if len(starts) == 0:
        return np.zeros(len(t), dtype=bool)
    i = np.searchsorted(starts, t, side="left") - 1  # last alert that started strictly before t
    return (i >= 0) & (ends[np.maximum(i, 0)] > t)


def _count_between(starts, lo, hi) -> np.ndarray:
    return np.searchsorted(starts, hi, side="left") - np.searchsorted(starts, lo, side="left")


def hourly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.ceil("h"), end.floor("h"), freq="h", tz="UTC")


def start_features(alerts: pd.DataFrame, times: pd.DatetimeIndex, regions: list[str]) -> pd.DataFrame:
    """One row per (region, t) with the features known at t."""
    t = _ns(times)
    per_region = {r: _intervals(alerts, r) for r in regions}
    active = {r: _active_at(s, e, t).astype(int) for r, (s, e) in per_region.items()}
    starts_1h = {r: _count_between(s, t - HOUR, t) for r, (s, _) in per_region.items()}
    n_active = sum(active.values())
    all_starts = np.sort(_ns(alerts.started_at))
    national_1h = _count_between(all_starts, t - HOUR, t)
    local = times.tz_convert(KYIV_TZ)
    hour, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()

    frames = []
    for r, (s, e) in per_region.items():
        # ends of alerts that are already over at t (ongoing ones have end = int64 max)
        fin = e != np.iinfo(np.int64).max
        order = np.argsort(e[fin])
        finished_ends, finished_dur = e[fin][order], (e[fin] - s[fin])[order] / 6e10
        j = np.searchsorted(finished_ends, t, side="right") - 1
        last_end = finished_ends[np.maximum(j, 0)] if len(finished_ends) else np.zeros(len(t), dtype=np.int64)
        gap = np.where(j >= 0, (t - last_end) / 6e10, MAX_GAP_MIN)
        # duration of the last alert that had already ended by t (NaN if none; LightGBM handles it)
        last_dur = np.where(j >= 0, finished_dur[np.maximum(j, 0)], np.nan) if len(finished_ends) else np.full(len(t), np.nan)
        nbs = [n for n in NEIGHBORS.get(r, []) if n in per_region]  # modelled neighbours only
        frames.append(pd.DataFrame({
            "t": times,
            "region": r,
            "hour_kyiv": hour,
            "dow_kyiv": dow,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "active_now": active[r],
            "min_since_last_end": np.minimum(gap, MAX_GAP_MIN),
            "starts_24h": _count_between(s, t - 24 * HOUR, t),
            "starts_7d": _count_between(s, t - 7 * 24 * HOUR, t),
            "n_active_oblasts": n_active,
            "national_starts_1h": national_1h,
            "nb_active_frac": sum(active[n] for n in nbs) / max(len(nbs), 1),
            "nb_starts_1h": sum(starts_1h[n] for n in nbs) if nbs else np.zeros(len(t), dtype=int),
            "prev_duration_log": np.log1p(last_dur),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["region"] = pd.Categorical(out.region, categories=regions)
    return out


def start_labels(alerts: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    """1 if an alert starts in [t, t + 1h) in that region."""
    y = np.zeros(len(feats), dtype=int)
    for r, idx in feats.groupby("region", observed=True).indices.items():
        s, _ = _intervals(alerts, r)
        t = _ns(feats.t.iloc[idx])
        y[idx] = (_count_between(s, t, t + HOUR) > 0).astype(int)
    return y


def duration_features(alerts: pd.DataFrame, events: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Features for alerts in ``events`` (columns region, started_at), known at their start time."""
    t = _ns(events.started_at)
    per_region = {r: _intervals(alerts, r) for r in regions}
    active = {r: _active_at(s, e, t).astype(int) for r, (s, e) in per_region.items()}
    n_active = sum(active.values())
    national_1h = _count_between(np.sort(_ns(alerts.started_at)), t - HOUR, t)

    prev = np.full(len(events), np.nan)
    starts_24h = np.zeros(len(events), dtype=int)
    gap = np.full(len(events), float(MAX_GAP_MIN))
    nb_frac = np.zeros(len(events))
    for r, idx in events.groupby("region").indices.items():
        a = alerts[alerts.region == r].sort_values("started_at")
        s, dur = _ns(a.started_at), a.duration_min.to_numpy()
        i = np.searchsorted(s, t[idx], side="left") - 1  # previous alert, started strictly before
        ok = i >= 0
        prev[idx[ok]] = dur[i[ok]]
        starts_24h[idx] = _count_between(s, t[idx] - 24 * HOUR, t[idx])
        ends = np.sort(per_region[r][1]) if r in per_region else np.array([], dtype=np.int64)
        ends = ends[ends != np.iinfo(np.int64).max]
        j = np.searchsorted(ends, t[idx], side="right") - 1
        if len(ends):
            gap[idx] = np.where(j >= 0, np.minimum((t[idx] - ends[np.maximum(j, 0)]) / 6e10, MAX_GAP_MIN), MAX_GAP_MIN)
        nbs = [n for n in NEIGHBORS.get(r, []) if n in per_region]
        if nbs:
            nb_frac[idx] = sum(active[n][idx] for n in nbs) / len(nbs)

    hour = events.started_at.dt.tz_convert(KYIV_TZ).dt.hour.to_numpy()
    out = pd.DataFrame({
        "region": events.region.to_numpy(),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "n_active_oblasts": n_active,
        # median-ish fill when there's no previous (or it's still unknown)
        "prev_duration_min": np.nan_to_num(prev, nan=20.0),
        "national_starts_1h": national_1h,
        "starts_24h": starts_24h,
        "min_since_last_end": gap,
        "nb_active_frac": nb_frac,
    }, index=events.index)
    return out
