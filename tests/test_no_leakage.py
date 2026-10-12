"""Features at time t must be identical whether computed from the full dataset or from only
what was knowable at t (alerts started before t, ongoing alerts with unknown end)."""
import numpy as np
import pandas as pd
import pytest

from training.data import load_alerts, regions, truncate_at
from training.features import duration_features, start_features, start_labels


@pytest.fixture(scope="module")
def alerts():
    return load_alerts()


def sample_times(alerts, n=40, seed=0):
    """Random hours in the last 60 days, plus exact alert start/end instants (edge cases)."""
    rng = np.random.default_rng(seed)
    end = alerts.finished_at.max().floor("h")
    hours = pd.date_range(end - pd.Timedelta(days=60), end, freq="h")
    picks = list(rng.choice(hours, n, replace=False))
    recent = alerts[alerts.started_at > end - pd.Timedelta(days=60)].sample(10, random_state=seed)
    picks += list(recent.started_at) + list(recent.finished_at)
    return [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC") for t in picks]


def test_start_features_use_only_past(alerts):
    regs = regions(alerts)
    for t in sample_times(alerts):
        full = start_features(alerts, pd.DatetimeIndex([t]), regs)
        past = start_features(truncate_at(alerts, t), pd.DatetimeIndex([t]), regs)
        pd.testing.assert_frame_equal(full, past, check_dtype=False, obj=f"features at {t}")


def test_duration_features_use_only_past(alerts):
    regs = regions(alerts)
    events = alerts[alerts.started_at > alerts.started_at.max() - pd.Timedelta(days=30)].sample(50, random_state=1)
    for _, ev in events.iterrows():
        one = ev.to_frame().T.astype({"started_at": alerts.started_at.dtype})
        full = duration_features(alerts, one, regs)
        past = duration_features(truncate_at(alerts, ev.started_at), one, regs)
        pd.testing.assert_frame_equal(full, past, check_dtype=False, obj=f"duration features at {ev.started_at}")


def test_negative_control_labels_do_depend_on_future(alerts):
    """Sanity check that the harness can detect leakage: labels look forward, so truncation changes them."""
    regs = regions(alerts)
    times = pd.DatetimeIndex(sample_times(alerts))
    feats = start_features(alerts, times, regs)
    y_full = start_labels(alerts, feats)
    y_past = np.concatenate([
        start_labels(truncate_at(alerts, t), feats[feats.t == t]) for t in times
    ])
    order = np.concatenate([np.flatnonzero((feats.t == t).to_numpy()) for t in times])
    assert y_full[order].sum() > 0 and y_past.sum() == 0


def test_region_without_alerts(alerts):
    """A region with no history (e.g. a quiet oblast in a short window) must not crash."""
    regs = regions(alerts)
    t = alerts.started_at.max()
    subset = alerts[alerts.region == regs[0]]
    feats = start_features(subset, pd.DatetimeIndex([t]), regs)
    quiet = feats[feats.region != regs[0]]
    assert (quiet.active_now == 0).all() and (quiet.starts_7d == 0).all()
    ev = subset.tail(3)
    assert len(duration_features(subset, ev, regs)) == 3
