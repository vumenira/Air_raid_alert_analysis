"""How far back should the start model's training data go? Scored on VALIDATION only (test stays untouched).

The positive rate drifts (the test window has many more alerts than the year before), so older data may hurt.
For each training start (3/6/12/24 months before the validation window, and the previous TRAIN_START, 2025-10-01) fit the
early-stopped LightGBM and report validation log loss, with a 90% day-block interval of the difference to the
previous choice. One extra row keeps 24 months but down-weights old rows (exponential, half-life 90 days).
Both validation windows are run: the usual one and the fortnight before it (the rule from 05: adopt a change
only if it wins in both).

Result (2026-10-12): window length is noise (all |Δ| <= 0.0017, CIs mostly span 0); 24 months + half-life 90 d
won in both windows (-0.0010 [-0.0022, +0.0002] and -0.0014 [-0.0026, -0.0002]) and is now used by train.py.

Usage: PYTHONPATH=. uv run python exploration/06_train_window.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from training import start_model
from training.data import load_alerts, regions
from training.evaluation import day_ci, row_logloss
from training.features import START_FEATURES, hourly_grid, start_features, start_labels
from training.train import split_times

MONTHS = [3, 6, 12, 24]
HALF_LIFE_DAYS = 90
PREVIOUS = pd.Timestamp("2025-10-01", tz="UTC")  # TRAIN_START before this study


def fit_score(X_tr, y_tr, X_va, y_va, weight=None):
    dtr = lgb.Dataset(X_tr[START_FEATURES], y_tr, weight=weight, categorical_feature=["region"], free_raw_data=False)
    dva = lgb.Dataset(X_va[START_FEATURES], y_va, reference=dtr)
    b = lgb.train({**start_model.PARAMS, "metric": "binary_logloss"}, dtr, start_model.MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(start_model.EARLY_STOPPING, verbose=False)])
    return b.best_iteration, row_logloss(y_va, b.predict(X_va[START_FEATURES], num_iteration=b.best_iteration))


def run(feats, y, valid_start, test_start):
    calm = (feats.active_now == 0).to_numpy()
    t = feats.t
    va = ((t >= valid_start) & (t < test_start)).to_numpy() & calm
    days = t[va].dt.floor("D").to_numpy()
    starts = {f"{m} months": valid_start - pd.DateOffset(months=m) for m in MONTHS}
    starts["previous (2025-10-01)"] = PREVIOUS
    rows = {}
    for name, s in sorted(starts.items(), key=lambda kv: kv[1], reverse=True):
        tr = ((t >= s) & (t < valid_start)).to_numpy() & calm
        rows[name] = (tr.sum(), y[tr].mean(), *fit_score(feats[tr], y[tr], feats[va], y[va]))
    s = valid_start - pd.DateOffset(months=24)
    tr = ((t >= s) & (t < valid_start)).to_numpy() & calm
    age_days = (valid_start - t[tr]).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age_days / HALF_LIFE_DAYS)
    rows[f"24 months, weighted (half-life {HALF_LIFE_DAYS} d)"] = (tr.sum(), np.average(y[tr], weights=w),
                                                                 *fit_score(feats[tr], y[tr], feats[va], y[va], w))

    ref = rows["previous (2025-10-01)"][3]
    print(f"validation {valid_start.date()} -> {test_start.date()}, positive rate {y[va].mean():.3f}")
    print(f"{'train start':38s} {'rows':>8s} {'pos':>6s} {'rounds':>6s} {'logloss':>8s} {'Δ vs previous':>12s} {'90% CI (by day)':>18s}")
    for name, (n, pos, rounds, ll) in rows.items():
        lo, hi = day_ci(ll - ref, days)
        print(f"{name:38s} {n:8d} {pos:6.3f} {rounds:6d} {ll.mean():8.4f} {ll.mean() - ref.mean():+12.4f} "
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>18s}", flush=True)


def main() -> None:
    alerts = load_alerts()
    regs = regions(alerts)
    valid_start, test_start, grid_end = split_times(alerts)
    earliest = valid_start - pd.Timedelta(days=14) - pd.DateOffset(months=max(MONTHS))
    feats = start_features(alerts, hourly_grid(earliest, grid_end), regs)
    y = start_labels(alerts, feats)
    for shift in (0, 14):
        d = pd.Timedelta(days=shift)
        print()
        run(feats, y, valid_start - d, test_start - d)


if __name__ == "__main__":
    main()
