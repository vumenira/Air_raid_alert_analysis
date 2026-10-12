"""Candidate features for both models, scored on the VALIDATION window only (test stays untouched).

Start model: each candidate is added to START_FEATURES on its own. The 14-day validation is noisy, so the
baseline is also refit with several seeds to show how much log loss moves by chance.
Duration model: log-normal AFT, validation log-likelihood / MAE of the median / 80% interval coverage, for
  (a) a forecast issued at the alert start (as now), and
  (b) a forecast issued 5 min after the start: only alerts still active at 5 min, fit with left truncation
      at 5 min, scored conditionally on lasting > 5 min. Features may then use everything known at start+5.
All candidates use only alerts started before the issue time and end times already in the past.

Robustness: Δ columns come with a 90% bootstrap interval that resamples whole days of the validation window
(alerts come in waves, so single rows aren't independent). ``--shift 14`` moves train/validation 14 days
earlier, to check that a gain is not a property of one particular fortnight.

Usage: PYTHONPATH=. uv run python exploration/05_feature_ideas.py [--shift DAYS]
"""
import argparse

import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter
from scipy.stats import norm

from training import duration_model, start_model
from training.data import DURATION_TRAIN_START, load_alerts, regions
from training.evaluation import day_ci
from training.features import HOUR, MAX_GAP_MIN, NEIGHBORS, START_FEATURES, _active_at, _count_between, _ns, duration_features
from training.train import split_times, start_feature_frame

MIN = HOUR // 60
NEVER = np.iinfo(np.int64).max


def region_arrays(alerts: pd.DataFrame, r: str):
    a = alerts[alerts.region == r].sort_values("started_at")
    s = _ns(a.started_at)
    e = np.where(a.finished_at.isna(), NEVER, _ns(a.finished_at.fillna(a.started_at)))
    return s, e, a.duration_min.to_numpy()


def last_finished(e, dur, t, k=1):
    """Mean log1p duration of the last ``k`` alerts that had ended by t (NaN if none)."""
    fin = e != NEVER
    order = np.argsort(e[fin])
    ends, d = e[fin][order], np.log1p(dur[fin][order])
    j = np.searchsorted(ends, t, side="right")  # number of alerts ended by t
    if len(d) == 0:
        return np.full(len(t), np.nan)
    c = np.concatenate([[0], np.cumsum(d)])
    lo = np.maximum(j - k, 0)
    return np.where(j > 0, (c[j] - c[lo]) / np.maximum(j - lo, 1), np.nan)


def n_active_at(per, t):
    return sum(_active_at(s, e, t).astype(int) for s, e, _ in per.values())


# ---------------------------------------------------------------- start model
def start_candidates(alerts, feats, regs) -> pd.DataFrame:
    per = {r: region_arrays(alerts, r) for r in regs}
    t = _ns(feats.t)
    all_s = np.sort(_ns(alerts.started_at))
    ut, inv = np.unique(t, return_inverse=True)
    out = pd.DataFrame(index=feats.index)
    out["national_starts_15m"] = _count_between(all_s, t - 15 * MIN, t)
    out["national_starts_3h"] = _count_between(all_s, t - 3 * HOUR, t)
    out["d_active_1h"] = (n_active_at(per, ut) - n_active_at(per, ut - HOUR))[inv]
    for col in ["prev_duration_log", "min_since_last_start", "nb_starts_3h", "starts_same_hour_7d"]:
        out[col] = np.nan
    for r, idx in feats.groupby("region", observed=True).indices.items():
        s, e, dur = per[r]
        tt = t[idx]
        out.iloc[idx, out.columns.get_loc("prev_duration_log")] = last_finished(e, dur, tt)
        i = np.searchsorted(s, tt, side="left") - 1
        out.iloc[idx, out.columns.get_loc("min_since_last_start")] = np.where(
            i >= 0, np.minimum((tt - s[np.maximum(i, 0)]) / MIN, MAX_GAP_MIN), MAX_GAP_MIN)
        nbs = [n for n in NEIGHBORS.get(r, []) if n in per]
        out.iloc[idx, out.columns.get_loc("nb_starts_3h")] = sum(_count_between(per[n][0], tt - 3 * HOUR, tt) for n in nbs) if nbs else 0
        # same hour-of-day window on each of the previous 7 days (seasonal-naive as a feature)
        out.iloc[idx, out.columns.get_loc("starts_same_hour_7d")] = sum(
            _count_between(s, tt - k * 24 * HOUR, tt - k * 24 * HOUR + HOUR) for k in range(1, 8))
    return out


def run_start(alerts, regs, valid_start, test_start, grid_end):
    feats, y = start_feature_frame(alerts, regs, grid_end)
    feats = pd.concat([feats, start_candidates(alerts, feats, regs)], axis=1)
    calm = (feats.active_now == 0).to_numpy()
    tr = (feats.t < valid_start).to_numpy() & calm
    va = ((feats.t >= valid_start) & (feats.t < test_start)).to_numpy() & calm

    def score(fs, seed=0):
        old = start_model.PARAMS["seed"]
        start_model.PARAMS["seed"] = seed
        try:
            b, _ = start_model.fit_with_validation(feats[tr], y[tr], feats[va], y[va], features=fs)
        finally:
            start_model.PARAMS["seed"] = old
        p = np.clip(start_model.predict(b, feats[va]), 1e-6, 1 - 1e-6)
        rows = -(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
        return b.best_iteration, start_model.metrics(y[va], p, "v"), rows

    print("\n=== START MODEL (validation, calm oblast-hours) ===")
    base = [score(START_FEATURES, s)[1]["v_logloss"] for s in range(5)]
    _, _, ref_rows = score(START_FEATURES)
    b_ll = ref_rows.mean()
    days = feats.t[va].dt.floor("D").to_numpy()
    print(f"baseline log loss over 5 seeds: {np.round(base, 4)}  (sd {np.std(base):.4f})")
    print(f"{'variant':34s} {'rounds':>6s} {'logloss':>8s} {'Δ':>8s} {'90% CI (by day)':>18s} {'auc':>6s} {'pr_auc':>6s}")
    cands = ["national_starts_15m", "national_starts_3h", "d_active_1h", "prev_duration_log",
             "min_since_last_start", "nb_starts_3h", "starts_same_hour_7d"]
    variants = [("current", START_FEATURES)] + [(f"+{c}", START_FEATURES + [c]) for c in cands]
    variants += [("+prev_duration_log+nb_starts_3h", START_FEATURES + ["prev_duration_log", "nb_starts_3h"]),
                 ("+prev_duration_log+min_since_last_start", START_FEATURES + ["prev_duration_log", "min_since_last_start"])]
    for name, fs in variants:
        n, m, rows = score(fs)
        lo, hi = day_ci(rows - ref_rows, days)
        print(f"{name:34s} {n:6d} {m['v_logloss']:8.4f} {m['v_logloss'] - b_ll:+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>18s} "
              f"{m['v_roc_auc']:6.4f} {m['v_pr_auc']:6.4f}", flush=True)


# ---------------------------------------------------------------- duration model
def duration_candidates(alerts, events, regs, offset_min: float) -> pd.DataFrame:
    """Candidates known at T = start + offset (the alert itself counts as active when offset > 0)."""
    per = {r: region_arrays(alerts, r) for r in regs}
    t0 = _ns(events.started_at)
    T = t0 + int(offset_min * MIN)
    all_s = np.sort(_ns(alerts.started_at))
    fin = alerts[alerts.finished_at.notna()]
    fin_e, fin_d = _ns(fin.finished_at), np.log1p(fin.duration_min.to_numpy())
    order = np.argsort(fin_e)
    fin_e, c = fin_e[order], np.concatenate([[0], np.cumsum(fin_d[order])])
    out = pd.DataFrame(index=events.index)
    out["n_active_T"] = n_active_at(per, T)
    out["wave_starts"] = np.log1p(_count_between(all_s, t0 - 15 * MIN, T))  # starts nationwide, 15 min before .. T
    out["national_starts_1h"] = np.log1p(_count_between(all_s, T - HOUR, T))
    lo, hi = np.searchsorted(fin_e, T - 3 * HOUR, side="right"), np.searchsorted(fin_e, T, side="right")
    out["nat_recent_logdur"] = np.where(hi > lo, (c[hi] - c[lo]) / np.maximum(hi - lo, 1), np.log1p(20))
    for col in ["nb_active_frac", "starts_24h", "gap_log", "prev_dur_mean5"]:
        out[col] = np.nan
    act = {r: _active_at(s, e, T).astype(float) for r, (s, e, _) in per.items()}
    for r, idx in events.groupby("region").indices.items():
        s, e, dur = per[r]
        nbs = [n for n in NEIGHBORS.get(r, []) if n in per]
        out.iloc[idx, out.columns.get_loc("nb_active_frac")] = (sum(act[n][idx] for n in nbs) / len(nbs)) if nbs else 0
        out.iloc[idx, out.columns.get_loc("starts_24h")] = np.log1p(_count_between(s, t0[idx] - 24 * HOUR, t0[idx]))
        fin_r = np.sort(e[e != NEVER])
        j = np.searchsorted(fin_r, t0[idx], side="right") - 1
        gap = np.where(j >= 0, (t0[idx] - fin_r[np.maximum(j, 0)]) / MIN, MAX_GAP_MIN) if len(fin_r) else MAX_GAP_MIN
        out.iloc[idx, out.columns.get_loc("gap_log")] = np.log1p(np.minimum(gap, MAX_GAP_MIN))
        out.iloc[idx, out.columns.get_loc("prev_dur_mean5")] = last_finished(e, dur, t0[idx], k=5)
    return out.fillna({"prev_dur_mean5": np.log1p(20)})


def fit_score(X_tr, d_tr, X_va, d_va, entry: float):
    df = X_tr.assign(duration=d_tr, event=1)
    kw = {}
    if entry > 0:
        df["entry"] = entry
        kw["entry_col"] = "entry"
    aft = LogNormalAFTFitter(penalizer=0.01).fit(df, duration_col="duration", event_col="event", **kw)
    p = aft.params_.loc["mu_"]
    mu = p["Intercept"] + X_va[p.drop("Intercept").index].to_numpy() @ p.drop("Intercept").to_numpy()
    sigma = float(np.exp(aft.params_.loc[("sigma_", "Intercept")]))
    y = d_va
    ly = np.log(y)
    # log-likelihood of the duration in minutes, conditional on lasting > entry
    ll = norm.logpdf(ly, mu, sigma) - ly - (norm.logsf((np.log(entry) - mu) / sigma) if entry > 0 else 0)
    med = duration_model.quantile(mu, sigma, 0.5, entry)
    lo, hi = duration_model.quantile(mu, sigma, 0.1, entry), duration_model.quantile(mu, sigma, 0.9, entry)
    return {"ll": ll, "loglik": float(np.mean(ll)), "mae": float(np.mean(np.abs(med - y))),
            "medae": float(np.median(np.abs(med - y))), "cov80": float(np.mean((y >= lo) & (y <= hi))), "sigma": sigma}


def run_duration(alerts, regs, valid_start, test_start):
    clean = alerts[~alerts.naive & alerts.finished_at.notna()]
    ev_tr = clean[(clean.started_at >= DURATION_TRAIN_START) & (clean.finished_at < valid_start)]
    ev_va = clean[(clean.started_at >= valid_start) & (clean.finished_at < test_start)]
    cands = ["nb_active_frac", "wave_starts", "national_starts_1h", "starts_24h", "gap_log", "prev_dur_mean5", "nat_recent_logdur"]
    combos = [("+nat1h+starts24h+gap", ["national_starts_1h", "starts_24h", "gap_log"]),
              ("+nat1h+starts24h+gap+nb_active", ["national_starts_1h", "starts_24h", "gap_log", "nb_active_frac"]),
              ("+nat1h+gap", ["national_starts_1h", "gap_log"])]
    for offset in (0, 5):
        tr, va = (ev_tr, ev_va) if offset == 0 else (ev_tr[ev_tr.duration_min > offset], ev_va[ev_va.duration_min > offset])
        base_tr = duration_model.design_matrix(duration_features(alerts, tr, regs), regs)
        base_va = duration_model.design_matrix(duration_features(alerts, va, regs), regs)
        c_tr, c_va = duration_candidates(alerts, tr, regs, offset), duration_candidates(alerts, va, regs, offset)
        d_tr, d_va = tr.duration_min.clip(lower=0.5).to_numpy(), va.duration_min.clip(lower=0.5).to_numpy()
        print(f"\n=== DURATION, forecast issued at start + {offset} min "
              f"({len(tr)} train / {len(va)} valid alerts{', scored given > 5 min' if offset else ''}) ===")
        print(f"{'variant':34s} {'loglik':>8s} {'Δ':>7s} {'90% CI (by day)':>18s} {'MAE':>6s} {'MedAE':>6s} {'cov80':>6s} {'sigma':>6s}")
        days = va.started_at.dt.floor("D").to_numpy()
        variants = [("current", [])]
        if offset:
            variants.append(("n_active @+5 (replaces)", ["n_active_T"]))
        variants += [(f"+{c}", [c]) for c in cands] + combos
        ref = None
        for name, extra in variants:
            X_tr, X_va = base_tr.copy(), base_va.copy()
            if "n_active_T" in extra:
                X_tr["n_active_oblasts"], X_va["n_active_oblasts"] = c_tr.n_active_T.to_numpy(), c_va.n_active_T.to_numpy()
                extra = []
            for c in extra:
                X_tr[c], X_va[c] = c_tr[c].to_numpy(), c_va[c].to_numpy()
            m = fit_score(X_tr, d_tr, X_va, d_va, offset)
            ref = m["ll"] if ref is None else ref
            lo, hi = day_ci(m["ll"] - ref, days)
            print(f"{name:34s} {m['loglik']:8.4f} {m['loglik'] - ref.mean():+7.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>18s} {m['mae']:6.1f} {m['medae']:6.1f} {m['cov80']:6.1%} {m['sigma']:6.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=int, default=0, help="move the validation window this many days earlier")
    args = ap.parse_args()
    alerts = load_alerts()
    regs = regions(alerts)
    valid_start, test_start, grid_end = split_times(alerts)
    if args.shift:  # validation = the fortnight before the usual one; grid ends where it ends (labels inside)
        shift = pd.Timedelta(days=args.shift)
        valid_start, test_start, grid_end = valid_start - shift, test_start - shift, test_start - shift
    print(f"validation window {valid_start} -> {test_start}")
    run_duration(alerts, regs, valid_start, test_start)
    run_start(alerts, regs, valid_start, test_start, grid_end)


if __name__ == "__main__":
    main()
