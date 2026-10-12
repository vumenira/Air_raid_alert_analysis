"""Log-normal vs Weibull for the duration model, on both validation windows (test untouched).

Log-likelihood scores the whole distribution, MAE scores the median (the dashboard's "most likely ends at").
Result (2026-10-12): Weibull wins log-likelihood in both windows (+0.043 [+0.011, +0.079], +0.064 [+0.045, +0.086])
but its median is worse (MAE +1.39 [+1.04, +1.79] and +0.60 [+0.36, +0.91] min) and doesn't beat the per-oblast
median baseline; log-normal's does. Log-normal ships (duration_model.FAMILY).

Usage: PYTHONPATH=. uv run python exploration/07_duration_family.py
"""
import numpy as np
import pandas as pd

from training import duration_model as dm
from training.data import DURATION_TRAIN_START, load_alerts, regions
from training.evaluation import day_ci
from training.features import duration_features
from training.train import split_times
alerts = load_alerts(); regs = regions(alerts)
valid_start, test_start, _ = split_times(alerts)
clean = alerts[~alerts.naive & alerts.finished_at.notna()]
for shift in (0, 14):
    vs, ts = valid_start - pd.Timedelta(days=shift), test_start - pd.Timedelta(days=shift)
    tr = clean[(clean.started_at >= DURATION_TRAIN_START) & (clean.finished_at < vs)]
    va = clean[(clean.started_at >= vs) & (clean.finished_at < ts)]
    f_tr, f_va = duration_features(alerts, tr, regs), duration_features(alerts, va, regs)
    y = va.duration_min.to_numpy(); days = va.started_at.dt.floor("D").to_numpy()
    base = np.abs(f_va.region.map(tr.groupby("region").duration_min.median()).to_numpy() - y)
    res = {}
    for fam in ("lognormal", "weibull"):
        p, _ = dm.fit(f_tr, tr.duration_min, regs, fam)
        mu, s = dm.predict_mu_sigma(p, f_va)
        ll = dm.STANDARD[fam].logpdf(np.log(np.maximum(y, .5)), mu, s) - np.log(np.maximum(y, .5))
        med = dm.quantile(mu, s, .5, family=fam)
        cov = np.mean((y >= dm.quantile(mu, s, .1, family=fam)) & (y <= dm.quantile(mu, s, .9, family=fam)))
        res[fam] = (ll, np.abs(med - y))
        print(f"shift {shift:2d} {fam:9s} loglik {ll.mean():.4f}  MAE {np.abs(med-y).mean():5.2f} (base {base.mean():5.2f})  cov80 {cov:.3f}")
    d = res["weibull"][0] - res["lognormal"][0]
    print(f"  weibull - lognormal loglik {d.mean():+.4f} CI {np.round(day_ci(d, days), 4)};"
          f" MAE diff {np.mean(res['weibull'][1]-res['lognormal'][1]):+.2f} CI {np.round(day_ci(res['weibull'][1]-res['lognormal'][1], days), 2)}")
