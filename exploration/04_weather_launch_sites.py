"""Weather at drone LAUNCH areas (national features, trailing 6h mean), validation only.

Usage: PYTHONPATH=. uv run python exploration/04_weather_launch_sites.py
Result 2026-10-12: real correlation (visibility rho=+0.27, cloud cover -0.28 vs national starts next 6h)
but valid logloss 0.4385 -> 0.4392 at the 1h horizon: redundant with recent-alert features.
"""
import json, urllib.request
from pathlib import Path
import pandas as pd
from training import start_model
from training.data import ROOT, TRAIN_START, load_alerts, regions
from training.features import START_FEATURES
from training.train import split_times, start_feature_frame

SITES = {"kursk": (51.75, 36.30), "bryansk": (53.10, 34.00), "oryol": (52.95, 36.05),
         "primorsko": (46.05, 38.17), "crimea": (45.10, 33.90)}
VARS = ["cloud_cover", "precipitation", "wind_speed_10m", "wind_gusts_10m", "visibility"]
CACHE = ROOT / "data" / "launch_wx.parquet"  # gitignored

alerts = load_alerts(); regs = regions(alerts)
valid_start, test_start, grid_end = split_times(alerts)
feats, y = start_feature_frame(alerts, regs, grid_end)
if not CACHE.exists():
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(a) for a, _ in SITES.values()) +
           "&longitude=" + ",".join(str(b) for _, b in SITES.values()) +
           f"&start_date={TRAIN_START:%Y-%m-%d}&end_date={grid_end:%Y-%m-%d}&hourly={','.join(VARS)}&timezone=UTC")
    data = json.loads(urllib.request.urlopen(url, timeout=120).read())
    pd.concat([pd.DataFrame(d["hourly"]).assign(site=s) for s, d in zip(SITES, data)]).to_parquet(CACHE)
raw = pd.read_parquet(CACHE); raw["t"] = pd.to_datetime(raw.time, utc=True)
wide = raw.pivot(index="t", columns="site", values=VARS)
# Strictly past: mean over hours [t-6h, t-1h]; values are archived *forecasts* for those hours, so known before t.
past = wide.shift(1).rolling(6, min_periods=1).mean()
agg = pd.DataFrame({f"lw_{v}_mean": past[v].mean(axis=1) for v in VARS})
agg["lw_precip_max"] = past["precipitation"].max(axis=1); agg["lw_gust_max"] = past["wind_gusts_10m"].max(axis=1)
LW = list(agg)
f2 = feats.join(agg, on="t")
print("missing:", f2[LW].isna().mean().max().round(4))
calm = (f2.active_now == 0).to_numpy()
tr = (f2.t < valid_start).to_numpy() & calm
va = ((f2.t >= valid_start) & (f2.t < test_start)).to_numpy() & calm
for name, fs in {"current": START_FEATURES, "+launch weather": START_FEATURES + LW}.items():
    b, _ = start_model.fit_with_validation(f2[tr], y[tr], f2[va], y[va], features=fs)
    m = start_model.metrics(y[va], start_model.predict(b, f2[va]), "v")
    print(f"{name:18s} {b.best_iteration:5d} logloss {m['v_logloss']:.4f} brier {m['v_brier']:.4f} auc {m['v_roc_auc']:.4f} pr {m['v_pr_auc']:.4f}", flush=True)
# Direct signal check: national starts in next 6h vs launch-area weather (daily-ish relationship)
nat = pd.Series(y, index=feats.index).groupby(feats.t).sum().rolling(6).sum().shift(-5)
d = agg.join(nat.rename("starts_next6h")).dropna()
print("spearman with national starts next 6h:\n", d.corr(method="spearman")["starts_next6h"].drop("starts_next6h").round(3).to_string())
