"""Does weather at the oblast centroid help the start model on VALIDATION? (test untouched)

Usage: PYTHONPATH=. uv run python exploration/03_weather_oblast.py   (Open-Meteo historical-forecast API, no key)
Result 2026-10-12: valid logloss 0.4385 -> 0.4393 with weather; no gain, weather not added to the model.
"""
import json, urllib.request
from pathlib import Path
import pandas as pd
from training import start_model
from training.data import ROOT, TRAIN_START, load_alerts, regions
from training.features import START_FEATURES
from training.train import split_times, start_feature_frame
from training.build_neighbors import vertices

CACHE = ROOT / "data" / "weather.parquet"  # gitignored
VARS = ["cloud_cover", "precipitation", "wind_speed_10m", "visibility", "temperature_2m"]

def fetch(regs, end):
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    g = json.loads((ROOT / "app/static/oblasts.geojson").read_text())
    cent = {f["properties"]["region"]: vertices(f["geometry"]).mean(0) for f in g["features"]}
    lat = ",".join(f"{cent[r][1]:.3f}" for r in regs); lon = ",".join(f"{cent[r][0]:.3f}" for r in regs)
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=" + lat + "&longitude=" + lon +
           f"&start_date={TRAIN_START:%Y-%m-%d}&end_date={end:%Y-%m-%d}&hourly={','.join(VARS)}&timezone=UTC")
    data = json.loads(urllib.request.urlopen(url, timeout=120).read())
    df = pd.concat([pd.DataFrame(d["hourly"]).assign(region=r) for r, d in zip(regs, data)])
    df["t"] = pd.to_datetime(df.time, utc=True); df = df.drop(columns="time")
    df.to_parquet(CACHE); return df

alerts = load_alerts(); regs = [str(r) for r in regions(alerts)]
valid_start, test_start, grid_end = split_times(alerts)
feats, y = start_feature_frame(alerts, regions(alerts), grid_end)
wx = fetch(regs, grid_end).rename(columns={v: "wx_" + v for v in VARS})
print("weather rows", len(wx), "missing", wx.filter(like="wx_").isna().mean().round(4).to_dict())
key = pd.MultiIndex.from_arrays([feats.region.astype(str), feats.t])
wxi = wx.set_index(["region", "t"]).reindex(key)
f2 = feats.copy()
for c in wxi: f2[c] = wxi[c].to_numpy()
calm = (f2.active_now == 0).to_numpy()
tr = (f2.t < valid_start).to_numpy() & calm
va = ((f2.t >= valid_start) & (f2.t < test_start)).to_numpy() & calm
WX = [c for c in f2 if c.startswith("wx_")]
print("unmatched weather rows in grid:", f2[WX[0]].isna().mean().round(4))
variants = {"current": START_FEATURES, "+weather(5)": START_FEATURES + WX,
            "+cloud,wind,precip": START_FEATURES + ["wx_cloud_cover", "wx_wind_speed_10m", "wx_precipitation"],
            "time,region only": ["region", "hour_kyiv", "dow_kyiv"],
            "time,region+weather": ["region", "hour_kyiv", "dow_kyiv"] + WX}
print(f"{'variant':24s} {'rounds':>6s} {'logloss':>8s} {'brier':>7s} {'auc':>6s} {'pr_auc':>6s}")
for name, fs in variants.items():
    b, _ = start_model.fit_with_validation(f2[tr], y[tr], f2[va], y[va], features=fs)
    m = start_model.metrics(y[va], start_model.predict(b, f2[va]), "v")
    print(f"{name:24s} {b.best_iteration:6d} {m['v_logloss']:8.4f} {m['v_brier']:7.4f} {m['v_roc_auc']:6.4f} {m['v_pr_auc']:6.4f}", flush=True)
    if name == "+weather(5)":
        imp = dict(zip(b.feature_name(), b.feature_importance("gain"))); tot = sum(imp.values())
        print("   gain share:", {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])})
