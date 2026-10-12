"""FastAPI backend: serves the dashboard and model outputs for replay (test period) and live "now" mode."""
import json
import time
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.regions import LABELS, REGIONS, UBILLING_TO_REGION
from training import duration_model, start_model
from training.data import MODELS_DIR, VOLUNTEER_CSV, download, load_alerts
from training.features import NEIGHBORS, duration_features, start_features

STATIC = Path(__file__).resolve().parent / "static"
LIVE_URL = "https://ubilling.net.ua/aerialalerts/"
CURVE_GRID = np.arange(1, 241)  # minutes, for survival curves

app = FastAPI(title="Air Raid Alert Forecast")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache
def artifacts() -> dict:
    missing = [f for f in ["start_model.txt", "duration_model.json", "replay_start.parquet"]
               if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"missing {missing} in models/ - run `uv run python -m training.train` first")
    replay = pd.read_parquet(MODELS_DIR / "replay_start.parquet")
    alerts = pd.read_parquet(MODELS_DIR / "replay_alerts.parquet")
    dur = duration_model.load(MODELS_DIR / "duration_model.json")
    regs = dur["regions"]
    # duration distribution parameters for every alert in the replay window (features known at its start)
    alerts = alerts[alerts.region.isin(regs)].reset_index(drop=True)
    alerts["mu"], sigma = duration_model.predict_mu_sigma(dur, duration_features(alerts, alerts, regs))
    return {
        "start": start_model.load(MODELS_DIR / "start_model.txt"),
        "dur": dur, "sigma": sigma, "regions": regs,
        "replay": replay, "alerts": alerts, "times": sorted(replay.t.unique()),
    }


def opt(x) -> float | None:
    return None if pd.isna(x) else float(x)


def duration_summary(mu: float, sigma: float, elapsed: float, started_at: pd.Timestamp) -> dict:
    family = artifacts()["dur"]["family"]
    q = lambda p: float(duration_model.quantile(mu, sigma, p, elapsed, family))
    return {
        "elapsed_min": elapsed, "median_min": q(0.5), "p10_min": q(0.1), "p90_min": q(0.9),
        "expected_end": (started_at + pd.Timedelta(minutes=q(0.5))).isoformat(),
        "curve": duration_model.survival_curve(mu, sigma, CURVE_GRID.astype(float), elapsed, family).round(4).tolist(),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    a = artifacts()
    r = a["replay"]
    national = r.groupby("t").agg(starts=("y", "sum"), active=("active_now", "sum"), p_sum=("p_start", "sum"))  # NaN skipped
    return {
        "regions": [{"region": x, "label": LABELS[x]} for x in a["regions"]],
        "times": [t.isoformat() for t in a["times"]],
        "national": {"t": [t.isoformat() for t in national.index], **{c: national[c].round(2).tolist() for c in national}},
        "curve_grid": CURVE_GRID.tolist(),
    }


@app.get("/api/replay")
def replay(t: str):
    a = artifacts()
    ts = pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
    rows = a["replay"][a["replay"].t == ts]
    if rows.empty:
        raise HTTPException(404, f"no replay data at {ts}")
    al = a["alerts"]
    active = al[(al.started_at < ts) & ((al.finished_at > ts) | al.finished_at.isna())].set_index("region")
    out = []
    for row in rows.itertuples():
        # no start forecast while under alert (p_start is NaN there) - the duration model applies instead
        item = {"region": row.region, "p_start": opt(row.p_start), "p_baseline": opt(row.p_baseline),
                "actual_start_next_hour": int(row.y), "active": bool(row.active_now)}
        if row.region in active.index:
            al_row = active.loc[row.region]
            elapsed = (ts - al_row.started_at).total_seconds() / 60
            item["duration"] = duration_summary(al_row.mu, a["sigma"], elapsed, al_row.started_at)
            item["duration"]["started_at"] = al_row.started_at.isoformat()
            # revealed ground truth - shown for evaluation only, never used as an input
            item["duration"]["actual_min"] = None if al_row.naive else float(al_row.duration_min)
        out.append(item)
    return {"t": ts.isoformat(), "mode": "replay", "regions": out}


@app.get("/api/region")
def region_history(region: str, t: str, live: bool = False):
    a = artifacts()
    ts = pd.Timestamp(t)
    r = a["replay"][a["replay"].region == region]
    al = load_alerts() if live else a["alerts"]  # replay window ends with the test period
    hist = al[(al.region == region) & (al.started_at < ts) & (al.started_at >= ts - pd.Timedelta(days=7))]
    return {
        "series": {"t": [x.isoformat() for x in r.t], "p": [opt(v) for v in r.p_start.round(4)], "y": r.y.tolist()},
        "history": [{"start": s.isoformat(), "end": (min(f, ts) if pd.notna(f) else ts).isoformat()}
                    for s, f in zip(hist.started_at, hist.finished_at)],
    }


DURATION_BINS = [0, 10, 20, 30, 60, 120, np.inf]
DURATION_LABELS = ["<10m", "10–20m", "20–30m", "30–60m", "1–2h", "2h+"]


@app.get("/api/glance")
def glance(t: str, period: str = "today", live: bool = False):
    """Situation summary for 'today' (Kyiv calendar day up to t) or the last 7 days, using only alerts known at t."""
    a = artifacts()
    end = pd.Timestamp(t)
    start = (end.tz_convert("Europe/Kyiv").normalize().tz_convert("UTC") if period == "today"
             else end - pd.Timedelta(days=7))
    al = load_alerts() if live else a["alerts"]
    al = al[al.region.isin(a["regions"]) & (al.started_at < end)]
    fin = al.finished_at.where(al.finished_at.notna() & (al.finished_at <= end), end)  # ongoing -> clipped at t
    w = al[fin > start].assign(s=lambda d: d.started_at.clip(lower=start), e=fin[fin > start])
    hours = ((w.e - w.s).dt.total_seconds() / 3600)
    started = w[w.started_at >= start]
    done = started[started.finished_at.notna() & (started.finished_at <= end) & ~started.naive.astype(bool)]

    # how many oblasts were under alert over time: count(start <= s) - count(end <= s)
    step = "10min" if period == "today" else "1h"
    grid = pd.date_range(start.ceil(step), end, freq=step)
    s_sorted, e_sorted = np.sort(w.s.values), np.sort(w.e.values)
    active = np.searchsorted(s_sorted, grid.values, "right") - np.searchsorted(e_sorted, grid.values, "right")

    by_region = (w.assign(h=hours).groupby("region").agg(hours=("h", "sum"))
                 .join(started.groupby("region").size().rename("alerts")).fillna(0)
                 .sort_values("hours", ascending=False))
    longest = (w.assign(m=(w.e - w.started_at).dt.total_seconds() / 60).sort_values("m").iloc[-1]
               if len(w) else None)
    peak_i = int(np.argmax(active)) if len(active) else None
    dur_hist = pd.cut(done.duration_min, DURATION_BINS, labels=DURATION_LABELS).value_counts().reindex(DURATION_LABELS)
    return {
        "period": period, "start": start.isoformat(), "end": end.isoformat(),
        "data_until": al.finished_at.max().isoformat() if live and len(al) else None,
        "alerts_started": int(len(started)),
        "alert_hours": round(float(hours.sum()), 1),
        "oblasts_affected": int(w.region.nunique()), "oblasts_total": len(a["regions"]),
        "median_duration_min": opt(done.duration_min.median()),
        "longest": None if longest is None else {"region": longest.region, "minutes": float(longest.m),
                                                 "ongoing": bool(pd.isna(longest.finished_at) or longest.finished_at > end)},
        "peak": None if peak_i is None else {"oblasts": int(active[peak_i]), "t": grid[peak_i].isoformat()},
        "by_region": [{"region": r, "label": LABELS[r], "hours": round(float(x.hours), 2), "alerts": int(x.alerts)}
                      for r, x in by_region.iterrows()],
        "timeline": {"t": [x.isoformat() for x in grid], "active": active.tolist()},
        "durations": [{"bin": k, "n": int(v)} for k, v in dur_hist.items()],
    }


_live_cache: dict = {}
# region -> (first time the server saw it on alert, whether it was seen calm just before = a real start time)
_live_seen: dict[str, tuple[pd.Timestamp, bool]] = {}


def live_states() -> dict[str, bool]:
    if _live_cache.get("at", 0) > time.time() - 60:
        return _live_cache["states"]
    req = Request(LIVE_URL, headers={"User-Agent": "air-raid-forecast/0.1"})
    raw = json.loads(urlopen(req, timeout=10).read())["states"]
    states = {UBILLING_TO_REGION[k]: bool(v["alertnow"]) for k, v in raw.items() if k in UBILLING_TO_REGION}
    now_ts, prev = pd.Timestamp.now(tz="UTC").floor("min"), _live_cache.get("states")
    for r, on in states.items():
        if not on:
            _live_seen.pop(r, None)
        elif r not in _live_seen:  # calm -> alert transition is only a start time if we saw it calm before
            _live_seen[r] = (now_ts, prev is not None and prev.get(r) is False)
    _live_cache.update(at=time.time(), states=states)
    return states


def merge_live(hist: pd.DataFrame, live: dict[str, bool], ts: pd.Timestamp) -> pd.DataFrame:
    """Add live alerts missing from the (lagging) daily dump as ongoing alerts, so every feature
    (active flags, neighbour share, start counts, elapsed time) agrees with the live map.
    Start time: the transition the server observed, else unknown -> ``ts`` (elapsed 0, a lower bound)."""
    ongoing = set(hist.loc[hist.finished_at.isna() | (hist.finished_at > ts), "region"])
    rows = []
    for r, on in live.items():
        if on and r not in ongoing:
            first, observed = _live_seen.get(r, (ts, False))
            rows.append({"region": r, "started_at": first if observed else ts, "naive": False, "duration_min": np.nan})
    if not rows:
        return hist
    new = pd.DataFrame(rows).astype({"started_at": hist.started_at.dtype})
    new["finished_at"] = pd.Series(pd.NaT, index=new.index, dtype=hist.finished_at.dtype)
    return pd.concat([hist, new], ignore_index=True)


@app.get("/api/now")
def now():
    a = artifacts()
    # refresh the daily dump at most once per hour
    if not VOLUNTEER_CSV.exists() or VOLUNTEER_CSV.stat().st_mtime < time.time() - 3600:
        download(force=True)
    hist = load_alerts()
    ts = pd.Timestamp.now(tz="UTC").floor("min")
    history_until = hist.finished_at.max()
    hist = hist[hist.started_at < ts]
    try:
        live = live_states()
    except Exception as e:  # live feed is optional; fall back to the dataset state
        live, live_error = None, str(e)
    else:
        live_error = None

    if live is not None:  # live state is fresher than the daily dump
        hist = merge_live(hist, live, ts)
    feats = start_features(hist, pd.DatetimeIndex([ts]), a["regions"])
    if live is not None:  # an alert that is over live but still open in the dump counts as over
        feats["active_now"] = feats.region.astype(str).map(live).fillna(feats.active_now).astype(int)
        feats["n_active_oblasts"] = int(sum(live.get(r, False) for r in a["regions"]))
        on = feats.set_index(feats.region.astype(str)).active_now
        feats["nb_active_frac"] = [np.mean([on[n] for n in nbs]) if (nbs := [n for n in NEIGHBORS.get(str(r), []) if n in on.index]) else 0.0
                                   for r in feats.region]
    p = np.where(feats.active_now == 1, np.nan, start_model.predict(a["start"], feats))
    # the alert currently running in each region, if its start is known (dump or observed live transition)
    ongoing = hist[hist.finished_at.isna() | (hist.finished_at > ts)].sort_values("started_at").groupby("region").last()

    out = []
    for (_, row), pr in zip(feats.iterrows(), p):
        r = str(row.region)
        item = {"region": r, "p_start": opt(pr), "active": bool(row.active_now)}
        if row.active_now:
            known = r in ongoing.index and (ongoing.loc[r].started_at < ts)
            started = ongoing.loc[r].started_at if known else ts  # unknown start -> elapsed 0 (median too late)
            ev = pd.DataFrame({"region": [r], "started_at": [started]})
            df = duration_features(hist[hist.started_at < started], ev, a["regions"])
            df["n_active_oblasts"] = row.n_active_oblasts
            mu, sigma = duration_model.predict_mu_sigma(a["dur"], df)
            elapsed = (ts - started).total_seconds() / 60
            item["duration"] = duration_summary(float(mu[0]), sigma, elapsed, started)
            if known:
                item["duration"]["started_at"] = started.isoformat()
        out.append(item)
    return {"t": ts.isoformat(), "mode": "now", "history_until": history_until.isoformat(),
            "live_source": None if live is None else LIVE_URL, "live_error": live_error, "regions": out}
