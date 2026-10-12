# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Application pet-project for an AI agentic school: **time series analysis/forecasting of air raid alerts in Ukraine**, in Python.
- Deadline: **Mon 12 Oct 2026, 23:59 Kyiv time (EEST, UTC+3)**.
- Deliverables: GitHub repo with working code, full AI conversation log, ~100-word reflection (what went wrong / how the approach changed / why the final version is better).
- Reviewers grade the *process* (the AI log) as much as the result: iterate, verify claims against data, correct errors explicitly.

## Hard rules

- **No data leakage.** Every feature used for a forecast made at time `t` must be computable from data with timestamp `< t` (alert *start* times, and *end* times only for alerts already finished by `t`; text only if *published* before `t`). Use time-ordered splits, never random splits.
- **Evaluate on the most recent period** available (the last weeks of data), against naive baselines (seasonal-naive / historical rate).

## Data sources (verified 2026-10-12)

Primary: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), raw CSVs under `datasets/`. All times are UTC.
- `official_data_en.csv`: columns `oblast, raion, hromada, level, started_at, finished_at, source`. Starts 2022-03-15. **Last record is 2026-09-07 05:28 UTC.** The repo still gets a commit every day, but this file hasn't grown since then. The official Telegram channel `t.me/s/air_alert_ua`, which feeds it, also stops posting at 2026-09-07 06:09 UTC.
- `volunteer_data_en.csv` (eTryvoga): columns `region, started_at, finished_at, naive`. Oblast level only, 25 regions, starts 2022-02-25, **current to the present day**. `naive=True` (~5%) means the end time is unknown and was set to start + 30 min.
- **Structural break:** official alerts moved from oblast level to **raion level** (118 raions) during 2025 (raion share rises from Q2 2025; oblast-level is near zero from Q1 2026). Counts per oblast are not comparable across this break unless raion alerts are merged into oblast-level "any alert active" intervals.
- Permanent alerts (Luhansk since 2022-04-04, Crimea since 2022-12-10) are not recorded as alerts. The volunteer file still has `Luhanska oblast` rows from before April 2022, so it lists 25 regions; we model 24.

Secondary:
- `t.me/s/kpszsu` (Ukrainian Air Force, live): public web preview of threat posts (drones, missiles, directions). Scrapeable without auth; each post has a `datetime` that can be used for leakage-safe text features.
- alerts.in.ua API: requires a token (HTTP 401 without one).

## Architecture

- `training/` is the single source of truth for features. `features.py` holds the leakage-safe feature code; it works on alert intervals where `finished_at = NaT` means "still active". `data.truncate_at(alerts, t)` produces exactly that "knowable at t" view. `train.py` fits both models and writes everything to `models/`.
- **Model 1** (`start_model.py`): LightGBM on an hourly (oblast × hour) grid, **calm oblast-hours only** (while an alert is active the UI shows the duration model's expected end instead). Label: an alert starts in `[t, t+1h)`. The split is train → 14-day validation (early stopping; per-round curves go to W&B and to `metrics.json["curves"]`) → 28-day test. The final model is refit on train+valid with the best round count. Training starts at `data.TRAIN_START` (2024-09-01) with exponential recency weights (`start_model.HALF_LIFE_DAYS = 90`, age measured from the end of the fitting data): window length was noise, the weighting won in both validation windows (`exploration/06_train_window.py`). The positive rate drifts (13% train, 22.5% valid, 24.5% test).
- Start-model baselines in `train.py`: the (oblast, hour) rate, and logistic regressions on recency features (`start_model.RECENCY1/RECENCY5`). All use the same weights. Test deltas get day-block bootstrap CIs (`training/evaluation.py`). The honest margin over recency-5 is ~0.008 log loss.
- Neighbour features use `training/neighbors.json`, generated from the map polygons by `training/build_neighbors.py`. Odesa↔Kherson is an edge across the estuary, not a land border.
- Raw `hour_kyiv`/`dow_kyiv` beat sin/cos encoding for LightGBM on validation (see `exploration/02_feature_ablation.py`); the sin/cos columns exist but aren't model inputs. `num_threads=4`: 8 threads crawl on this 4-core laptop when the CPU is busy.
- Weather was tested and rejected for the start model (validation only, 2026-10-12): Open-Meteo *historical-forecast* API (no key, leakage-safe because it stores archived forecasts) at oblast centroids (`exploration/03_weather_oblast.py`) and at drone launch areas (`exploration/04_weather_launch_sites.py`). Launch-area visibility/cloud correlate with national activity (|ρ|≈0.27), but log loss doesn't improve at the 1 h horizon. Revisit only for longer horizons (6–24 h).
- Feature candidates for both models are scored by `exploration/05_feature_ideas.py` (validation only; `--shift 14` re-runs on the previous fortnight; Δ comes with a day-block bootstrap CI). Adopt a feature only if it wins in both windows. Seed-to-seed noise of the start model is ~0.0001 log loss, but day-to-day sampling noise is ~0.001–0.002.
- **Model 2** (`duration_model.py`): lifelines log-normal AFT, trained from `data.DURATION_TRAIN_START` (2025-10-01, the window its features were selected on). Weibull is supported through the same location-scale form (ln T = mu + sigma·W, W ~ normal or `gumbel_l`). It wins validation log-likelihood but loses on median MAE in both windows (`exploration/07_duration_family.py`), so `FAMILY = "lognormal"`. Inputs: region, hour, n_active_oblasts, previous duration, national starts 1 h, oblast starts 24 h, time since the last alert, neighbours on alert (log1p transforms in `design_matrix`). Only its coefficients are exported to `models/duration_model.json`, and prediction (quantiles and survival curves conditioned on elapsed time) is closed-form numpy/scipy, checked against lifelines at train time.
- `app/server.py` (FastAPI) imports `training.features` so live predictions use the same feature code. Replay mode reads `models/replay_*.parquet`. Now mode re-downloads the CSV (at most hourly). `merge_live` adds live alerts missing from the dump as ongoing alerts, so all features match the live map. The start time is the calm→alert transition observed by the 60 s poll (`_live_seen`), otherwise it's unknown (elapsed 0).
- `app/static/` is a no-build dashboard (d3 v7 from cdnjs). The map uses a planar `geoTransform` (x·cos 48.5°), because geoBoundaries polygon winding breaks d3's spherical projections.
- Region names are mapped dataset ↔ geoBoundaries ↔ ubilling in `app/regions.py`. Luhansk is excluded from modelling (`data.EXCLUDED_REGIONS`); it has only pre-April-2022 rows.

## Commands

The venv uses Python 3.13 (`requires-python >= 3.12`), managed by `uv`.
- Train: `uv run python -m training.train [--refresh] [--no-wandb]` (~20 s). W&B reads `WANDB_API_KEY` from `.env` via python-dotenv. The file uses `KEY = value` with spaces, so don't `source` it.
- Dashboard: `uv run python -m app [--port 8000]`. The bottom row has Today / Last 7 days tabs (`/api/glance`, computed only from alerts known at the selected time). Model metrics live in W&B and `models/metrics.json`, not in the UI.
- Feature ablation (validation only): `PYTHONPATH=. uv run python exploration/02_feature_ablation.py [--wandb]`
- EDA figures → `docs/figures/`: `PYTHONPATH=. uv run python exploration/00_time_series_eda.py`
- Render a raw session log with tool outputs: `python claude_conversation/render_jsonl.py ~/.claude/projects/<proj>/<id>.jsonl -o out.txt`
- Tests: `uv run pytest`; single test: `uv run pytest tests/test_no_leakage.py::test_start_features_use_only_past`
- Rebuild map file: `uv run python -m app.prepare_geojson`
- `data/` (raw CSV) and `wandb/` are gitignored; `models/` is committed.
