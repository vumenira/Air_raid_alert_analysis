# Technical details

The README has the short version. This file has the full protocol, all the numbers, and how to run each piece.

## Running

The project uses Python ≥ 3.12, managed by [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m training.train [--refresh] [--no-wandb]   # ~20 s: download data, train, write models/
uv run python -m app [--port 8000]                         # dashboard at http://127.0.0.1:8000
uv run pytest                                              # leakage tests (first run downloads the ~9 MB CSV)
```

- `--refresh` re-downloads the alert CSV.
- W&B logging is used if `WANDB_API_KEY` is set in `.env` (format `WANDB_API_KEY = ...`).
- Trained models are committed in `models/`, so the dashboard runs without retraining.

Studies (validation only, the test window is never touched):

```bash
PYTHONPATH=. uv run python exploration/00_time_series_eda.py      # figures -> docs/figures/
PYTHONPATH=. uv run python exploration/02_feature_ablation.py     # sin/cos vs raw time, neighbour features
PYTHONPATH=. uv run python exploration/03_weather_oblast.py       # weather at oblast centroids
PYTHONPATH=. uv run python exploration/04_weather_launch_sites.py # weather at drone launch areas
PYTHONPATH=. uv run python exploration/05_feature_ideas.py [--shift 14]  # candidate features, both models
PYTHONPATH=. uv run python exploration/06_train_window.py         # training window / recency weights
PYTHONPATH=. uv run python exploration/07_duration_family.py      # log-normal vs Weibull
```

## Layout

```
training/      data loading, leakage-safe features (features.py), both models, evaluation helpers, train.py
app/           FastAPI server + static d3 dashboard; reuses training.features for live predictions
models/        trained artifacts, metrics.json (all test metrics, calibration, training curves), replay data
tests/         leakage tests (truncation equality, negative control)
exploration/   EDA (00) and the validation studies (02–07) behind every design decision
docs/figures/  EDA figures
claude_conversation/  AI conversation logs, the mock review, and the log renderer
```

## Data

- **Alerts:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), `volunteer_data_en.csv` (eTryvoga).
  - It is at oblast level, updated daily, all times in UTC.
  - `naive=True` (~5–8% of rows) means the end time is unknown.
  - The official file in the same repo stops on 2026-09-07, so it can't be used to evaluate on the latest period.
- **Live state ("Now" mode):** `ubilling.net.ua/aerialalerts`. It says which oblasts are on alert right now, but not when each alert started.
- **Map:** geoBoundaries UKR ADM1 (ODbL). Neighbour pairs are derived from the polygons (`training/build_neighbors.py`).
- **Excluded:** Luhansk oblast, Crimea and Sevastopol have been under permanent alert since 2022. They are not recorded as alerts, so they have no model.

## Evaluation protocol

- **Split:** time-ordered.
  - Train: the start model from 2024-09-01 with recency weights (half-life 90 days); the duration model from 2025-10-01.
  - Validation: 14 days.
  - Test: the last 28 days (14 Sep – 12 Oct 2026).
  - Validation picks the number of trees (early stopping), the features, the training window and the duration family. The final models are refit on train+validation and scored on test once.
- **No leakage:** a feature at time t uses only alerts that started before t, and end times only of alerts already over by t. `data.truncate_at(alerts, t)` defines this "knowable at t" view.
  - `tests/test_no_leakage.py` recomputes the features on truncated data and asserts they are identical. Alert start and end instants are included as edge cases.
  - A negative control checks that the test detects look-ahead.
  - Injecting a 1 h leak into the feature code makes the test fail (history-1).
- **Two-window rule:** a change is adopted only if it wins on the usual validation fortnight *and* on the one before it (`--shift 14`).
- **Uncertainty:** differences are reported with a 90% bootstrap interval that resamples whole days (`training/evaluation.py`). Alerts come in waves, so hours within a day aren't independent.
  - Seed-to-seed noise of the start model: ~0.0001 log loss.
  - Day-to-day sampling noise: ~0.001–0.002.

## Start model: full test results

The test set is calm oblast-hours only (14,155 rows). All baselines are fit on the same rows with the same weights:
- **oblast × hour:** the historical start rate per (oblast, Kyiv hour), i.e. a seasonal-naive baseline;
- **recency-1:** logistic regression on `log1p(starts_7d)`;
- **recency-5:** logistic regression on oblast starts in 24 h and 7 d, time since the last alert, national starts in 1 h, oblasts on alert, plus the hour of day.

| | LightGBM | recency-5 | recency-1 | oblast × hour |
|---|---|---|---|---|
| Log loss ↓ | **0.454** | 0.462 | 0.465 | 0.511 |
| Brier ↓ | **0.150** | 0.152 | 0.153 | 0.164 |
| ROC-AUC ↑ | **0.783** | 0.769 | 0.764 | 0.739 |
| PR-AUC ↑ | **0.530** | 0.521 | 0.508 | 0.493 |
| Δ log loss (model − baseline), 90% CI | – | −0.008 [−0.011, −0.005] | −0.011 [−0.015, −0.008] | −0.057 [−0.067, −0.047] |

Model features: oblast, Kyiv hour, weekday, minutes since the last alert ended, oblast starts in 24 h and 7 d, oblasts on alert, national starts in 1 h, share of neighbours on alert, neighbour starts in 1 h, and the log duration of the last finished alert.

**Drift.** The share of calm oblast-hours followed by a start within 1 h was 13.3% in train, 22.5% in validation and 24.5% in test (`docs/figures/03_base_rate_drift.png`).
- The model's mean test prediction is 0.228, so it under-predicts a little.
- Calibration bins: predicted 0.15 → observed 0.19, and predicted 0.03 → observed 0.04.
- Mitigations: recency features, recency weights, and in production frequent refits (training takes ~20 s).

## Duration model: full test results

4,021 test alerts with a recorded end.

| | Log-normal AFT | Per-oblast median |
|---|---|---|
| 80% interval coverage | 81% | – |
| Mean log-likelihood (per alert, minutes) | −4.21 | – |
| MAE of the median ↓ | 18.8 min | 19.5 min |
| Δ MAE, 90% CI | −0.8 [−1.0, −0.6] min | – |
| C-index | 0.62 | – |

Inputs: oblast, hour (sin/cos), oblasts on alert, previous duration, national starts in 1 h, oblast starts in 24 h, time since the last alert, and the share of neighbours on alert.
- Only the coefficients are exported (`models/duration_model.json`).
- Prediction (quantiles, and survival curves conditioned on elapsed time) is closed-form numpy/scipy, checked against lifelines at train time.

## Validation studies behind each decision

| Study | Result | Decision |
|---|---|---|
| sin/cos vs raw hour/weekday (`02`) | raw slightly better for LightGBM (log loss 0.4420 vs 0.4433) | raw time features |
| Neighbour features (`02`) | log loss 0.4420 → 0.4385 | adopted |
| Weather at oblast centroids (`03`), Open-Meteo archived forecasts | no gain at the 1 h horizon | rejected |
| Weather at drone launch areas (`04`) | correlates with national activity (\|ρ\| ≈ 0.27), no log-loss gain | rejected |
| 7 start-model candidates (`05`) | only `prev_duration_log` wins, ≈ −0.004 in both windows | `prev_duration_log` adopted |
| Duration candidates (`05`) | national starts 1 h + starts 24 h + time since last + neighbours: ≈ +0.025 log-likelihood | adopted |
| Duration forecast issued 5 min after the start (`05`) | slightly worse, and the forecast comes later | rejected |
| Training window 3/6/12/24 months (`06`) | all within noise | window length doesn't matter |
| 24 months + recency weights, half-life 90 d (`06`) | −0.0010 [−0.0022, +0.0002] and −0.0014 [−0.0026, −0.0002] | adopted (small gain) |
| Log-normal vs Weibull (`07`) | Weibull: +0.04 / +0.06 log-likelihood, but median MAE +1.4 / +0.6 min (worse than the baseline) | log-normal, since the dashboard shows the median |

## Time-series analysis (`exploration/00_time_series_eda.py`)

- `01_daily_starts.png`: starts per day rose from ~50 (2022–23) to ~70 (2024–25) and ~112 (2026).
- `02_hour_weekday.png`: strong time-of-day pattern (busiest 18–19 h and 12 h Kyiv time, quietest 2–7 h); almost no weekday effect (max/min ratio 1.06).
- `03_base_rate_drift.png`: the start model's label drifts up into the test window.
- `04_acf_hourly_starts.png`: hourly national starts are autocorrelated at every lag up to a week (lag 1 h: 0.17), with peaks every 24 h. Alerts come in waves.
- `05_duration_by_year.png`: alerts are getting shorter; the median fell from 38 min (2022) to 21 min (2026).

## Dashboard modes

- **Replay:** steps through the test period with the stored forecasts, the baseline, and what actually happened.
- **Now:** re-downloads the daily dump (at most hourly) and merges the live feed into it, so every feature agrees with the live map:
  - Live alerts missing from the dump are added as ongoing alerts.
  - A calm→alert transition the server observes (it polls every 60 s while the page is open) is used as the start time.
  - If the start time is unknown (the alert was already active when the server started), elapsed = 0 is assumed, so the expected end is too late by however long the alert has already run.
