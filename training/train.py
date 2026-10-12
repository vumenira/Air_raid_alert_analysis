"""Train both models on a time-ordered split and save artifacts.

    train [TRAIN_START, valid_start) -> valid [valid_start, test_start) -> test [test_start, end]

Validation picks the number of boosting rounds (early stopping) and the duration distribution family;
the final models are refit on train+valid and evaluated once on the test window.

Usage: uv run python -m training.train [--refresh] [--no-wandb]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from training import duration_model, evaluation, start_model
from training.data import DURATION_TRAIN_START, MODELS_DIR, ROOT, TEST_DAYS, TRAIN_START, download, load_alerts, regions
from training.features import START_FEATURES, duration_features, hourly_grid, start_features, start_labels

VALID_DAYS = 14


def split_times(alerts: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    # Last hour whose 1h label window is fully covered by the data snapshot.
    grid_end = alerts.finished_at.max().floor("h") - pd.Timedelta(hours=1)
    test_start = grid_end - pd.Timedelta(days=TEST_DAYS)
    return test_start - pd.Timedelta(days=VALID_DAYS), test_start, grid_end


def init_wandb(enabled: bool, config: dict):
    load_dotenv(ROOT / ".env")
    if not enabled or not os.getenv("WANDB_API_KEY"):
        print("W&B logging disabled")
        return None
    import wandb
    run = wandb.init(project="air-raid-alerts", config=config, job_type="train")
    run.define_metric("boosting_round")
    run.define_metric("start/*", step_metric="boosting_round")
    return run


def start_feature_frame(alerts, regs, grid_end):
    feats = start_features(alerts, hourly_grid(TRAIN_START, grid_end), regs)
    return feats, start_labels(alerts, feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the dataset")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    download(force=args.refresh)
    alerts = load_alerts()
    regs = regions(alerts)
    valid_start, test_start, grid_end = split_times(alerts)
    print(f"data until {alerts.finished_at.max()}\n  train {TRAIN_START} -> {valid_start}"
          f"\n  valid -> {test_start}\n  test  -> {grid_end}")
    MODELS_DIR.mkdir(exist_ok=True)

    run = init_wandb(not args.no_wandb, {
        "train_start": str(TRAIN_START), "valid_start": str(valid_start), "test_start": str(test_start),
        "test_end": str(grid_end), "valid_days": VALID_DAYS, "test_days": TEST_DAYS, "n_regions": len(regs),
        "start_params": start_model.PARAMS, "start_half_life_days": start_model.HALF_LIFE_DAYS, "start_max_rounds": start_model.MAX_ROUNDS,
        "start_early_stopping": start_model.EARLY_STOPPING, "start_features": START_FEATURES,
    })

    # ---- Model 1: alert start within the next hour (calm oblast-hours only) ----
    t0 = time.time()
    feats, y = start_feature_frame(alerts, regs, grid_end)
    calm = (feats.active_now == 0).to_numpy()
    t = feats.t
    tr = ((t < valid_start).to_numpy()) & calm
    va = ((t >= valid_start) & (t < test_start)).to_numpy() & calm
    pre = ((t < test_start).to_numpy()) & calm
    te = ((t >= test_start).to_numpy()) & calm

    w_tr = start_model.recency_weights(t[tr], valid_start)
    w_pre = start_model.recency_weights(t[pre], test_start)
    booster, curves = start_model.fit_with_validation(feats[tr], y[tr], feats[va], y[va], weight=w_tr)
    best_rounds = booster.best_iteration
    p_va = start_model.predict(booster, feats[va])
    model = start_model.fit(feats[pre], y[pre], best_rounds, weight=w_pre)  # refit on train+valid
    p = start_model.predict(model, feats[te])
    table = start_model.baseline_table(feats[pre], y[pre], w_pre)
    p_base = start_model.baseline_predict(table, feats[te])
    baselines = {  # all fit on train+valid calm hours with the same recency weights, like the final model
        "start_baseline": p_base,
        "start_recency1": start_model.recency_baseline_predict(
            start_model.recency_baseline_fit(feats[pre], y[pre], start_model.RECENCY1, weight=w_pre), feats[te]),
        "start_recency5": start_model.recency_baseline_predict(
            start_model.recency_baseline_fit(feats[pre], y[pre], start_model.RECENCY5, with_hour=True, weight=w_pre),
            feats[te]),
    }
    m1 = {**start_model.metrics(y[te], p, "start"), **start_model.metrics(y[va], p_va, "start_valid"),
          "start_best_rounds": best_rounds, "start_train_rows": int(pre.sum()), "start_test_rows": int(te.sum()),
          # drift: the positive rate per split, and whether the model's mean prediction follows it on test
          "start_train_pos_rate": float(y[tr].mean()), "start_valid_pos_rate": float(y[va].mean()),
          "start_test_pos_rate": float(y[te].mean()), "start_test_mean_pred": float(p.mean()),
          "start_fit_seconds": time.time() - t0}
    te_days = t[te].dt.floor("D").to_numpy()
    for name, pb in baselines.items():
        m1.update(start_model.metrics(y[te], pb, name))
        m1[f"{name}_test_mean_pred"] = float(pb.mean())
        # model minus baseline per row: negative = the model is better; 90% CI resamples whole test days
        m1.update(evaluation.delta(evaluation.row_logloss(y[te], p), evaluation.row_logloss(y[te], pb), te_days,
                                   f"start_d_logloss_vs_{name.removeprefix('start_')}"))
        m1.update(evaluation.delta(evaluation.row_brier(y[te], p), evaluation.row_brier(y[te], pb), te_days,
                                   f"start_d_brier_vs_{name.removeprefix('start_')}"))
    calib = start_model.calibration(y[te], p)
    start_model.save(model, MODELS_DIR / "start_model.txt")

    # ---- Model 2: alert duration distribution ----
    t0 = time.time()
    clean = alerts[~alerts.naive & alerts.finished_at.notna()]
    ev_tr = clean[(clean.started_at >= DURATION_TRAIN_START) & (clean.finished_at < valid_start)]
    ev_va = clean[(clean.started_at >= valid_start) & (clean.finished_at < test_start)]
    ev_pre = clean[(clean.started_at >= DURATION_TRAIN_START) & (clean.finished_at < test_start)]
    ev_te = clean[(clean.started_at >= test_start) & (clean.started_at <= grid_end)]
    families = duration_model.compare_families(
        duration_features(alerts, ev_tr, regs), ev_tr.duration_min,
        duration_features(alerts, ev_va, regs), ev_va.duration_min, regs)
    # family fixed by the two-window study (median MAE vs likelihood trade-off); likelihoods logged for reference
    family = duration_model.FAMILY
    print(f"  duration family: {family} (validation log-likelihood {families})")
    params, aft = duration_model.fit(duration_features(alerts, ev_pre, regs), ev_pre.duration_min, regs, family)
    te_feats = duration_features(alerts, ev_te, regs)
    baseline_median = ev_pre.groupby("region").duration_min.median()
    m2 = duration_model.evaluate(params, te_feats, ev_te.duration_min, baseline_median)
    # MAE gain over the per-oblast median, with a day-block CI (the point gain alone is under a minute)
    mu_te, sigma_te = duration_model.predict_mu_sigma(params, te_feats)
    err_model = np.abs(duration_model.quantile(mu_te, sigma_te, 0.5, family=family) - ev_te.duration_min.to_numpy())
    err_base = np.abs(te_feats.region.map(baseline_median).to_numpy() - ev_te.duration_min.to_numpy())
    m2.update(evaluation.delta(err_model, err_base, ev_te.started_at.dt.floor("D").to_numpy(), "dur_d_mae_vs_baseline"))
    # alerts with an unknown end (naive=True) are excluded from training and test
    all_te = alerts[(alerts.started_at >= test_start) & (alerts.started_at <= grid_end)]
    m2["dur_test_naive_share"] = float(all_te.naive.mean())
    # sanity check: our closed-form median must match lifelines' own prediction
    X_te = duration_model.design_matrix(te_feats, regs)
    med = duration_model.quantile(mu_te, sigma_te, 0.5, family=family)
    gap = np.max(np.abs(med - aft.predict_median(X_te).to_numpy()))
    assert gap < 1e-6 * med.max(), f"median mismatch vs lifelines: {gap}"
    m2.update({f"dur_valid_loglik_{k}": v for k, v in families.items()})
    m2["dur_family"] = family
    m2.update({"dur_train_events": len(ev_pre), "dur_test_events": len(ev_te), "dur_fit_seconds": time.time() - t0})
    duration_model.save(params, MODELS_DIR / "duration_model.json")

    # ---- Artifacts for the dashboard replay ----
    # replay covers all test hours; p_start is NaN while the oblast is under alert
    test_all = (t >= test_start).to_numpy()
    replay = feats[test_all][["t", "region", "active_now", "n_active_oblasts"]].assign(y=y[test_all])
    replay["p_start"], replay["p_baseline"] = np.nan, np.nan
    replay.loc[replay.active_now == 0, "p_start"] = p
    replay.loc[replay.active_now == 0, "p_baseline"] = p_base
    replay["region"] = replay.region.astype(str)
    replay.to_parquet(MODELS_DIR / "replay_start.parquet", index=False)
    lookback = test_start - pd.Timedelta(days=7)  # history panel needs a week before the test window
    alerts[alerts.started_at >= lookback].to_parquet(MODELS_DIR / "replay_alerts.parquet", index=False)
    importance = dict(zip(model.feature_name(), model.feature_importance("gain").round(1).tolist()))
    curve_rows = [{"round": i + 1, **{f"{split}_{m}": curves[split][m][i] for split in curves for m in curves[split]}}
                  for i in range(len(curves["train"]["binary_logloss"]))]

    metrics = {**m1, **m2, "valid_start": str(valid_start), "test_start": str(test_start),
               "test_end": str(grid_end), "data_until": str(alerts.finished_at.max())}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(
        {"metrics": metrics, "calibration": calib.to_dict("records"), "importance": importance,
         "curves": curve_rows}, indent=1))

    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")

    if run:
        import wandb
        for row in curve_rows:  # one step per boosting round -> real training curves
            run.log({"boosting_round": row["round"],
                     "start/train_logloss": row["train_binary_logloss"], "start/valid_logloss": row["valid_binary_logloss"],
                     "start/train_auc": row["train_auc"], "start/valid_auc": row["valid_auc"]})
        run.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        run.log({
            "start_calibration": wandb.plot.line(wandb.Table(dataframe=calib), "mean_pred", "frac_pos",
                                                 title="Calibration (test)"),
            "start_importance": wandb.plot.bar(wandb.Table(data=list(importance.items()), columns=["feature", "gain"]),
                                               "feature", "gain", title="Feature importance (gain)"),
            "duration_families": wandb.plot.bar(wandb.Table(data=list(families.items()), columns=["family", "valid_loglik"]),
                                                "family", "valid_loglik", title="Duration family: validation log-likelihood"),
        })
        run.finish()


if __name__ == "__main__":
    main()
