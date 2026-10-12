"""Feature ablation for the start model, scored on the VALIDATION window only (test stays untouched).

Usage: PYTHONPATH=. uv run python exploration/02_feature_ablation.py [--wandb]
"""
import argparse

from training import start_model
from training.data import load_alerts, regions
from training.features import START_FEATURES
from training.train import init_wandb, split_times, start_feature_frame

BASE = [f for f in START_FEATURES if f not in ("nb_active_frac", "nb_starts_1h")]
SINCOS = lambda fs: [f for f in fs if f not in ("hour_kyiv", "dow_kyiv")] + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
VARIANTS = {
    "raw_time": BASE,
    "sincos_time": SINCOS(BASE),
    "raw_time+neighbors": BASE + ["nb_active_frac", "nb_starts_1h"],
    "sincos_time+neighbors": SINCOS(BASE) + ["nb_active_frac", "nb_starts_1h"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    alerts = load_alerts()
    regs = regions(alerts)
    valid_start, test_start, grid_end = split_times(alerts)
    feats, y = start_feature_frame(alerts, regs, grid_end)
    calm = (feats.active_now == 0).to_numpy()
    tr = (feats.t < valid_start).to_numpy() & calm
    va = ((feats.t >= valid_start) & (feats.t < test_start)).to_numpy() & calm

    print(f"{'variant':24s} {'rounds':>6s} {'logloss':>8s} {'brier':>7s} {'auc':>6s} {'pr_auc':>6s}")
    for name, fs in VARIANTS.items():
        booster, curves = start_model.fit_with_validation(feats[tr], y[tr], feats[va], y[va], features=fs)
        m = start_model.metrics(y[va], start_model.predict(booster, feats[va]), "valid")
        print(f"{name:24s} {booster.best_iteration:6d} {m['valid_logloss']:8.4f} {m['valid_brier']:7.4f} "
              f"{m['valid_roc_auc']:6.4f} {m['valid_pr_auc']:6.4f}")
        if args.wandb:
            run = init_wandb(True, {"variant": name, "features": fs})
            run.tags = ("ablation",)
            for i, (a, b) in enumerate(zip(curves["train"]["binary_logloss"], curves["valid"]["binary_logloss"])):
                run.log({"boosting_round": i + 1, "start/train_logloss": a, "start/valid_logloss": b})
            run.summary.update({**m, "best_rounds": booster.best_iteration})
            run.finish()


if __name__ == "__main__":
    main()
