"""Model 1: probability that an alert starts in a region within the next hour (LightGBM)."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from training.features import START_FEATURES

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "verbose": -1,
    "seed": 0,
    "num_threads": 4,  # physical cores; 8 hyper-threads oversubscribe a busy 4-core laptop badly
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100  # stop when validation log loss hasn't improved for this many rounds
# Rows are down-weighted by age (relative to the end of the fitting data): the alert rate drifts, so recent
# months matter more. Chosen on two validation windows in exploration/06_train_window.py.
HALF_LIFE_DAYS = 90


def recency_weights(t: pd.Series, ref: pd.Timestamp, half_life_days: float = HALF_LIFE_DAYS) -> np.ndarray:
    age_days = (ref - t).dt.total_seconds().to_numpy() / 86400
    return 0.5 ** (age_days / half_life_days)


def _dataset(X: pd.DataFrame, y: np.ndarray, features: list[str], reference=None, weight=None) -> lgb.Dataset:
    cat = ["region"] if "region" in features else []
    return lgb.Dataset(X[features], y, weight=weight, categorical_feature=cat, reference=reference, free_raw_data=False)


def fit_with_validation(X_tr, y_tr, X_va, y_va, features=START_FEATURES, weight=None) -> tuple[lgb.Booster, dict]:
    """Early-stopped fit; returns the booster and per-round train/valid curves (log loss, AUC)."""
    curves: dict = {}
    dtr = _dataset(X_tr, y_tr, features, weight=weight)
    dva = _dataset(X_va, y_va, features, reference=dtr)
    booster = lgb.train(
        {**PARAMS, "metric": ["binary_logloss", "auc"]}, dtr, MAX_ROUNDS,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True, verbose=False),
                   lgb.record_evaluation(curves)],
    )
    return booster, curves


def fit(X: pd.DataFrame, y: np.ndarray, rounds: int, features=START_FEATURES, weight=None) -> lgb.Booster:
    """Final refit with a fixed number of rounds (chosen on validation)."""
    return lgb.train(PARAMS, _dataset(X, y, features, weight=weight), rounds)


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    # use the features the model was trained with; best_iteration is 0 (= all trees) for refit models
    return model.predict(X[model.feature_name()], num_iteration=model.best_iteration or None)


def baseline_table(X: pd.DataFrame, y: np.ndarray, weight=None) -> pd.Series:
    """Historical rate per (region, Kyiv hour) on the training set (optionally recency-weighted)."""
    w = pd.Series(np.ones(len(y)) if weight is None else weight, index=X.index)
    keys = [X.region, X.hour_kyiv]
    return (w * y).groupby(keys, observed=True).sum() / w.groupby(keys, observed=True).sum()


def baseline_predict(table: pd.Series, X: pd.DataFrame) -> np.ndarray:
    idx = pd.MultiIndex.from_arrays([X.region.astype(str), X.hour_kyiv])
    table = table.copy()
    table.index = table.index.set_levels(table.index.levels[0].astype(str), level=0)
    return table.reindex(idx).fillna(table.mean()).to_numpy()


# Recency baselines: logistic regression on "how active has this oblast / the country been lately".
# The (oblast, hour) rate above is learned over the whole training year and goes stale when the base rate drifts.
RECENCY1 = ["starts_7d"]
RECENCY5 = ["starts_24h", "starts_7d", "min_since_last_end", "national_starts_1h", "n_active_oblasts"]


def _recency_X(X: pd.DataFrame, cols: list[str], with_hour: bool) -> np.ndarray:
    num = np.log1p(X[cols].to_numpy(dtype=float))
    if not with_hour:
        return num
    return np.hstack([num, np.eye(24)[X.hour_kyiv.to_numpy()]])


def recency_baseline_fit(X: pd.DataFrame, y: np.ndarray, cols: list[str], with_hour: bool = False, weight=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(_recency_X(X, cols, with_hour), y, logisticregression__sample_weight=weight), cols, with_hour


def recency_baseline_predict(fitted, X: pd.DataFrame) -> np.ndarray:
    model, cols, with_hour = fitted
    return model.predict_proba(_recency_X(X, cols, with_hour))[:, 1]


def metrics(y: np.ndarray, p: np.ndarray, prefix: str) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        f"{prefix}_brier": brier_score_loss(y, p),
        f"{prefix}_logloss": log_loss(y, p),
        f"{prefix}_roc_auc": roc_auc_score(y, p),
        f"{prefix}_pr_auc": average_precision_score(y, p),
    }


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
    return df.reset_index()


def save(model: lgb.Booster, path: Path) -> None:
    model.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
