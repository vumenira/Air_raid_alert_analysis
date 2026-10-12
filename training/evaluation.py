"""Uncertainty for metric differences. Alerts come in waves, so rows within a day are not independent:
intervals resample whole days (day-block bootstrap)."""
import numpy as np
import pandas as pd


def day_ci(diff: np.ndarray, days: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """90% interval of mean(diff), resampling whole days."""
    df = pd.DataFrame({"d": diff, "day": days}).groupby("day").d.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(df), (n, len(df)))
    means = df["sum"].to_numpy()[k].sum(1) / df["size"].to_numpy()[k].sum(1)
    return tuple(float(q) for q in np.quantile(means, [0.05, 0.95]))


def row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def row_brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def delta(rows_a: np.ndarray, rows_b: np.ndarray, days: np.ndarray, prefix: str) -> dict:
    """mean(a - b) with its 90% day-block interval; negative = a has the lower loss."""
    lo, hi = day_ci(rows_a - rows_b, days)
    return {f"{prefix}": float(np.mean(rows_a - rows_b)), f"{prefix}_ci_lo": lo, f"{prefix}_ci_hi": hi}
