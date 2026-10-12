"""Model 2: distribution of alert duration (accelerated failure time model, log-normal or Weibull).

Both families are location-scale models for the log duration: ln T = mu(x) + sigma * W, where W is standard
normal (log-normal) or standard minimum-Gumbel (Weibull with lambda = exp(mu), rho = 1 / sigma). So
S(t) = sf_W((ln t - mu) / sigma). The family is picked on validation log-likelihood in train.py.
The fitted coefficients are exported to JSON; prediction below needs only numpy/scipy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from scipy.stats import gumbel_l, norm

NUMERIC = ["hour_sin", "hour_cos", "n_active_oblasts", "log_prev_duration",
           "log_national_starts_1h", "log_starts_24h", "log_min_since_last_end", "nb_active_frac"]


def design_matrix(feats: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=feats.index)
    for r in regions[1:]:  # first region is the reference level
        X[f"region={r}"] = (feats.region == r).astype(float)
    X["hour_sin"], X["hour_cos"] = feats.hour_sin, feats.hour_cos
    X["n_active_oblasts"] = feats.n_active_oblasts.astype(float)
    X["log_prev_duration"] = np.log1p(feats.prev_duration_min)
    X["log_national_starts_1h"] = np.log1p(feats.national_starts_1h)
    X["log_starts_24h"] = np.log1p(feats.starts_24h)
    X["log_min_since_last_end"] = np.log1p(feats.min_since_last_end)
    X["nb_active_frac"] = feats.nb_active_frac.astype(float)
    return X


FAMILIES = {"lognormal": LogNormalAFTFitter, "weibull": WeibullAFTFitter, "loglogistic": LogLogisticAFTFitter}
STANDARD = {"lognormal": norm, "weibull": gumbel_l}  # distribution of W; loglogistic is compared but not supported
# Weibull has the higher validation log-likelihood, but its median (the dashboard's headline number) is worse than
# log-normal's and than the per-oblast baseline, in both validation windows (exploration/07_duration_family.py).
FAMILY = "lognormal"


def fit(feats: pd.DataFrame, durations: pd.Series, regions: list[str], family: str = "lognormal"):
    X = design_matrix(feats, regions)
    df = X.assign(duration=durations.clip(lower=0.5).to_numpy())
    aft = FAMILIES[family](penalizer=0.01).fit(df, duration_col="duration")
    if family == "lognormal":
        mu, log_sigma = aft.params_.loc["mu_"], float(aft.params_.loc[("sigma_", "Intercept")])
    else:  # lifelines Weibull: lambda = exp(x beta), rho = exp(alpha) -> mu = x beta, sigma = 1 / rho
        mu, log_sigma = aft.params_.loc["lambda_"], -float(aft.params_.loc[("rho_", "Intercept")])
    params = {
        "family": family,
        "regions": regions,
        "mu_intercept": float(mu["Intercept"]),
        "mu_coef": {c: float(mu[c]) for c in X.columns},
        "log_sigma": log_sigma,
    }
    return params, aft


def compare_families(f_tr: pd.DataFrame, d_tr: pd.Series, f_va: pd.DataFrame, d_va: pd.Series,
                     regions: list[str]) -> dict[str, float]:
    """Mean validation log-likelihood (per alert, minutes scale) of each AFT distribution family."""
    # every alert here has an observed end (event = 1); score() needs the column explicitly
    tr = design_matrix(f_tr, regions).assign(duration=d_tr.clip(lower=0.5).to_numpy(), event=1)
    va = design_matrix(f_va, regions).assign(duration=d_va.clip(lower=0.5).to_numpy(), event=1)
    out = {}
    for name, cls in FAMILIES.items():
        m = cls(penalizer=0.01).fit(tr, duration_col="duration", event_col="event")
        out[name] = float(m.score(va, scoring_method="log_likelihood"))
    return out


def predict_mu_sigma(params: dict, feats: pd.DataFrame) -> tuple[np.ndarray, float]:
    X = design_matrix(feats, params["regions"])
    coef = pd.Series(params["mu_coef"])
    mu = params["mu_intercept"] + X[coef.index].to_numpy() @ coef.to_numpy()
    return mu, float(np.exp(params["log_sigma"]))


def quantile(mu, sigma, q, elapsed=0.0, family: str = "lognormal"):
    """Duration quantile, optionally conditional on the alert having lasted ``elapsed`` minutes."""
    w = STANDARD[family]
    elapsed = np.asarray(elapsed, dtype=float)
    s_e = np.where(elapsed > 0, w.sf((np.log(np.maximum(elapsed, 1e-9)) - mu) / sigma), 1.0)
    return np.exp(mu + sigma * w.ppf(1 - (1 - q) * s_e))


def survival_curve(mu: float, sigma: float, grid_min: np.ndarray, elapsed: float = 0.0,
                   family: str = "lognormal") -> np.ndarray:
    w = STANDARD[family]
    s = w.sf((np.log(grid_min) - mu) / sigma)
    if elapsed > 0:
        s = np.minimum(s / w.sf((np.log(elapsed) - mu) / sigma), 1.0)
        s[grid_min < elapsed] = 1.0
    return s


def evaluate(params: dict, feats: pd.DataFrame, durations: pd.Series, baseline_median: pd.Series) -> dict:
    mu, sigma = predict_mu_sigma(params, feats)
    family = params.get("family", "lognormal")
    med = quantile(mu, sigma, 0.5, family=family)
    lo, hi = quantile(mu, sigma, 0.1, family=family), quantile(mu, sigma, 0.9, family=family)
    y = durations.to_numpy()
    base = feats.region.map(baseline_median).to_numpy()
    return {
        "dur_mae_median_min": float(np.mean(np.abs(med - y))),
        "dur_mae_baseline_min": float(np.mean(np.abs(base - y))),
        "dur_medae_min": float(np.median(np.abs(med - y))),
        "dur_coverage_80": float(np.mean((y >= lo) & (y <= hi))),
        "dur_c_index": float(concordance_index(y, med)),
        # log-density of the duration in minutes: density of ln y, minus ln y
        "dur_mean_log_lik": float(np.mean(STANDARD[family].logpdf(np.log(np.maximum(y, 0.5)), mu, sigma)
                                          - np.log(np.maximum(y, 0.5)))),
        "sigma": sigma,
    }


def save(params: dict, path: Path) -> None:
    path.write_text(json.dumps(params, indent=1, ensure_ascii=False))


def load(path: Path) -> dict:
    return {"family": "lognormal", **json.loads(path.read_text())}  # older files had no family key
