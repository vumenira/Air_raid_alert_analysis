"""Descriptive time-series analysis of the alert history (volunteer data, oblast level, UTC -> Kyiv time).

Writes five figures to docs/figures/ and prints the numbers behind them:
  1. daily alert starts (all modelled oblasts), with a 7-day mean;
  2. start rate by hour x weekday (Kyiv time), last 12 months;
  3. weekly share of calm oblast-hours followed by a start within 1 h (the start model's label), with the
     train / validation / test windows: the base-rate drift;
  4. autocorrelation of hourly national start counts (clustering and the daily cycle);
  5. alert duration distribution by year.

Usage: PYTHONPATH=. uv run python exploration/00_time_series_eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.data import KYIV_TZ, ROOT, TRAIN_START, load_alerts, regions
from training.features import hourly_grid, start_features, start_labels
from training.train import split_times

OUT = ROOT / "docs" / "figures"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#9ec5f4", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 140,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote docs/figures/{name}")


def daily_counts(alerts):
    daily = alerts.set_index("started_at").resample("D").size().iloc[:-1]  # last day is incomplete
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(daily.index, daily, color=BLUE_LIGHT, lw=0.8)
    ax.plot(daily.index, daily.rolling(7).mean(), color=BLUE, lw=2)
    ax.set_title("Alert starts per day, 24 oblasts (light: daily, dark: 7-day mean)")
    ax.set_ylabel("alerts started")
    save(fig, "01_daily_starts.png")
    yearly = daily.groupby(daily.index.year).mean()
    print("mean starts/day by year:", yearly.round(1).to_dict())


def hour_weekday(alerts):
    recent = alerts[alerts.started_at >= alerts.started_at.max() - pd.Timedelta(days=365)]
    local = recent.started_at.dt.tz_convert(KYIV_TZ)
    n_weeks = 365 / 7
    table = pd.crosstab(local.dt.dayofweek, local.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0) / n_weeks
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(table, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0f5fc", "#86b6ef", "#256abf", "#0d366b"]))
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour (Kyiv time)")
    ax.grid(False)
    ax.set_title("Alert starts per hour slot, last 12 months (average over 24 oblasts, per week)")
    fig.colorbar(im, ax=ax, label="starts / week", shrink=0.9)
    save(fig, "02_hour_weekday.png")
    by_hour = table.sum(0)
    print(f"busiest Kyiv hours: {by_hour.nlargest(3).index.tolist()}, quietest: {by_hour.nsmallest(3).index.tolist()}; "
          f"weekday max/min ratio {table.sum(1).max() / table.sum(1).min():.2f}")


def base_rate_drift(alerts, regs):
    valid_start, test_start, grid_end = split_times(alerts)
    feats = start_features(alerts, hourly_grid(TRAIN_START, grid_end), regs)
    y = start_labels(alerts, feats)
    calm = feats.active_now == 0
    s = pd.Series(y[calm.to_numpy()], index=feats.t[calm]).resample("W").mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axvspan(valid_start, test_start, color="#efeeea", lw=0)
    ax.axvspan(test_start, grid_end, color="#d9d8d3", lw=0)
    ax.plot(s.index, s, color=BLUE, lw=2)
    ax.set_ylim(0, 0.3)
    ax.text(valid_start, 0.29, "valid ", ha="right", va="top", color=INK_2, fontsize=9)
    ax.text(grid_end, 0.29, " test", ha="left", va="top", color=INK_2, fontsize=9)
    ax.set_xlim(None, grid_end + pd.Timedelta(days=30))
    ax.set_ylabel("P(start within 1 h | calm)")
    ax.set_title(f"Weekly base rate of the start model's label since {TRAIN_START:%Y-%m}: it drifts up into the test window")
    save(fig, "03_base_rate_drift.png")
    t = feats.t[calm]
    for name, m in [("train", t < valid_start), ("valid", (t >= valid_start) & (t < test_start)), ("test", t >= test_start)]:
        print(f"positive rate {name}: {y[calm.to_numpy()][m.to_numpy()].mean():.3f}")


def autocorrelation(alerts):
    recent = alerts[alerts.started_at >= alerts.started_at.max() - pd.Timedelta(days=365)]
    x = recent.set_index("started_at").resample("h").size().to_numpy(dtype=float)
    x = x - x.mean()
    lags = np.arange(1, 24 * 7 + 1)
    acf = np.array([np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in lags])
    band = 1.96 / np.sqrt(len(x))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhspan(-band, band, color=GRID, lw=0)
    ax.vlines(lags, 0, acf, color=BLUE, lw=1.2)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_xticks(range(0, 24 * 7 + 1, 24), [f"{d} d" for d in range(8)])
    ax.set_xlabel("lag")
    ax.set_title("Autocorrelation of hourly national alert starts, last 12 months (grey: 95% noise band)")
    save(fig, "04_acf_hourly_starts.png")
    print(f"ACF lag 1 h {acf[0]:.2f}, 2 h {acf[1]:.2f}, 12 h {acf[11]:.2f}, 24 h {acf[23]:.2f}, 7 d {acf[-1]:.2f}")


def durations(alerts):
    clean = alerts[~alerts.naive & alerts.finished_at.notna() & (alerts.duration_min > 0)]
    years = sorted(clean.started_at.dt.year.unique())
    data = [np.log10(clean[clean.started_at.dt.year == y].duration_min.clip(1, 24 * 60)) for y in years]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE_LIGHT)
        b.set_edgecolor(BLUE)
        b.set_alpha(1)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(BLUE)
        parts[k].set_linewidth(1)
    ax.set_xticks(range(1, len(years) + 1), years)
    ticks = [1, 5, 15, 30, 60, 120, 360, 1440]
    ax.set_yticks(np.log10(ticks), [f"{t} min" if t < 60 else f"{t // 60} h" for t in ticks])
    ax.set_title("Alert duration by year (log scale, clipped to 1 min - 24 h; line = median; naive end times excluded)")
    save(fig, "05_duration_by_year.png")
    print("median duration (min) by year:", clean.groupby(clean.started_at.dt.year).duration_min.median().round(1).to_dict())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alerts = load_alerts()
    regs = regions(alerts)
    alerts = alerts[alerts.region.isin(regs)]
    daily_counts(alerts)
    hour_weekday(alerts)
    base_rate_drift(alerts, regs)
    autocorrelation(alerts)
    durations(alerts)


if __name__ == "__main__":
    main()
