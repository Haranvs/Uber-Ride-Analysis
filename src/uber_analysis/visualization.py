"""Plotting helpers. Each function draws one figure and saves it as a PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .preprocessing import MONTH_ORDER, WEEKDAY_ORDER  # noqa: E402

UBER_BLACK = "#1f1f1f"
ACCENT = "#276EF1"   # Uber blue
ACCENT_2 = "#05A357"  # green
ACCENT_3 = "#E11900"  # red

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def _label_bars(ax, fmt="{:.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=8)


def rides_by_hour(df, out):
    counts = df["hour"].value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = [ACCENT_3 if v >= counts.nlargest(4).min() else ACCENT for v in counts]
    ax.bar(counts.index, counts.values, color=colors)
    _label_bars(ax)
    ax.set(title="Ride demand by hour of day (red = top-4 peak hours)", xlabel="Hour of day", ylabel="Rides",
           xticks=range(24))
    return _save(fig, out, "01_rides_by_hour")


def rides_by_weekday(df, out):
    counts = df["weekday"].value_counts().reindex(WEEKDAY_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts.index, counts.values, color=[ACCENT if d not in ("Saturday", "Sunday") else ACCENT_2
                                               for d in counts.index])
    _label_bars(ax)
    ax.set(title="Rides by day of week (green = weekend)", xlabel="", ylabel="Rides")
    return _save(fig, out, "02_rides_by_weekday")


def rides_by_month(df, out):
    g = df.groupby("month", observed=False).agg(rides=("miles", "size"), miles=("miles", "sum"))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(g.index.astype(str), g["rides"], color=ACCENT, label="Rides")
    _label_bars(ax)
    ax2 = ax.twinx()
    ax2.plot(g.index.astype(str), g["miles"], color=ACCENT_3, marker="o", label="Total miles")
    ax2.set_ylabel("Total miles", color=ACCENT_3)
    ax2.grid(False)
    ax.set(title="Monthly rides and total miles", xlabel="", ylabel="Rides")
    return _save(fig, out, "03_rides_by_month")


def weekday_hour_heatmap(df, out):
    pivot = df.pivot_table(index="weekday", columns="hour", values="miles", aggfunc="size",
                           observed=False).reindex(index=WEEKDAY_ORDER, columns=range(24)).fillna(0)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.4, cbar_kws={"label": "Rides"}, ax=ax)
    ax.set(title="Heatmap: rides by weekday and hour", xlabel="Hour of day", ylabel="")
    return _save(fig, out, "04_heatmap_weekday_hour")


def month_weekday_heatmap(df, out):
    pivot = df.pivot_table(index="month", columns="weekday", values="miles", aggfunc="size",
                           observed=False).reindex(index=MONTH_ORDER, columns=WEEKDAY_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f", linewidths=0.4, cbar_kws={"label": "Rides"}, ax=ax)
    ax.set(title="Heatmap: rides by month and weekday", xlabel="", ylabel="")
    return _save(fig, out, "05_heatmap_month_weekday")


def daily_timeseries(daily, out):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(daily.index, daily["rides"], color="#b8c7e8", width=1, label="Daily rides")
    ax.plot(daily.index, daily["rides"].rolling(7, center=True).mean(), color=ACCENT, lw=2, label="7-day average")
    ax.plot(daily.index, daily["rides"].rolling(30, center=True).mean(), color=ACCENT_3, lw=2, label="30-day average")
    ax.set(title="Daily ride demand through 2016", xlabel="", ylabel="Rides per day")
    ax.legend(loc="upper left")
    return _save(fig, out, "06_daily_demand_timeseries")


def category_purpose(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2.4]})
    cat = df["category"].value_counts()
    axes[0].bar(cat.index, cat.values, color=[ACCENT, ACCENT_2])
    _label_bars(axes[0])
    axes[0].set(title="Trip category", ylabel="Rides")
    pur = df["purpose"].value_counts().sort_values()
    axes[1].barh(pur.index, pur.values, color=[UBER_BLACK if p == "Unspecified" else ACCENT for p in pur.index])
    for i, v in enumerate(pur.values):
        axes[1].text(v + 3, i, str(v), va="center", fontsize=8)
    axes[1].set(title="Trip purpose (black = not recorded)", xlabel="Rides")
    return _save(fig, out, "07_category_and_purpose")


def top_locations(df, out, n=12):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in [(axes[0], "start_location", "Top pickup locations"),
                           (axes[1], "stop_location", "Top drop-off locations")]:
        s = df[col].value_counts().head(n).sort_values()
        ax.barh(s.index, s.values, color=[UBER_BLACK if x == "Unknown Location" else ACCENT for x in s.index])
        ax.set(title=title, xlabel="Rides")
    return _save(fig, out, "08_top_locations")


def distance_duration(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["miles"], bins=50, log_scale=True, color=ACCENT, ax=axes[0])
    axes[0].set(title="Trip distance (log scale)", xlabel="Miles")
    sns.histplot(df["duration_min"], bins=50, log_scale=True, color=ACCENT_2, ax=axes[1])
    axes[1].set(title="Trip duration (log scale)", xlabel="Minutes")
    normal = df[~df["is_speed_anomaly"]]
    axes[2].scatter(normal["duration_min"], normal["miles"], s=10, alpha=0.5, color=ACCENT, label="Normal")
    anom = df[df["is_speed_anomaly"] | df["is_long_trip_outlier"]]
    axes[2].scatter(anom["duration_min"], anom["miles"], s=25, color=ACCENT_3, label="Flagged outlier/anomaly")
    axes[2].set(xscale="log", yscale="log", title="Distance vs duration", xlabel="Minutes", ylabel="Miles")
    axes[2].legend()
    return _save(fig, out, "09_distance_duration_anomalies")


def purpose_distance_box(df, out):
    order = df.groupby("purpose")["miles"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="purpose", y="miles", order=order, color="#b8c7e8", fliersize=2, ax=ax)
    ax.set(yscale="log", title="Trip distance by purpose", xlabel="", ylabel="Miles (log scale)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out, "10_distance_by_purpose")


def fare_trends(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    m = df.groupby("month", observed=False)["estimated_fare"].agg(["sum", "mean", "median"])
    axes[0].bar(m.index.astype(str), m["sum"], color=ACCENT)
    axes[0].set(title="Estimated monthly spend (USD)", ylabel="USD")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].plot(m.index.astype(str), m["mean"], marker="o", color=ACCENT_3, label="Mean fare")
    axes[1].plot(m.index.astype(str), m["median"], marker="s", color=ACCENT_2, label="Median fare")
    axes[1].set(title="Average estimated fare per ride", ylabel="USD")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)
    h = df.groupby("hour")["estimated_fare"].mean().reindex(range(24))
    axes[2].bar(h.index, h.values, color=UBER_BLACK)
    axes[2].set(title="Average estimated fare by pickup hour", xlabel="Hour", ylabel="USD")
    return _save(fig, out, "11_fare_trends")


def fare_by_distance_band(df, out):
    g = df.groupby("distance_band", observed=False).agg(
        rides=("miles", "size"), avg_fare=("estimated_fare", "mean"), fare_per_mile=("fare_per_mile", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(g.index.astype(str), g["avg_fare"], color=ACCENT)
    _label_bars(axes[0], "${:.0f}")
    axes[0].set(title="Average estimated fare by distance band", ylabel="USD")
    axes[1].bar(g.index.astype(str), g["fare_per_mile"], color=ACCENT_3)
    _label_bars(axes[1], "${:.2f}")
    axes[1].set(title="Median estimated fare per mile", ylabel="USD per mile")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    return _save(fig, out, "12_fare_by_distance_band")


def forecast_plot(daily, preds, best_model, out):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    hist = daily.loc["2016-09-01":, "rides"]
    ax.plot(hist.index, hist.values, color="#999999", lw=1, label="Actual (history)")
    ax.plot(preds.index, preds["actual"], color=UBER_BLACK, lw=1.8, marker="o", ms=3, label="Actual (test)")
    ax.plot(preds.index, preds[best_model], color=ACCENT_3, lw=2, label=f"{best_model} forecast")
    ax.plot(preds.index, preds["7-day average"], color=ACCENT, lw=1.3, ls="--", label="7-day average baseline")
    ax.axvline(preds.index.min(), color="k", ls=":", lw=1)
    ax.text(preds.index.min(), ax.get_ylim()[1] * 0.95, "  test period ->", fontsize=9)
    ax.set(title="Daily ride-demand forecast vs actual (Nov-Dec 2016 held out)", ylabel="Rides per day")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "13_demand_forecast")


def model_comparison(metrics: dict, out):
    m = pd.DataFrame(metrics).T.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [ACCENT if t == "ml" else "#9e9e9e" for t in m["type"]]
    ax.barh(m.index, m["MAE"].astype(float), color=colors)
    for i, v in enumerate(m["MAE"].astype(float)):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(title="Test MAE (rides/day) - lower is better (blue = ML, grey = baseline)", xlabel="MAE")
    return _save(fig, out, "14_model_comparison")


def feature_importance(imp: pd.Series, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    s = imp.sort_values()
    ax.barh(s.index, s.values, color=ACCENT)
    ax.set(title="Random Forest feature importance", xlabel="Importance")
    return _save(fig, out, "15_feature_importance")


def make_all_eda_figures(df, daily, out_dir: Path) -> list[Path]:
    return [
        rides_by_hour(df, out_dir), rides_by_weekday(df, out_dir), rides_by_month(df, out_dir),
        weekday_hour_heatmap(df, out_dir), month_weekday_heatmap(df, out_dir), daily_timeseries(daily, out_dir),
        category_purpose(df, out_dir), top_locations(df, out_dir), distance_duration(df, out_dir),
        purpose_distance_box(df, out_dir), fare_trends(df, out_dir), fare_by_distance_band(df, out_dir),
    ]
