"""Run the full Uber Ride Analysis pipeline.

    python run_analysis.py

Steps: load + clean data -> summary statistics -> EDA figures ->
ride-demand forecasting -> save figures, tables and metrics to reports/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from uber_analysis import visualization as viz  # noqa: E402
from uber_analysis.forecasting import miles_regression_check, run_forecasting  # noqa: E402
from uber_analysis.preprocessing import daily_demand, load_clean_data  # noqa: E402

DATA = ROOT / "data" / "uber_drives_2016.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def summary_statistics(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    hour_counts = df["hour"].value_counts()
    weekday_counts = df["weekday"].value_counts()
    month_counts = df["month"].value_counts()
    return {
        "total_rides": len(df),
        "date_range": f"{df['start_time'].min():%Y-%m-%d} to {df['start_time'].max():%Y-%m-%d}",
        "total_miles": round(df["miles"].sum(), 1),
        "mean_miles": round(df["miles"].mean(), 2),
        "median_miles": round(df["miles"].median(), 2),
        "mean_duration_min": round(df["duration_min"].mean(), 1),
        "median_duration_min": round(df["duration_min"].median(), 1),
        "median_speed_mph": round(df["speed_mph"].median(), 1),
        "business_share_pct": round(100 * (df["category"] == "Business").mean(), 1),
        "purpose_recorded_pct": round(100 * (df["purpose"] != "Unspecified").mean(), 1),
        "round_trip_share_pct": round(100 * df["round_trip"].mean(), 1),
        "peak_hours": [int(h) for h in hour_counts.nlargest(4).index],
        "rides_13_to_18h_pct": round(100 * df["hour"].between(13, 18).mean(), 1),
        "rides_0_to_6h_pct": round(100 * df["hour"].between(0, 6).mean(), 1),
        "busiest_weekday": str(weekday_counts.idxmax()),
        "quietest_weekday": str(weekday_counts.idxmin()),
        "busiest_month": str(month_counts.idxmax()),
        "quietest_month": str(month_counts.idxmin()),
        "avg_rides_per_day": round(daily["rides"].mean(), 2),
        "max_rides_in_a_day": int(daily["rides"].max()),
        "days_with_zero_rides": int((daily["rides"] == 0).sum()),
        "estimated_total_spend_usd": round(df["estimated_fare"].sum(), 0),
        "estimated_mean_fare_usd": round(df["estimated_fare"].mean(), 2),
        "estimated_median_fare_usd": round(df["estimated_fare"].median(), 2),
        "top_pickup_locations": df["start_location"].value_counts().head(5).to_dict(),
        "purpose_counts": df["purpose"].value_counts().to_dict(),
        "share_of_miles_from_trips_over_25mi_pct": round(
            100 * df.loc[df["miles"] > 25, "miles"].sum() / df["miles"].sum(), 1),
        "share_of_trips_over_25mi_pct": round(100 * (df["miles"] > 25).mean(), 1),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("1/4 Loading and cleaning data ...")
    df, cleaning_log = load_clean_data(DATA)
    daily = daily_demand(df)
    df.to_csv(REPORTS / "uber_rides_clean.csv", index=False)
    for k, v in cleaning_log.items():
        print(f"    {k:32s} {v}")

    print("2/4 Summary statistics ...")
    stats = summary_statistics(df, daily)
    df.describe(include="number").round(2).to_csv(REPORTS / "numeric_summary.csv")
    (df.groupby("purpose").agg(rides=("miles", "size"), avg_miles=("miles", "mean"),
                               avg_minutes=("duration_min", "mean"), avg_est_fare=("estimated_fare", "mean"))
       .round(2).sort_values("rides", ascending=False).to_csv(REPORTS / "purpose_summary.csv"))

    print("3/4 EDA figures ...")
    figs = viz.make_all_eda_figures(df, daily, FIGURES)

    print("4/4 Demand forecasting ...")
    fc = run_forecasting(daily)
    figs += [viz.forecast_plot(daily, fc["predictions"], fc["best_model"], FIGURES),
             viz.model_comparison(fc["metrics"], FIGURES),
             viz.feature_importance(fc["feature_importance"], FIGURES)]
    fc["predictions"].round(2).to_csv(REPORTS / "forecast_test_predictions.csv")
    miles_check = miles_regression_check(df)

    results = {
        "cleaning_log": cleaning_log,
        "summary_statistics": stats,
        "forecasting": {
            "train_period": fc["train_period"],
            "test_period": fc["test_period"],
            "best_model": fc["best_model"],
            "daily_metrics": fc["metrics"],
            "feature_importance": fc["feature_importance"].round(3).to_dict(),
        },
        "first_draft_miles_regression": miles_check,
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\nSaved {len(figs)} figures to {FIGURES.relative_to(ROOT)}/ and results to reports/results.json")
    print("\nDaily forecast metrics (test period):")
    print(pd.DataFrame(fc["metrics"]).T.to_string())
    print(f"\nBest ML model: {fc['best_model']}")


if __name__ == "__main__":
    main()
