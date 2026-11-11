"""Load and clean the raw Uber trip log.

The raw file ("My Uber Drives - 2016") has seven columns whose names end in
"*", a "Totals" footer row, one duplicate trip, missing trip purposes and a few
trips with zero or impossible durations. ``load_clean_data`` fixes all of these
and adds the time and trip features used by the rest of the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "START_DATE*": "start_time",
    "END_DATE*": "end_time",
    "CATEGORY*": "category",
    "START*": "start_location",
    "STOP*": "stop_location",
    "MILES*": "miles",
    "PURPOSE*": "purpose",
}

DATE_FORMAT = "%m/%d/%Y %H:%M"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# The dataset has no fare column. Fares are ESTIMATED with this assumed
# UberX-style rate card (USD) so pricing trends can be explored. Change these
# values to match a real rate card if one is available.
FARE_RATES = {
    "base": 2.00,        # flag-fall
    "per_mile": 1.00,
    "per_minute": 0.20,
    "booking_fee": 1.50,
    "minimum": 6.00,
}

# Trips faster than this average speed are physically implausible for city
# driving and are flagged as anomalies.
MAX_PLAUSIBLE_MPH = 80


def estimate_fare(miles: pd.Series, minutes: pd.Series, rates: dict = FARE_RATES) -> pd.Series:
    """Estimated fare = base + distance + time + booking fee, never below the minimum."""
    fare = rates["base"] + rates["per_mile"] * miles + rates["per_minute"] * minutes + rates["booking_fee"]
    return fare.clip(lower=rates["minimum"]).round(2)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned trips and a log of what each cleaning step removed or filled."""
    log: dict[str, int] = {"raw_rows": len(raw)}
    df = raw.rename(columns=RAW_COLUMNS).copy()

    # 1. Drop the "Totals" summary row at the bottom of the file.
    is_total = df["start_time"].astype(str).str.strip().str.lower().eq("totals")
    log["totals_rows_removed"] = int(is_total.sum())
    df = df[~is_total]

    # 2. Parse timestamps with an explicit format (month/day/year hour:minute).
    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce")
    bad_dates = df["start_time"].isna() | df["end_time"].isna()
    log["unparseable_dates_removed"] = int(bad_dates.sum())
    df = df[~bad_dates]

    # 3. Remove exact duplicate trips.
    dupes = df.duplicated()
    log["duplicates_removed"] = int(dupes.sum())
    df = df[~dupes]

    # 4. Missing purpose is not random: the driver simply did not tag the trip.
    #    Keep these trips and label them "Unspecified" instead of dropping 43% of the data.
    log["purpose_filled_unspecified"] = int(df["purpose"].isna().sum())
    df["purpose"] = df["purpose"].fillna("Unspecified")

    # 5. Tidy text columns.
    for col in ("category", "start_location", "stop_location", "purpose"):
        df[col] = df[col].astype(str).str.strip()
    # The raw file has a broken character in "Karachi".
    for col in ("start_location", "stop_location"):
        df[col] = df[col].replace({"Kar?chi": "Karachi"})

    # 6. Trip duration; trips that end before or when they start cannot be real.
    df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    bad_duration = df["duration_min"] <= 0
    log["non_positive_duration_removed"] = int(bad_duration.sum())
    df = df[~bad_duration].copy()

    df = add_features(df)

    # 7. Flag (do not drop) statistical / physical anomalies so they stay visible in the EDA.
    q1, q3 = df["miles"].quantile([0.25, 0.75])
    df["is_long_trip_outlier"] = df["miles"] > q3 + 3 * (q3 - q1)
    df["is_speed_anomaly"] = df["speed_mph"] > MAX_PLAUSIBLE_MPH
    log["long_trip_outliers_flagged"] = int(df["is_long_trip_outlier"].sum())
    log["speed_anomalies_flagged"] = int(df["is_speed_anomaly"].sum())

    df = df.sort_values("start_time").reset_index(drop=True)
    log["clean_rows"] = len(df)
    return df, log


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, trip and estimated-fare features."""
    df = df.copy()
    st = df["start_time"]
    df["date"] = st.dt.normalize()
    df["hour"] = st.dt.hour
    df["weekday"] = pd.Categorical(st.dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
    df["month"] = pd.Categorical(st.dt.strftime("%b"), categories=MONTH_ORDER, ordered=True)
    df["month_num"] = st.dt.month
    df["is_weekend"] = st.dt.dayofweek >= 5
    df["time_of_day"] = pd.cut(
        df["hour"], bins=[-1, 5, 11, 16, 20, 23],
        labels=["Night (0-5)", "Morning (6-11)", "Afternoon (12-16)", "Evening (17-20)", "Late (21-23)"],
    )
    df["speed_mph"] = df["miles"] / (df["duration_min"] / 60)
    df["round_trip"] = df["start_location"].eq(df["stop_location"])
    df["estimated_fare"] = estimate_fare(df["miles"], df["duration_min"])
    df["fare_per_mile"] = df["estimated_fare"] / df["miles"]
    df["distance_band"] = pd.cut(
        df["miles"], bins=[0, 3, 10, 25, np.inf],
        labels=["Short (<3 mi)", "Medium (3-10 mi)", "Long (10-25 mi)", "Very long (25+ mi)"],
    )
    return df


def load_clean_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    return clean_data(load_raw_data(path))


def daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Rides per calendar day, with days that had no rides filled as 0."""
    counts = df.groupby("date").size()
    full_range = pd.date_range(counts.index.min().replace(month=1, day=1),
                               counts.index.max().replace(month=12, day=31), freq="D")
    daily = counts.reindex(full_range, fill_value=0).rename("rides").to_frame()
    daily.index.name = "date"
    miles = df.groupby("date")["miles"].sum().reindex(full_range, fill_value=0)
    daily["miles"] = miles.values
    return daily
