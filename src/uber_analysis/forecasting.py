"""Daily ride-demand forecasting.

Goal: predict how many rides will be taken on a given day, using only
information that would be known the evening before (calendar features plus
lagged demand). Models are trained on January-October 2016 and tested on the
unseen November-December 2016 period, so the test mimics real forecasting
(no shuffling, no peeking at the future).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_START = "2016-11-01"

FEATURES = [
    "dayofweek", "is_weekend", "month", "dayofmonth",
    "lag_1", "lag_2", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_28", "roll_std_7", "zero_days_last_7",
]


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build calendar + lag features. Every lag/rolling value is shifted so it uses past days only."""
    f = pd.DataFrame(index=daily.index)
    y = daily["rides"]
    f["rides"] = y
    f["dayofweek"] = daily.index.dayofweek
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(int)
    f["month"] = daily.index.month
    f["dayofmonth"] = daily.index.day
    for lag in (1, 2, 7, 14):
        f[f"lag_{lag}"] = y.shift(lag)
    past = y.shift(1)
    f["roll_mean_7"] = past.rolling(7).mean()
    f["roll_mean_28"] = past.rolling(28).mean()
    f["roll_std_7"] = past.rolling(7).std()
    f["zero_days_last_7"] = past.eq(0).rolling(7).sum()
    return f.dropna()


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def build_models() -> dict:
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42),
    }


def run_forecasting(daily: pd.DataFrame) -> dict:
    """Train baselines + ML models, return metrics, test predictions and feature importances."""
    data = make_features(daily)
    train = data[data.index < TEST_START]
    test = data[data.index >= TEST_START]
    X_train, y_train = train[FEATURES], train["rides"]
    X_test, y_test = test[FEATURES], test["rides"]

    preds = pd.DataFrame({"actual": y_test})
    results = {}

    # Baselines a model must beat to be useful.
    preds["Naive (yesterday)"] = test["lag_1"]
    preds["Seasonal naive (same day last week)"] = test["lag_7"]
    preds["7-day average"] = test["roll_mean_7"]
    for name in ["Naive (yesterday)", "Seasonal naive (same day last week)", "7-day average"]:
        results[name] = {**_metrics(y_test, preds[name]), "type": "baseline"}

    tscv = TimeSeriesSplit(n_splits=5)
    fitted = {}
    for name, model in build_models().items():
        cv_mae = -cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
        model.fit(X_train, y_train)
        p = np.clip(model.predict(X_test), 0, None)
        preds[name] = p
        results[name] = {**_metrics(y_test, p), "CV_MAE_train": round(float(cv_mae.mean()), 3), "type": "ml"}
        fitted[name] = model

    best = min((n for n, r in results.items() if r["type"] == "ml"), key=lambda n: results[n]["MAE"])
    importances = pd.Series(
        fitted["Random Forest"].feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {
        "train_period": f"{train.index.min().date()} to {train.index.max().date()} ({len(train)} days)",
        "test_period": f"{test.index.min().date()} to {test.index.max().date()} ({len(test)} days)",
        "metrics": results,
        "best_model": best,
        "predictions": preds,
        "feature_importance": importances,
        "models": fitted,
    }


def miles_regression_check(df: pd.DataFrame) -> dict:
    """Re-run the first-draft idea (predict trip miles from hour/day/month) to document why it fails."""
    from sklearn.model_selection import train_test_split

    X = pd.DataFrame({"hour": df["hour"], "day": df["start_time"].dt.dayofweek, "month": df["month_num"]})
    y = df["miles"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    base = np.full_like(y_te, y_tr.median(), dtype=float)
    return {"model": _metrics(y_te, pred), "median_baseline": _metrics(y_te, base)}
