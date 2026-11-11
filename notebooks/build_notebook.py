"""Generate notebooks/Uber_Ride_Analysis.ipynb (run once, then execute with nbconvert)."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Uber Ride Analysis (2016)\n\n"
       "End-to-end data science walkthrough of the *My Uber Drives - 2016* trip log: data preprocessing, "
       "exploratory data analysis (ride patterns, peak hours, fare trends, anomalies) and a daily "
       "ride-demand forecasting model.\n\n"
       "All reusable logic lives in `src/uber_analysis/`; this notebook calls it step by step and explains the results."),
    code("import sys, json\nfrom pathlib import Path\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT / 'src'))\n\n"
         "import pandas as pd\nimport matplotlib.pyplot as plt\n%matplotlib inline\n\n"
         "from uber_analysis.preprocessing import load_raw_data, clean_data, daily_demand, FARE_RATES\n"
         "from uber_analysis import visualization as viz\n"
         "from uber_analysis.forecasting import run_forecasting, miles_regression_check, FEATURES\n\n"
         "FIG = ROOT / 'reports' / 'figures'\n"
         "def show(path):\n"
         "    from IPython.display import Image, display\n"
         "    display(Image(filename=str(path)))"),
    md("## 1. The raw dataset\n\nOne person's Uber trips in 2016: start/end time, category (Business/Personal), "
       "start and stop location, miles, and trip purpose."),
    code("raw = load_raw_data(ROOT / 'data' / 'uber_drives_2016.csv')\nprint(raw.shape)\nraw.head()"),
    code("raw.tail(3)  # note the 'Totals' summary row at the end"),
    code("raw.isna().sum()"),
    md("## 2. Preprocessing\n\n"
       "1. Remove the `Totals` footer row.\n2. Parse dates with an explicit `%m/%d/%Y %H:%M` format.\n"
       "3. Drop exact duplicates.\n4. Fill missing `purpose` with **Unspecified** (dropping would lose 43% of trips).\n"
       "5. Fix the `Kar?chi` encoding error.\n6. Remove trips with zero/negative duration.\n"
       "7. Engineer features: hour, weekday, month, time of day, duration, speed, round trip, distance band, estimated fare.\n"
       "8. **Flag** (not drop) outliers: very long trips (IQR rule) and speeds > 80 mph."),
    code("df, log = clean_data(raw)\npd.Series(log, name='rows')"),
    code("df[['start_time','category','start_location','stop_location','miles','purpose',\n"
         "    'duration_min','speed_mph','hour','weekday','estimated_fare']].head()"),
    md("### Fare estimation\n\nThe dataset has **no fare column**, so fares are estimated with an assumed UberX-style "
       "rate card. Fare trends below therefore reflect distance and time, not real surge pricing."),
    code("FARE_RATES"),
    md("## 3. Summary statistics"),
    code("df[['miles','duration_min','speed_mph','estimated_fare']].describe().round(2)"),
    code("daily = daily_demand(df)\n"
         "print('Rides per day: mean %.2f, max %d, days with no rides %d' % (\n"
         "    daily.rides.mean(), daily.rides.max(), (daily.rides == 0).sum()))\n"
         "df['category'].value_counts(normalize=True).round(3)"),
    md("## 4. Exploratory data analysis"),
    code("figs = viz.make_all_eda_figures(df, daily, FIG)\nplt.close('all')\nlen(figs)"),
    md("### 4.1 Peak hours\nDemand builds from 8 am and stays high from **1 pm to 6 pm**; almost no rides between 1 am and 6 am."),
    code("show(FIG / '01_rides_by_hour.png')"),
    md("### 4.2 Day of week\n**Friday** is clearly the busiest day; weekends are not quieter than mid-week."),
    code("show(FIG / '02_rides_by_weekday.png')"),
    md("### 4.3 Monthly trend"),
    code("show(FIG / '03_rides_by_month.png')"),
    md("### 4.4 Heatmaps\nFriday late morning and early afternoon are the hottest slots."),
    code("show(FIG / '04_heatmap_weekday_hour.png')"),
    code("show(FIG / '05_heatmap_month_weekday.png')"),
    md("### 4.5 Time series of daily demand\nDemand comes in bursts (travel periods) with quiet spells in April-May and September."),
    code("show(FIG / '06_daily_demand_timeseries.png')"),
    md("### 4.6 Category, purpose and locations"),
    code("show(FIG / '07_category_and_purpose.png')"),
    code("show(FIG / '08_top_locations.png')"),
    md("### 4.7 Distance, duration and anomalies"),
    code("show(FIG / '09_distance_duration_anomalies.png')"),
    code("df.loc[df.is_speed_anomaly, ['start_time','start_location','stop_location','miles','duration_min','speed_mph']]"
         ".round(1).head(10)"),
    code("show(FIG / '10_distance_by_purpose.png')"),
    md("### 4.8 Fare trends (estimated)"),
    code("show(FIG / '11_fare_trends.png')"),
    code("show(FIG / '12_fare_by_distance_band.png')"),
    md("## 5. Ride-demand forecasting\n\n"
       "**Target:** number of rides tomorrow.  **Features:** calendar (weekday, month, day of month, weekend) and "
       "past demand (lags 1/2/7/14 days, 7- and 28-day rolling mean/std, zero-ride days in last week). Every lag "
       "uses only past days.\n\n"
       "**Split:** train Jan-Oct, test Nov-Dec (time-ordered, no shuffling). TimeSeriesSplit CV on the training set.\n\n"
       "**Baselines:** yesterday's count, same day last week, 7-day average. A model is only useful if it beats these."),
    code("FEATURES"),
    code("fc = run_forecasting(daily)\nprint('Train:', fc['train_period'])\nprint('Test :', fc['test_period'])\n"
         "pd.DataFrame(fc['metrics']).T"),
    code("viz.forecast_plot(daily, fc['predictions'], fc['best_model'], FIG)\n"
         "viz.model_comparison(fc['metrics'], FIG)\nviz.feature_importance(fc['feature_importance'], FIG)\n"
         "plt.close('all')\nshow(FIG / '13_demand_forecast.png')"),
    code("show(FIG / '14_model_comparison.png')"),
    code("show(FIG / '15_feature_importance.png')"),
    md("### Why not predict trip miles?\nThe first draft (`notebooks/early_work/`) predicted miles from hour/day/month. "
       "Re-running it shows it is no better than always guessing the median, because trip length depends on the "
       "destination, not the time of day:"),
    code("miles_regression_check(df)"),
    md("## 6. Conclusions\n\nSee the README for the full list of insights, challenges, improvements and business "
       "applications. In short: demand peaks 1-6 pm and on Fridays, 93% of trips are business, a few long "
       "inter-city trips account for 39% of all miles, and short trips cost the most per mile. The best demand "
       "model (Ridge regression) beats the naive baselines, but daily rides for a single rider are noisy, so "
       "forecasts are best used for smoothed demand levels rather than exact daily counts."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, Path(__file__).with_name("Uber_Ride_Analysis.ipynb"))
print("written")
