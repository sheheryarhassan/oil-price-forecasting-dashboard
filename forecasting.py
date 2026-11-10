"""Monthly WTI futures forecasts and scenario comparisons."""
import numpy as np
import pandas as pd
from config import DATA_DIR, SCENARIOS


def future_row(history, date, feature_cols):
    values = {f"WTI_Lag{lag}": history[-lag] for lag in (1, 2, 3, 6, 12)}
    for months in (3, 6, 12):
        window = history[-months:]
        values[f"WTI_Roll_Mean_{months}M"] = np.mean(window)
        values[f"WTI_Roll_Std_{months}M"] = np.std(window, ddof=1)
    values["Month_Sin"] = np.sin(2 * np.pi * date.month / 12)
    values["Month_Cos"] = np.cos(2 * np.pi * date.month / 12)
    return np.array([[values[name] for name in feature_cols]])


def generate_forecasts(df_hist, feature_cols, trained_models):
    start = df_hist.index[-1] + pd.offsets.MonthBegin()
    dates = pd.date_range(start=start, periods=36, freq="MS", name="date")
    prices = df_hist["WTI"]
    models = {"LinearRegression": "WTI_LR", "RandomForest": "WTI_RF", "XGBoost": "WTI_XGB"}
    # Each model feeds its own prediction back into the next month's lags.
    paths = {"Persistence": np.full(len(dates), prices.iloc[-1])}
    for name in models:
        history = prices.iloc[-12:].tolist()
        predicted = []
        for date in dates:
            value = float(trained_models[name].predict(future_row(history, date, feature_cols))[0])
            history.append(value)
            predicted.append(value)
        paths[name] = np.array(predicted)

    reference = pd.DataFrame(index=dates)
    for name, column in models.items():
        reference[column] = paths[name]
    if trained_models.get("ARIMA") is not None:
        reference["WTI_ARIMA"] = np.asarray(trained_models["ARIMA"].forecast(len(dates)))
    if trained_models.get("Prophet") is not None:
        future = pd.DataFrame({"ds": dates})
        reference["WTI_Prophet"] = trained_models["Prophet"].predict(future)["yhat"].to_numpy()

    base = SCENARIOS["Base"]
    last_risk = df_hist["GeopoliticalRiskIndex"].iloc[-1]
    # Scenario premiums use the project's assumptions, relative to Base.
    # Phase in each premium over several months instead of changing history.
    phase = 1 - 0.85 ** np.arange(1, len(dates) + 1)
    uncertainty = 1.5 * prices.tail(24).std() * (1 + 0.05 * np.arange(len(dates)))
    all_forecasts = {}
    for scenario_name, scenario in SCENARIOS.items():
        risk = np.clip(last_risk * scenario["geo_risk_multiplier"], 0, 100)
        premium = (0.25 * (risk - last_risk)
                   + 12 * (scenario["opec_cut_factor"] - base["opec_cut_factor"])
                   + 30 * (scenario["hormuz_disruption"] - base["hormuz_disruption"]))
        frame = reference.copy()
        frame["WTI_Seed"] = paths["Persistence"] + phase * premium
        frame["WTI_Ensemble"] = paths[trained_models["forecast_method"]] + phase * premium
        frame["WTI_Lower"] = frame["WTI_Ensemble"] - uncertainty
        frame["WTI_Upper"] = frame["WTI_Ensemble"] + uncertainty
        frame["GeoRisk"] = risk
        frame["Quarter"] = dates.to_period("Q").astype(str)
        frame["Year"] = dates.year
        all_forecasts[scenario_name] = frame

    # Save all forecasts
    for scenario_name, df_fc in all_forecasts.items():
        safe_name = scenario_name.lower().replace(" ", "_")
        path = f"{DATA_DIR}/forecast_{safe_name}.csv"
        df_fc.to_csv(path)
        print(f"  Saved → {path}")

    combine_forecasts(all_forecasts).to_csv(f"{DATA_DIR}/forecast_combined.csv")
    return all_forecasts


def combine_forecasts(all_forecasts):
    combined = pd.DataFrame(index=all_forecasts["Base"].index)
    for scenario_name, df_fc in all_forecasts.items():
        col_name = scenario_name.replace(" ", "_")
        combined[f"WTI_{col_name}"] = df_fc["WTI_Ensemble"].reindex(combined.index)
        combined[f"Lower_{col_name}"] = df_fc["WTI_Lower"].reindex(combined.index)
        combined[f"Upper_{col_name}"] = df_fc["WTI_Upper"].reindex(combined.index)

    return combined


# ------------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------------

def summarize_forecast(forecasts: dict) -> pd.DataFrame:
    """
    Create a human-readable summary table of the forecast.
    Shows yearly average prices for each scenario.
    """
    rows = []
    for scenario_name, df_fc in forecasts.items():
        for year in df_fc["Year"].unique():
            yr_data = df_fc[df_fc["Year"] == year]["WTI_Ensemble"]
            if len(yr_data) == 0:
                continue
            rows.append({
                "Scenario":   scenario_name,
                "Year":       year,
                "Avg WTI ($)": round(yr_data.mean(), 2),
                "Min ($)":    round(yr_data.min(), 2),
                "Max ($)":    round(yr_data.max(), 2),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{DATA_DIR}/forecast_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    from data_collection     import collect_all_data
    from data_cleaning       import clean_and_merge_data
    from feature_engineering import engineer_features
    from models              import train_all_models

    raw      = collect_all_data()
    clean    = clean_and_merge_data(raw)
    feat, fc = engineer_features(clean)
    results  = train_all_models(feat, fc)

    forecasts = generate_forecasts(clean, fc, results["models"])
    summary   = summarize_forecast(forecasts)
    print("\nForecast Summary:")
    print(summary.to_string(index=False))
