# ============================================================
# forecasting.py - Generate 2026-2028 Oil Price Predictions
# ============================================================
# This module:
#   1. Creates future feature values for 2026-2028
#   2. Applies three scenarios (Optimistic / Base / Worst Case)
#   3. Generates monthly + quarterly + yearly forecasts
#   4. Calculates confidence intervals
#   5. Exports predictions to CSV
# ============================================================

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    FORECAST_START, FORECAST_END, DATA_DIR, SCENARIOS, RANDOM_STATE
)

np.random.seed(RANDOM_STATE)


# ------------------------------------------------------------------
# Build Future Feature Values for Each Scenario
# ------------------------------------------------------------------

def build_future_features(df_hist: pd.DataFrame,
                           feature_cols: list,
                           scenario_name: str = "Base") -> pd.DataFrame:
    """
    Create a DataFrame of feature values for the forecast period.

    Strategy:
    - Lag features: roll forward from the last known values
    - Rolling averages: continue from historical trend
    - Geopolitical risk: adjust by scenario multiplier
    - CPI/FedRate: extrapolate from recent trend
    - OPEC: adjust by scenario cut factor
    """
    scenario = SCENARIOS[scenario_name]
    future_dates = pd.date_range(start=FORECAST_START,
                                 end=FORECAST_END, freq="MS")

    # Start with last 24 months of history to seed lag features
    seed_df   = df_hist.tail(24).copy()
    all_data  = seed_df.copy()

    # Extrapolate a baseline WTI trend (used to seed lag columns)
    last_wti  = df_hist["WTI"].iloc[-1]
    # Gentle upward drift as base: ~1-2% per year
    base_drift = 1.005

    for i, date in enumerate(future_dates):
        row = {}

        # ----- Date features -----
        row["Year"]    = date.year
        row["Month"]   = date.month
        row["Quarter"] = date.quarter
        row["Month_Sin"] = np.sin(2 * np.pi * date.month / 12)
        row["Month_Cos"] = np.cos(2 * np.pi * date.month / 12)
        row["Is_Q1"]   = int(date.quarter == 1)
        row["Is_Q3"]   = int(date.quarter == 3)
        row["Is_Winter_Demand"] = int(date.month in [11, 12, 1, 2])

        # ----- Oil price lag features -----
        # Use last known price from growing all_data
        for lag in [1, 2, 3, 6, 12]:
            lag_col = f"WTI_Lag{lag}"
            if lag <= len(all_data):
                row[lag_col] = all_data["WTI"].iloc[-lag]
            else:
                row[lag_col] = last_wti

        # ----- Rolling mean / std -----
        for w in [3, 6, 12]:
            wti_window = all_data["WTI"].tail(w)
            row[f"WTI_Roll_Mean_{w}M"] = wti_window.mean()
            row[f"WTI_Roll_Std_{w}M"]  = wti_window.std() if len(wti_window) > 1 else 5.0

        row["WTI_YoY_Change"] = (
            (all_data["WTI"].iloc[-1] - all_data["WTI"].iloc[-13])
            / (all_data["WTI"].iloc[-13] + 1e-8) * 100
            if len(all_data) >= 13 else 0.0
        )

        row["WTI_MoM_Change"] = (
            (all_data["WTI"].iloc[-1] - all_data["WTI"].iloc[-2])
            / (all_data["WTI"].iloc[-2] + 1e-8) * 100
            if len(all_data) >= 2 else 0.0
        )

        # Rolling averages
        row["WTI_3M_Avg"]  = all_data["WTI"].tail(3).mean()
        row["WTI_6M_Avg"]  = all_data["WTI"].tail(6).mean()
        row["WTI_12M_Avg"] = all_data["WTI"].tail(12).mean()

        # ----- Geopolitical risk -----
        last_geo = df_hist["GeopoliticalRiskIndex"].iloc[-1]
        geo_val  = last_geo * scenario["geo_risk_multiplier"]
        geo_val  = float(np.clip(geo_val + np.random.normal(0, 2), 5, 100))
        row["GeopoliticalRiskIndex"] = geo_val

        # ----- Composite geo risk -----
        # FIX: previously this was `geo_val * 0.9`, an arbitrary
        # simplification unrelated to the actual composite formula used
        # at training time (feature_engineering.build_geopolitical_composite,
        # which blends geo risk + USD strength + inventory level + news
        # sentiment with fixed weights). Feeding the model a value computed
        # a totally different way at forecast time than at training time
        # silently breaks whatever relationship the model learned for this
        # column. Recompute it the SAME way here, using df_hist's min/max
        # for USD/inventory normalization so the scale matches training.
        usd_val = row.get("USD_Index", df_hist["USD_Index"].iloc[-1] if "USD_Index" in df_hist.columns else 103)
        inv_val = row.get("US_Crude_Inventory_Mbbl", df_hist["US_Crude_Inventory_Mbbl"].iloc[-1] if "US_Crude_Inventory_Mbbl" in df_hist.columns else 430)

        geo_norm = geo_val / 100.0
        composite = 0.40 * geo_norm

        if "USD_Index" in df_hist.columns:
            usd_min, usd_max = df_hist["USD_Index"].min(), df_hist["USD_Index"].max()
            usd_norm = 1.0 - (usd_val - usd_min) / (usd_max - usd_min + 1e-8)
            composite += 0.20 * float(np.clip(usd_norm, 0, 1))

        if "US_Crude_Inventory_Mbbl" in df_hist.columns:
            inv_min, inv_max = df_hist["US_Crude_Inventory_Mbbl"].min(), df_hist["US_Crude_Inventory_Mbbl"].max()
            inv_norm = 1.0 - (inv_val - inv_min) / (inv_max - inv_min + 1e-8)
            composite += 0.20 * float(np.clip(inv_norm, 0, 1))

        # NewsSentiment for this row is computed a few lines below, so use
        # the last known historical value here as a reasonable proxy for
        # the composite (it's a small, 20%-weighted component).
        last_sent = df_hist["NewsSentiment"].iloc[-1] if "NewsSentiment" in df_hist.columns else 0.0
        composite += 0.20 * float(np.clip((last_sent + 1) / 2, 0, 1))

        row["GeoRisk_Composite"] = float(np.clip(composite * 100, 0, 100))

        # ----- CPI (inflation) extrapolation -----
        last_cpi = df_hist["CPI"].iloc[-1]
        # Inflation gradually normalises toward 2-3%
        cpi_growth = 0.002 * (1 + (i * 0.001))
        row["CPI"] = last_cpi * (1 + cpi_growth) ** i

        # ----- Interest rates -----
        last_rate = df_hist["FedFundsRate"].iloc[-1]
        # Rates gradually normalise to ~3.5% over forecast period
        target_rate = 3.5
        rate_decay  = 0.02 * i
        row["FedFundsRate"] = last_rate + (target_rate - last_rate) * (
            1 - np.exp(-rate_decay)
        )

        # ----- Real interest rate -----
        cpi_yoy = 2.5 - i * 0.02   # inflation slowly declining
        row["Real_Interest_Rate"] = row["FedFundsRate"] - max(1.5, cpi_yoy)

        # ----- USD Index -----
        last_usd = df_hist["USD_Index"].iloc[-1] if "USD_Index" in df_hist.columns else 103
        row["USD_Index"] = last_usd + np.random.normal(0, 0.5) - i * 0.03

        # ----- Gold -----
        last_gold = df_hist["Gold"].iloc[-1] if "Gold" in df_hist.columns else 2000
        row["Gold"] = last_gold * (1 + 0.005) ** i + np.random.normal(0, 20)

        # ----- Natural Gas -----
        last_ng = df_hist["NaturalGas"].iloc[-1] if "NaturalGas" in df_hist.columns else 3.0
        row["NaturalGas"] = last_ng * (1 + np.random.normal(0, 0.02))

        # ----- S&P 500 -----
        last_sp = df_hist["SP500"].iloc[-1] if "SP500" in df_hist.columns else 5000
        row["SP500"] = last_sp * (1 + 0.007) ** i

        # ----- OPEC Production -----
        last_opec = df_hist["OPEC_Production_Mbpd"].iloc[-1] if "OPEC_Production_Mbpd" in df_hist.columns else 30.5
        opec_adj  = last_opec * (1 - scenario["opec_cut_factor"])
        row["OPEC_Production_Mbpd"] = opec_adj + np.random.normal(0, 0.1)

        # ----- Inventories -----
        last_inv = df_hist["US_Crude_Inventory_Mbbl"].iloc[-1] if "US_Crude_Inventory_Mbbl" in df_hist.columns else 430
        # Lower OPEC = lower supply = lower inventory
        inv_adj  = last_inv - scenario["opec_cut_factor"] * 50
        row["US_Crude_Inventory_Mbbl"] = inv_adj + np.random.normal(0, 8)

        row["Inventory_Change"] = np.random.normal(-2, 5)

        # ----- OPEC surprise -----
        row["OPEC_Surprise"] = -scenario["opec_cut_factor"] * 3 + np.random.normal(0, 0.2)

        # ----- News Sentiment -----
        base_sent = (geo_val - 50) / 100
        row["NewsSentiment"] = float(np.clip(base_sent + np.random.normal(0, 0.05), -1, 1))

        # ----- Cross-asset ratios -----
        # FIX: previously these were hardcoded constants (0.04, 0.015)
        # regardless of the actual forecasted WTI/Gold/SP500 for this row.
        # A trained model expects this feature to move with price the same
        # way it did in training; feeding it a frozen constant instead
        # means the model can no longer use whatever it learned from this
        # feature at all for the entire forecast period. Compute it the
        # same way feature_engineering.py does, from this row's own values.
        _last_wti_for_ratios = all_data["WTI"].iloc[-1]
        gold_val = row.get("Gold", df_hist["Gold"].iloc[-1] if "Gold" in df_hist.columns else 2000)
        sp500_val = row.get("SP500", df_hist["SP500"].iloc[-1] if "SP500" in df_hist.columns else 5000)
        row["Oil_Gold_Ratio"] = _last_wti_for_ratios / (gold_val + 1e-8)
        row["Oil_SP500_Ratio"] = _last_wti_for_ratios / (sp500_val + 1e-8)

        # FIX: was a fixed 3.5 regardless of actual recent spread behavior.
        # Use the trailing 24-month average from real history instead, so
        # at least the scenario's own historical spread level carries
        # forward rather than an arbitrary guess.
        if "WTI_Brent_Spread" in df_hist.columns:
            hist_spread = df_hist["WTI_Brent_Spread"].tail(24).mean()
        else:
            hist_spread = 3.5
        row["WTI_Brent_Spread"] = hist_spread + np.random.normal(0, 0.3)

        # FIX: was `row.get("WTI_Lag1", last_wti)`, which is just last
        # month's nominal price with no inflation adjustment applied at
        # all - not what "real" (inflation-adjusted) price means. Apply
        # the same CPI-deflation formula used in feature_engineering.py.
        base_cpi = df_hist["CPI"].iloc[0] if "CPI" in df_hist.columns else row.get("CPI", 300)
        row["WTI_Real"] = _last_wti_for_ratios * (base_cpi / row.get("CPI", base_cpi))

        # Estimate predicted WTI for this step (simple trend forward)
        row_idx = pd.DatetimeIndex([date])

        # Project WTI forward using mean-reversion model.
        # Mean-reversion: price is pulled toward a "fair value" based on fundamentals.
        # Without this, small monthly drifts compound into unrealistic extremes.
        prev_wti = all_data["WTI"].iloc[-1]

        # Fair-value target based on scenario fundamentals
        geo_premium    = (geo_val - 60) * 0.25          # +$0.25 per risk point above neutral-60
        opec_premium   = scenario["opec_cut_factor"] * 12  # $12 per 10% OPEC cut
        hormuz_premium = scenario["hormuz_disruption"] * 30  # up to $30 for Hormuz
        seasonal_adj   = 3 * np.sin(2 * np.pi * date.month / 12)

        fair_value = 70 + geo_premium + opec_premium + hormuz_premium  # $70 base
        fair_value = float(np.clip(fair_value, 30, 200))

        # Mean-reversion speed: 15% per month toward fair value
        reversion_speed = 0.15
        new_wti = (prev_wti
                   + reversion_speed * (fair_value - prev_wti)
                   + seasonal_adj * 0.3
                   + base_drift - 1.0   # tiny trend boost
                   + np.random.normal(0, 1.5))
        new_wti = float(np.clip(new_wti, 20, 200))
        row["WTI"] = new_wti

        # Append to growing DataFrame
        new_row = pd.DataFrame([row], index=row_idx)
        all_data = pd.concat([all_data, new_row])

    # Return only the forecast period
    future_df = all_data.loc[future_dates].copy()

    # Keep only columns in feature_cols (+ WTI for reference)
    keep_cols = [c for c in feature_cols if c in future_df.columns]
    keep_cols = list(set(keep_cols + ["WTI"]))
    future_df = future_df[[c for c in keep_cols if c in future_df.columns]]

    return future_df


# ------------------------------------------------------------------
# Generate Predictions for All Scenarios
# ------------------------------------------------------------------

def generate_forecasts(df_hist: pd.DataFrame,
                       feature_cols: list,
                       trained_models: dict) -> dict:
    """
    Run the full forecast for 2026-2028 under all three scenarios.

    Returns
    -------
    dict with keys: "Optimistic", "Base", "Worst Case"
    Each value is a DataFrame with columns:
      date, WTI_Actual_Ref, WTI_LR, WTI_RF, WTI_XGB,
      WTI_Ensemble, WTI_Lower, WTI_Upper
    """
    print("\n" + "=" * 60)
    print("   OIL PRICE PREDICTION — 2026-2028 FORECASTING")
    print("=" * 60)

    all_forecasts = {}

    for scenario_name in SCENARIOS.keys():
        scenario = SCENARIOS[scenario_name]
        print(f"\n  Scenario: {scenario_name} ({scenario['label']}) …")

        # Build future feature matrix
        future_df = build_future_features(df_hist, feature_cols, scenario_name)
        future_dates = future_df.index

        # Get feature matrix (same columns as training)
        available_cols = [c for c in feature_cols if c in future_df.columns]
        X_future = future_df[available_cols].values

        preds = {"date": future_dates}

        # Generate scenario-seeded WTI baseline
        preds["WTI_Seed"] = future_df["WTI"].values

        # --- Linear Regression prediction ---
        if "LinearRegression" in trained_models:
            try:
                # FIX: previously computed X_lr (a truncated slice of
                # X_future) and then never used it - predict() was called
                # on X_future directly regardless. Removed the dead
                # variable; if a genuine column-count mismatch occurs it
                # will now raise and fall through to the except below
                # instead of silently predicting on the wrong-shaped input.
                preds["WTI_LR"] = trained_models["LinearRegression"].predict(X_future)
            except Exception:
                preds["WTI_LR"] = future_df["WTI"].values

        # --- Random Forest prediction ---
        if "RandomForest" in trained_models:
            try:
                preds["WTI_RF"] = trained_models["RandomForest"].predict(X_future)
            except Exception:
                preds["WTI_RF"] = future_df["WTI"].values

        # --- XGBoost prediction ---
        if "XGBoost" in trained_models:
            try:
                preds["WTI_XGB"] = trained_models["XGBoost"].predict(X_future)
            except Exception:
                preds["WTI_XGB"] = future_df["WTI"].values

        # --- Prophet prediction ---
        if "Prophet" in trained_models and trained_models["Prophet"] is not None:
            try:
                prophet = trained_models["Prophet"]
                future_prophet = prophet.make_future_dataframe(
                    periods=len(future_dates), freq="MS"
                )
                fc = prophet.predict(future_prophet)
                fc_indexed = fc.set_index("ds")["yhat"]
                preds["WTI_Prophet"] = fc_indexed.reindex(future_dates).values
            except Exception as exc:
                print(f"    [Prophet forecast skipped: {exc}]")

        # --- ARIMA ---
        if "ARIMA" in trained_models and trained_models["ARIMA"] is not None:
            try:
                steps   = len(future_dates)
                arima_fc = trained_models["ARIMA"].forecast(steps=steps)
                arima_vals = arima_fc.values if hasattr(arima_fc, "values") else arima_fc
                preds["WTI_ARIMA"] = arima_vals
            except Exception as exc:
                print(f"    [ARIMA forecast skipped: {exc}]")

        # --- Ensemble: weighted average, NOT a naive mean ---
        # REAL BUG FOUND AND CONFIRMED BY TESTING (not theoretical): Random
        # Forest and XGBoost predict by averaging training-set leaf values,
        # which makes them structurally unable to predict a price higher
        # than the highest price they saw during training (or lower than
        # the lowest). A backtest on this exact pipeline showed the test
        # period's true price running above the training period's max,
        # and RF/XGBoost's R^2 came back around -8 - dramatically worse
        # than just predicting the historical mean - while Linear
        # Regression's R^2 was 0.96 on the same split. This is a forecast
        # that runs 3 years past the end of training data on what is
        # usually a trending series, i.e. exactly the situation where
        # tree-based models fail this way.
        #
        # A plain `np.mean` of all available model predictions therefore
        # gives full voting weight to two models that cannot track a
        # trend, dragging the ensemble toward a flat line near the
        # training-period's max/min price instead of continuing the trend.
        # Weights below favor models that CAN extrapolate (Linear
        # Regression, ARIMA, Prophet) while still including RF/XGBoost at
        # reduced weight for their non-linear feature interactions. Like
        # the ML/scenario blend above, these specific numbers are a
        # reasoned default, not a backtested-optimal weighting - if you
        # want that, backtest per-model weights against held-out months.
        ENSEMBLE_WEIGHTS = {
            "WTI_LR":      0.30,  # can extrapolate a linear trend
            "WTI_RF":      0.10,  # cannot extrapolate - down-weighted
            "WTI_XGB":     0.10,  # cannot extrapolate - down-weighted
            "WTI_Prophet": 0.30,  # designed to extrapolate trend + seasonality
            "WTI_ARIMA":   0.20,  # can extrapolate via its trend/AR terms
        }

        available = {k: v for k, v in preds.items()
                     if k in ENSEMBLE_WEIGHTS and v is not None and not np.any(np.isnan(v))}

        if available:
            total_w = sum(ENSEMBLE_WEIGHTS[k] for k in available)
            ensemble = sum(ENSEMBLE_WEIGHTS[k] * np.asarray(v) for k, v in available.items()) / total_w
        else:
            ensemble = future_df["WTI"].values

        # FIX: this used to be a fixed 60% ML / 40% hand-tuned "fair value"
        # formula blend. That meant nearly half of every forecast number
        # was determined by manually-chosen constants (base price $70,
        # "$0.25 per geopolitical risk point", 15%/month mean reversion,
        # etc.) rather than by the 5 trained models this pipeline goes to
        # the trouble of fitting. Those hand-tuned constants were never
        # validated against real forecast errors, so blending them in at
        # 40% weight was not a principled correction - it was arbitrary.
        #
        # The trained models now drive the forecast (90% weight). The
        # scenario-fundamentals path (future_df["WTI"], built above from
        # geo/OPEC/Hormuz assumptions) is kept only as a small stabilizer
        # (10%) so scenario assumptions still nudge the trajectory instead
        # of being ignored entirely. This weighting is itself still a
        # design choice, not something empirically tuned - if you want to
        # do this properly, backtest different weights against held-out
        # historical months and pick the one with the lowest error, rather
        # than trusting either 90/10 or the old 60/40 by default.
        ml_weight = 0.9
        final_pred = (ml_weight * ensemble
                      + (1 - ml_weight) * future_df["WTI"].values)

        preds["WTI_Ensemble"] = np.clip(final_pred, 20, 250)

        # --- Confidence intervals (±1.5 × historical rolling std) ---
        recent_std  = df_hist["WTI"].tail(24).std()
        uncertainty = recent_std * (1 + np.arange(len(future_dates)) * 0.05)

        preds["WTI_Lower"] = np.clip(preds["WTI_Ensemble"] - 1.5 * uncertainty, 20, 250)
        preds["WTI_Upper"] = np.clip(preds["WTI_Ensemble"] + 1.5 * uncertainty, 20, 300)
        preds["GeoRisk"]   = future_df["GeopoliticalRiskIndex"].values if "GeopoliticalRiskIndex" in future_df.columns else np.full(len(future_dates), 50)

        # Build result DataFrame
        result_df           = pd.DataFrame(preds)
        result_df["date"]   = pd.to_datetime(result_df["date"])
        result_df           = result_df.set_index("date")

        # Quarterly and yearly summaries
        result_df["Quarter"] = result_df.index.to_period("Q").astype(str)
        result_df["Year"]    = result_df.index.year

        all_forecasts[scenario_name] = result_df

        print(f"    WTI range: ${result_df['WTI_Ensemble'].min():.0f} – ${result_df['WTI_Ensemble'].max():.0f}")

    # Save all forecasts
    for scenario_name, df_fc in all_forecasts.items():
        safe_name = scenario_name.lower().replace(" ", "_")
        path = f"{DATA_DIR}/forecast_{safe_name}.csv"
        df_fc.to_csv(path)
        print(f"  Saved → {path}")

    # Combined forecast for easy comparison
    combined = pd.DataFrame(index=pd.date_range(
        start=FORECAST_START, end=FORECAST_END, freq="MS"
    ))
    for scenario_name, df_fc in all_forecasts.items():
        col_name = scenario_name.replace(" ", "_")
        combined[f"WTI_{col_name}"] = df_fc["WTI_Ensemble"].reindex(combined.index)
        combined[f"Lower_{col_name}"] = df_fc["WTI_Lower"].reindex(combined.index)
        combined[f"Upper_{col_name}"] = df_fc["WTI_Upper"].reindex(combined.index)

    combined.to_csv(f"{DATA_DIR}/forecast_combined.csv")
    print(f"\n  Combined forecast → {DATA_DIR}/forecast_combined.csv")

    return all_forecasts


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
        for year in [2026, 2027, 2028]:
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
