# ============================================================
# models.py - Train, Evaluate and Compare All ML Models
# ============================================================
# Models included:
#   1. Linear Regression   – simple baseline
#   2. Random Forest       – ensemble of decision trees
#   3. XGBoost             – gradient boosted trees (usually best)
#   4. Prophet             – Facebook's time-series model
#   5. ARIMA               – classical statistical time-series model
# ============================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
import os

warnings.filterwarnings("ignore")

from config import DATA_DIR, RANDOM_STATE, TEST_SIZE


# ------------------------------------------------------------------
# Train / Test Split (time-series aware)
# ------------------------------------------------------------------

def split_data(df_feat: pd.DataFrame,
               feature_cols: list,
               target: str = "WTI"):
    """
    Split data into train and test sets.

    IMPORTANT: For time series we NEVER shuffle the data.
    We always use the most recent data as the test set.
    """
    X = df_feat[feature_cols].values
    y = df_feat[target].values

    split_idx = int(len(X) * (1 - TEST_SIZE))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    dates_train = df_feat.index[:split_idx]
    dates_test  = df_feat.index[split_idx:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   model_name: str = "Model") -> dict:
    """
    Calculate standard regression metrics and explain them simply.

    RMSE  = Root Mean Square Error  → average error in $ per barrel
    MAE   = Mean Absolute Error     → typical prediction error
    R²    = R-squared               → % of variance explained (1.0 = perfect)
    MAPE  = Mean Absolute % Error   → error as % of actual price
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        "Model":  model_name,
        "RMSE":   round(rmse, 2),    # $ per barrel
        "MAE":    round(mae, 2),     # $ per barrel
        "R2":     round(r2, 4),      # 0-1, higher is better
        "MAPE":   round(mape, 2),    # %,  lower is better
    }


# ------------------------------------------------------------------
# Model 1: Linear Regression (Baseline)
# ------------------------------------------------------------------

def train_linear_regression(X_train, y_train):
    """
    Ridge Regression: linear model with L2 regularization to prevent
    overfitting when many correlated features are present (like lag features).
    alpha=10 adds regularization penalty that stops the model memorizing training data.
    """
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# Model 2: Random Forest
# ------------------------------------------------------------------

def train_random_forest(X_train, y_train):
    """
    Random Forest: builds 200 decision trees and averages their predictions.
    Good at capturing non-linear relationships (e.g., oil price spikes).

    n_estimators  = number of trees (more = better, slower)
    max_depth     = how deep each tree grows (limits overfitting)
    """
    model = RandomForestRegressor(
        n_estimators  = 200,
        max_depth     = 10,
        min_samples_split = 5,
        random_state  = RANDOM_STATE,
        n_jobs        = -1    # Use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# Model 3: XGBoost
# ------------------------------------------------------------------

def train_xgboost(X_train, y_train):
    """
    XGBoost (Extreme Gradient Boosting): builds trees sequentially,
    each one correcting the errors of the previous one.
    Usually the most accurate model for structured/tabular data.

    learning_rate  = how fast the model learns (lower = more careful)
    n_estimators   = how many trees to build
    max_depth      = complexity of each tree
    """
    model = xgb.XGBRegressor(
        n_estimators    = 300,
        learning_rate   = 0.05,
        max_depth       = 6,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.1,    # L1 regularization (reduces overfitting)
        reg_lambda      = 1.0,    # L2 regularization
        random_state    = RANDOM_STATE,
        verbosity       = 0
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# Model 4: Prophet (Facebook's Time-Series Model)
# ------------------------------------------------------------------

def train_prophet(df_feat: pd.DataFrame, target: str = "WTI"):
    """
    Prophet is designed specifically for time series.
    It automatically handles:
      - Seasonal patterns (monthly, yearly)
      - Holiday effects
      - Trend changes

    Prophet requires input in a specific format:
      ds = date column, y = value column
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("  [WARNING] Prophet not installed. Skipping.")
        return None

    # Prepare Prophet format
    prophet_df = df_feat[[target]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Remove any infinities or nulls
    prophet_df = prophet_df.dropna()
    prophet_df = prophet_df[np.isfinite(prophet_df["y"])]

    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = False,  # Monthly data – no weekly patterns
        daily_seasonality   = False,
        changepoint_prior_scale = 0.3,  # Flexibility in trend changes
        seasonality_prior_scale = 10,
        uncertainty_samples = 300
    )

    # Add monthly seasonality
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(prophet_df)
    return model


# ------------------------------------------------------------------
# Model 5: ARIMA
# ------------------------------------------------------------------

def train_arima(series: pd.Series):
    """
    ARIMA (AutoRegressive Integrated Moving Average):
    A classical statistical time-series model.

    Parameters: ARIMA(p, d, q)
      p = autoregressive order    (how many past values to use)
      d = differencing order      (how many times to difference for stationarity)
      q = moving average order    (past error terms)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools   import adfuller

        # Check stationarity (ARIMA needs stationary data)
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05

        d = 0 if is_stationary else 1

        model = ARIMA(series.dropna(),
                      order=(2, d, 2),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        fitted = model.fit()
        return fitted

    except Exception as exc:
        print(f"  [WARNING] ARIMA failed: {exc}")
        return None


# ------------------------------------------------------------------
# Get Feature Importance
# ------------------------------------------------------------------

def get_feature_importance(model, feature_cols: list,
                            model_type: str = "rf") -> pd.DataFrame:
    """
    Extract which features each model considers most important.
    This tells us: which factors drive oil price the most?
    """
    if model_type == "lr":
        importance = np.abs(model.coef_)
    elif model_type in ("rf", "xgb"):
        importance = model.feature_importances_
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importance,
    }).sort_values("Importance", ascending=False).head(20)

    df_imp["Importance_Pct"] = (
        df_imp["Importance"] / df_imp["Importance"].sum() * 100
    ).round(1)

    return df_imp


# ------------------------------------------------------------------
# Master Training Function
# ------------------------------------------------------------------

def train_all_models(df_feat: pd.DataFrame,
                     feature_cols: list,
                     target: str = "WTI") -> dict:
    """
    Train all models and return them in a dictionary.
    Also evaluates each model on the test set.
    """
    print("\n" + "=" * 60)
    print("   OIL PRICE PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # Split data
    X_tr, X_te, y_tr, y_te, d_tr, d_te = split_data(df_feat,
                                                       feature_cols,
                                                       target)
    print(f"  Train: {len(X_tr)} samples ({d_tr[0].date()} → {d_tr[-1].date()})")
    print(f"  Test:  {len(X_te)} samples ({d_te[0].date()} → {d_te[-1].date()})")

    models    = {}
    metrics   = []
    forecasts = {}   # test-set predictions for each model

    # --- Linear Regression ---
    print("\n  [1/5] Training Linear Regression …")
    lr = train_linear_regression(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    models["LinearRegression"]   = lr
    metrics.append(evaluate_model(y_te, lr_pred, "Linear Regression"))
    forecasts["LinearRegression"] = (d_te, lr_pred, y_te)
    print(f"       RMSE: {metrics[-1]['RMSE']:.2f}  R²: {metrics[-1]['R2']:.4f}")

    # --- Random Forest ---
    print("  [2/5] Training Random Forest …")
    rf = train_random_forest(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    models["RandomForest"]   = rf
    metrics.append(evaluate_model(y_te, rf_pred, "Random Forest"))
    forecasts["RandomForest"] = (d_te, rf_pred, y_te)
    print(f"       RMSE: {metrics[-1]['RMSE']:.2f}  R²: {metrics[-1]['R2']:.4f}")

    # --- XGBoost ---
    print("  [3/5] Training XGBoost …")
    xgb_model = train_xgboost(X_tr, y_tr)
    xgb_pred  = xgb_model.predict(X_te)
    models["XGBoost"]   = xgb_model
    metrics.append(evaluate_model(y_te, xgb_pred, "XGBoost"))
    forecasts["XGBoost"] = (d_te, xgb_pred, y_te)
    print(f"       RMSE: {metrics[-1]['RMSE']:.2f}  R²: {metrics[-1]['R2']:.4f}")

    # --- Prophet ---
    print("  [4/5] Training Prophet …")
    prophet_model = train_prophet(df_feat.loc[d_tr], target)
    if prophet_model:
        try:
            # Predict in-sample + test period
            prophet_df_full = df_feat[[target]].reset_index()
            prophet_df_full.columns = ["ds", "y"]
            future_prophet = prophet_model.make_future_dataframe(
                periods=len(d_te), freq="MS"
            )
            prophet_fc = prophet_model.predict(future_prophet)
            # Take only the last len(d_te) rows that correspond to test period
            prophet_pred = prophet_fc.tail(len(d_te))["yhat"].values
            prophet_pred = np.clip(prophet_pred, 20, 200)
            models["Prophet"]    = prophet_model
            metrics.append(evaluate_model(y_te, prophet_pred, "Prophet"))
            forecasts["Prophet"] = (d_te, prophet_pred, y_te)
            print(f"       RMSE: {metrics[-1]['RMSE']:.2f}  R²: {metrics[-1]['R2']:.4f}")
        except Exception as exc:
            print(f"       [Prophet evaluation error: {exc}]")
    else:
        print("       [SKIPPED – Prophet not installed]")

    # --- ARIMA ---
    print("  [5/5] Training ARIMA …")
    arima_series = df_feat.loc[d_tr, target]
    arima_model  = train_arima(arima_series)
    if arima_model:
        arima_fc = arima_model.forecast(steps=len(d_te))
        arima_pred = arima_fc.values if hasattr(arima_fc, "values") else arima_fc
        models["ARIMA"]    = arima_model
        metrics.append(evaluate_model(y_te, arima_pred, "ARIMA"))
        forecasts["ARIMA"] = (d_te, arima_pred, y_te)
        print(f"       RMSE: {metrics[-1]['RMSE']:.2f}  R²: {metrics[-1]['R2']:.4f}")
    else:
        print("       [SKIPPED – ARIMA error]")

    # --- Summary table ---
    metrics_df = pd.DataFrame(metrics)
    print("\n  Model Comparison:")
    print(metrics_df.to_string(index=False))

    # --- Feature importance (from RF and XGB) ---
    rf_importance  = get_feature_importance(rf,        feature_cols, "rf")
    xgb_importance = get_feature_importance(xgb_model, feature_cols, "xgb")

    rf_importance.to_csv(f"{DATA_DIR}/rf_importance.csv",  index=False)
    xgb_importance.to_csv(f"{DATA_DIR}/xgb_importance.csv", index=False)
    metrics_df.to_csv(f"{DATA_DIR}/model_metrics.csv", index=False)

    print(f"\n  Results saved to {DATA_DIR}/\n")

    return {
        "models":      models,
        "metrics":     metrics_df,
        "forecasts":   forecasts,
        "split":       (X_tr, X_te, y_tr, y_te, d_tr, d_te),
        "importance":  {"rf": rf_importance, "xgb": xgb_importance},
    }


if __name__ == "__main__":
    from data_collection     import collect_all_data
    from data_cleaning       import clean_and_merge_data
    from feature_engineering import engineer_features

    raw      = collect_all_data()
    clean    = clean_and_merge_data(raw)
    feat, fc = engineer_features(clean)
    results  = train_all_models(feat, fc)
