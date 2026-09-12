# ============================================================
# feature_engineering.py - Build Model Features
# ============================================================
# What this file does:
#   1. Create lag features (past prices as predictors)
#   2. Create rolling statistics (moving averages, volatility)
#   3. Build the Geopolitical Risk Index components
#   4. Add seasonal and cyclical features
#   5. Prepare final feature matrix (X) and target (y)
# ============================================================

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR


# ------------------------------------------------------------------
# 1. Lag Features
# ------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame,
                     target_col: str = "WTI",
                     lags: list = None) -> pd.DataFrame:
    """
    Lag features use past values to predict the future.
    Example: WTI_Lag1 = last month's price, WTI_Lag3 = price 3 months ago.

    This is essential for time-series models because yesterday's price
    is usually the strongest predictor of today's price.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12]   # 1, 2, 3, 6, 12 months back

    for lag in lags:
        col_name = f"{target_col}_Lag{lag}"
        df[col_name] = df[target_col].shift(lag)

    return df


# ------------------------------------------------------------------
# 2. Rolling Statistics
# ------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame,
                         target_col: str = "WTI",
                         windows: list = None) -> pd.DataFrame:
    """
    Rolling statistics capture trends and volatility over time.
      - Rolling mean: smoothed price trend
      - Rolling std:  price volatility (higher = more uncertain)
    """
    if windows is None:
        windows = [3, 6, 12]  # 3-month, 6-month, 12-month windows

    for w in windows:
        df[f"{target_col}_Roll_Mean_{w}M"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_Roll_Std_{w}M"]  = df[target_col].rolling(w).std()

    # Price momentum: current vs 12-months-ago (year-on-year change)
    df[f"{target_col}_YoY_Change"] = (
        (df[target_col] - df[target_col].shift(12)) /
        df[target_col].shift(12) * 100
    )

    return df


# ------------------------------------------------------------------
# 3. Geopolitical Risk Composite Index
# ------------------------------------------------------------------

def build_geopolitical_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple risk factors into one composite Geopolitical Risk Score.

    Components (if available):
      - Raw GeopoliticalRiskIndex    (40% weight)
      - USD Index inverse            (20% weight – strong USD = bearish oil)
      - Inventory level inverse      (20% weight – high inventory = bearish)
      - News Sentiment               (20% weight)

    Final score: 0-100, where higher = more bullish for oil price.
    """
    geo_weight  = 0.40
    usd_weight  = 0.20
    inv_weight  = 0.20
    sent_weight = 0.20

    composite = pd.Series(0.0, index=df.index)

    # Component 1: Base geopolitical risk
    if "GeopoliticalRiskIndex" in df.columns:
        geo_norm = df["GeopoliticalRiskIndex"] / 100.0
        composite += geo_weight * geo_norm

    # Component 2: USD strength (inverse – strong dollar depresses oil)
    if "USD_Index" in df.columns:
        usd_norm = 1.0 - (df["USD_Index"] - df["USD_Index"].min()) / (
            df["USD_Index"].max() - df["USD_Index"].min() + 1e-8
        )
        composite += usd_weight * usd_norm

    # Component 3: Low inventory = supply tightness = bullish oil
    if "US_Crude_Inventory_Mbbl" in df.columns:
        inv_norm = 1.0 - (df["US_Crude_Inventory_Mbbl"] - df["US_Crude_Inventory_Mbbl"].min()) / (
            df["US_Crude_Inventory_Mbbl"].max() - df["US_Crude_Inventory_Mbbl"].min() + 1e-8
        )
        composite += inv_weight * inv_norm

    # Component 4: Bullish news sentiment
    if "NewsSentiment" in df.columns:
        sent_norm = (df["NewsSentiment"] + 1) / 2   # -1..+1 → 0..1
        composite += sent_weight * sent_norm

    df["GeoRisk_Composite"] = (composite * 100).clip(0, 100)

    return df


# ------------------------------------------------------------------
# 4. Seasonal / Cyclical Features
# ------------------------------------------------------------------

def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode month and quarter as cyclical (sine/cosine) features.
    This prevents the model from treating December and January as far apart
    (they are adjacent months but month 12 and 1 are numerically distant).
    """
    # Cyclical month encoding
    df["Month_Sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Season flags (0/1)
    df["Is_Q1"] = (df.index.quarter == 1).astype(int)  # Jan-Mar: demand low
    df["Is_Q3"] = (df.index.quarter == 3).astype(int)  # Jul-Sep: driving season

    # US winter heating demand peak (Nov-Feb)
    df["Is_Winter_Demand"] = df.index.month.isin([11, 12, 1, 2]).astype(int)

    return df


# ------------------------------------------------------------------
# 5. Cross-Asset Interaction Features
# ------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-asset interactions capture relationships between variables:
      - Oil/Gold ratio   → tracks risk-off sentiment
      - Oil/SP500 ratio  → energy sector relative performance
      - Real rate        → Fed rate minus inflation (negative = bullish oil)
    """
    if "WTI" in df.columns and "Gold" in df.columns:
        df["Oil_Gold_Ratio"] = df["WTI"] / (df["Gold"] + 1e-8)

    if "WTI" in df.columns and "SP500" in df.columns:
        df["Oil_SP500_Ratio"] = df["WTI"] / (df["SP500"] + 1e-8)

    if "FedFundsRate" in df.columns and "CPI" in df.columns:
        # YoY CPI change as inflation rate
        cpi_yoy = df["CPI"].pct_change(12) * 100
        df["Real_Interest_Rate"] = df["FedFundsRate"] - cpi_yoy

    return df


# ------------------------------------------------------------------
# 6. OPEC Surprise Feature
# ------------------------------------------------------------------

def add_opec_surprise(df: pd.DataFrame) -> pd.DataFrame:
    """
    OPEC surprise = actual production change vs expected (trend).
    A surprise production cut → bullish for prices.
    A surprise increase → bearish.
    """
    if "OPEC_Production_Mbpd" not in df.columns:
        return df

    # Expected = 12-month rolling average
    expected = df["OPEC_Production_Mbpd"].rolling(12).mean()
    df["OPEC_Surprise"] = df["OPEC_Production_Mbpd"] - expected
    # Negative = cut relative to trend = bullish

    return df


# ------------------------------------------------------------------
# Master Feature Engineering Pipeline
# ------------------------------------------------------------------

def engineer_features(df: pd.DataFrame,
                      target: str = "WTI") -> tuple:
    """
    Run the complete feature engineering pipeline.

    Returns
    -------
    df_feat : DataFrame with all features added
    feature_cols : list of feature column names to use as model input X
    """
    print("\n" + "=" * 60)
    print("   OIL PRICE PREDICTION — FEATURE ENGINEERING")
    print("=" * 60)

    df_feat = df.copy()

    # Add all feature types
    print("  Adding lag features …")
    df_feat = add_lag_features(df_feat, target_col=target)

    print("  Adding rolling statistics …")
    df_feat = add_rolling_features(df_feat, target_col=target)

    print("  Building geopolitical composite …")
    df_feat = build_geopolitical_composite(df_feat)

    print("  Adding seasonal features …")
    df_feat = add_seasonal_features(df_feat)

    print("  Adding cross-asset interactions …")
    df_feat = add_interaction_features(df_feat)

    print("  Adding OPEC surprise …")
    df_feat = add_opec_surprise(df_feat)

    # Drop rows where lag features create NaN (first 12 rows)
    df_feat.dropna(subset=[f"{target}_Lag12"], inplace=True)

    # Define which columns to use as features (X)
    # We exclude the raw target and purely descriptive columns
    exclude_from_x = [target, "Brent", "WTI_Real",
                      "Gasoline_ETF", "Year", "Quarter"]

    feature_cols = [
        c for c in df_feat.columns
        if c not in exclude_from_x
        and df_feat[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        and not df_feat[c].isna().any()
    ]

    print(f"  Total features created: {len(feature_cols)}")
    print(f"  Training samples:       {len(df_feat)}")

    # Save feature dataset
    df_feat.to_csv(f"{DATA_DIR}/features.csv")
    print(f"  Saved → {DATA_DIR}/features.csv\n")

    return df_feat, feature_cols


def get_feature_importance_labels() -> dict:
    """
    Human-readable descriptions for each feature category.
    Used in the dashboard's feature importance chart.
    """
    return {
        "WTI_Lag":           "Past Oil Price (Momentum)",
        "WTI_Roll_Mean":     "Price Trend (Moving Avg)",
        "WTI_Roll_Std":      "Price Volatility",
        "WTI_YoY":           "Year-over-Year Change",
        "GeoRisk_Composite": "Geopolitical Risk Index",
        "GeopoliticalRisk":  "Raw Geopolitical Risk",
        "CPI":               "Inflation (CPI)",
        "FedFundsRate":      "Interest Rates",
        "OPEC_Production":   "OPEC Production Level",
        "OPEC_Surprise":     "OPEC Supply Surprise",
        "US_Crude_Inventory":"US Crude Inventories",
        "Inventory_Change":  "Inventory Build/Draw",
        "NewsSentiment":     "News Sentiment",
        "Real_Interest_Rate":"Real Interest Rate",
        "Oil_Gold_Ratio":    "Oil/Gold Ratio",
        "USD_Index":         "US Dollar Strength",
        "NaturalGas":        "Natural Gas Prices",
        "Gold":              "Gold Prices",
        "SP500":             "S&P 500 (Economy)",
        "Month_Sin":         "Seasonal Pattern",
        "Month_Cos":         "Seasonal Pattern",
        "Is_Winter_Demand":  "Winter Demand Season",
    }


if __name__ == "__main__":
    from data_collection import collect_all_data
    from data_cleaning import clean_and_merge_data

    raw  = collect_all_data()
    clean = clean_and_merge_data(raw)
    df_feat, feat_cols = engineer_features(clean)

    print("Top features (first 10):")
    for f in feat_cols[:10]:
        print(f"  {f}")
