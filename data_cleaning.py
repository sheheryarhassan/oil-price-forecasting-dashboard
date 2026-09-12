# ============================================================
# data_cleaning.py - Merge, Clean and Prepare All Data
# ============================================================
# Steps:
#   1. Merge all data sources on a common monthly date index
#   2. Handle missing values
#   3. Normalize numeric features
#   4. Convert dates and types correctly
# ============================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

from config import START_DATE, END_DATE, DATA_DIR


def load_all_csvs() -> dict:
    """Load all saved CSV files from the data directory."""
    files = {
        "market":    f"{DATA_DIR}/market_data.csv",
        "economic":  f"{DATA_DIR}/economic_data.csv",
        "geo":       f"{DATA_DIR}/geopolitical_data.csv",
        "opec":      f"{DATA_DIR}/opec_data.csv",
        "inventory": f"{DATA_DIR}/inventory_data.csv",
        "sentiment": f"{DATA_DIR}/sentiment_data.csv",
    }
    data = {}
    for key, path in files.items():
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            data[key] = df
        except FileNotFoundError:
            print(f"  [WARNING] {path} not found – run data_collection.py first")
    return data


def merge_all_data(data: dict) -> pd.DataFrame:
    """
    Merge all individual DataFrames into one master DataFrame.

    All data is resampled to monthly frequency (MS = month start).
    This aligns different sources (daily, weekly, monthly).
    """
    # Common monthly date range
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    master = pd.DataFrame(index=date_range)

    for key, df in data.items():
        if df is None or df.empty:
            continue
        # Resample to monthly if needed
        if df.index.freq != "MS":
            df = df.resample("MS").mean(numeric_only=True)
        # Merge on index (date)
        master = master.join(df, how="left")

    return master


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using multiple strategies:
      - Forward fill for price series (last known value carries forward)
      - Backward fill for any remaining gaps at the start
      - Interpolation for smooth time-series data
    """
    print("  Handling missing values …")

    original_nulls = df.isna().sum().sum()

    # Step 1: Forward fill (use previous month's value)
    df = df.ffill()

    # Step 2: Backward fill (for leading NaN at start)
    df = df.bfill()

    # Step 3: If still missing, interpolate linearly
    df = df.interpolate(method="linear")

    remaining_nulls = df.isna().sum().sum()
    print(f"    Nulls: {original_nulls} → {remaining_nulls}")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing columns:
      - WTI/Brent spread     → market premium signal
      - Oil price % change   → momentum
      - Inflation-adjusted oil price → real price
      - Inventory change     → supply pressure
    """
    print("  Adding derived features …")

    # WTI-Brent spread (usually Brent > WTI by ~$3-5)
    if "WTI" in df.columns and "Brent" in df.columns:
        df["WTI_Brent_Spread"] = df["Brent"] - df["WTI"]

    # Month-over-month % change in oil price
    if "WTI" in df.columns:
        df["WTI_MoM_Change"] = df["WTI"].pct_change() * 100
        df["WTI_3M_Avg"]     = df["WTI"].rolling(3).mean()
        df["WTI_6M_Avg"]     = df["WTI"].rolling(6).mean()
        df["WTI_12M_Avg"]    = df["WTI"].rolling(12).mean()

    # Real (inflation-adjusted) oil price
    if "WTI" in df.columns and "CPI" in df.columns:
        base_cpi = df["CPI"].iloc[0]   # CPI at start of series = baseline
        df["WTI_Real"] = df["WTI"] * (base_cpi / df["CPI"])

    # Inventory change (draw or build)
    if "US_Crude_Inventory_Mbbl" in df.columns:
        df["Inventory_Change"] = df["US_Crude_Inventory_Mbbl"].diff()

    # Year and month columns (useful as model features)
    df["Year"]  = df.index.year
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter

    return df


def remove_outliers(df: pd.DataFrame,
                    columns: list = None,
                    z_threshold: float = 4.0) -> pd.DataFrame:
    """
    Replace extreme outliers (|z-score| > threshold) with rolling median.
    Uses a high threshold (4.0) to preserve real events like COVID crash.
    """
    if columns is None:
        columns = ["WTI", "Brent"]

    for col in columns:
        if col not in df.columns:
            continue
        z = (df[col] - df[col].mean()) / df[col].std()
        outlier_mask = z.abs() > z_threshold
        if outlier_mask.sum() > 0:
            rolling_median = df[col].rolling(5, center=True, min_periods=1).median()
            df.loc[outlier_mask, col] = rolling_median[outlier_mask]
            print(f"    Outliers capped in {col}: {outlier_mask.sum()} values")

    return df


def normalize_features(df: pd.DataFrame,
                       exclude_cols: list = None):
    """
    Scale numeric features to 0-1 range using MinMaxScaler.
    Returns (normalized_df, scaler_dict) so we can reverse-transform later.

    Exclude columns like WTI (target) and date parts from scaling.
    """
    if exclude_cols is None:
        exclude_cols = ["WTI", "Brent", "Year", "Month", "Quarter"]

    scaler_dict = {}
    df_norm = df.copy()

    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            continue

        scaler = MinMaxScaler()
        valid = df[[col]].dropna()
        if len(valid) < 5:
            continue

        scaler.fit(valid)
        df_norm.loc[valid.index, col] = scaler.transform(valid).flatten()
        scaler_dict[col] = scaler

    return df_norm, scaler_dict


def clean_and_merge_data(data: dict = None,
                         normalize: bool = False) -> pd.DataFrame:
    """
    Master cleaning pipeline. Call this from other modules.

    Parameters
    ----------
    data : dict of DataFrames (from collect_all_data) or None to load CSVs
    normalize : if True, scale features to 0-1

    Returns
    -------
    Cleaned, merged DataFrame ready for feature engineering & modelling
    """
    print("\n" + "=" * 60)
    print("   OIL PRICE PREDICTION — DATA CLEANING")
    print("=" * 60)

    # Load data
    if data is None:
        data = load_all_csvs()

    # Merge
    print("  Merging data sources …")
    df = merge_all_data(data)
    print(f"    Merged shape: {df.shape}")

    # Handle nulls
    df = handle_missing_values(df)

    # Remove extreme outliers
    df = remove_outliers(df)

    # Add derived features
    df = add_derived_features(df)

    # Final cleanup: drop rows with too many nulls
    df.dropna(thresh=int(df.shape[1] * 0.5), inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Optional normalization
    if normalize:
        df, _ = normalize_features(df)

    print(f"  Final clean shape: {df.shape}")
    print("  Cleaning complete!\n")

    # Save cleaned data
    df.to_csv(f"{DATA_DIR}/cleaned_data.csv")
    print(f"  Saved → {DATA_DIR}/cleaned_data.csv")

    return df


if __name__ == "__main__":
    # Test standalone
    from data_collection import collect_all_data
    raw = collect_all_data()
    clean = clean_and_merge_data(raw)
    print("\nCleaned data sample:")
    print(clean[["WTI", "Brent", "CPI", "FedFundsRate",
                 "GeopoliticalRiskIndex"]].tail(6))
