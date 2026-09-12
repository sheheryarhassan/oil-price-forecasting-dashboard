# ============================================================
# data_collection.py - Fetch & Generate All Data
# ============================================================
# Data sources, and what's REAL vs. PROXY/SYNTHETIC in this version:
#
#   REAL, no API key required:
#     1. Yahoo Finance          -> WTI, Brent, Gold, NatGas, USD, S&P500
#     2. FRED public CSV export -> CPI, Federal Funds Rate
#     3. EIA public XLS export  -> US commercial crude oil inventories
#     4. Caldara & Iacoviello   -> Geopolitical Risk (GPR) Index
#        (Federal Reserve Board research index, CC-BY licensed,
#         https://www.matteoiacoviello.com/gpr.htm)
#
#   PROXY / SYNTHETIC (clearly labeled, not scraped from a real feed):
#     5. OPEC+ production       -> no free no-key official source exists;
#        this remains a hand-built approximation of known OPEC+ decisions.
#        To get REAL OPEC/world-supply data, get a free EIA API key
#        (https://www.eia.gov/opendata/) and use the EIA API v2
#        international dataset — this is a documented upgrade path, not
#        implemented here to keep the project key-free by default.
#     6. News sentiment         -> no free no-key news-sentiment API exists.
#        Rebuilt here as an explicit, transparent proxy derived from the
#        REAL GPR index plus REAL WTI realized volatility (see
#        generate_risk_sentiment_proxy). It is NOT independent news data
#        and should not be read as one - it's a second view on the same
#        real geopolitical signal, not a new one.
#
# Every "real" fetch function below falls back to a clearly-labeled
# synthetic series (same as the previous version of this module) if the
# live source is unreachable or its format changes, so the pipeline never
# hard-crashes on a network hiccup. Every fallback path prints a warning
# so it's never silently mistaken for real data.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from io import BytesIO
from datetime import datetime

warnings.filterwarnings("ignore")

from config import (
    FRED_API_KEY, EIA_API_KEY, START_DATE, END_DATE, DATA_DIR, KEY_EVENTS
)

REQUEST_TIMEOUT = 30  # seconds, for all external HTTP calls below

# Sent on every external request below. Some government/data sites are
# known to reject requests carrying Python's default User-Agent as basic
# anti-bot protection, so a standard browser-like one is used defensively.
# (Note: this sandbox's own network egress is restricted to an allowlist
# that doesn't include fred.stlouisfed.org, eia.gov, or matteoiacoviello.com,
# so these three fetch functions could not be tested against the live
# internet from here — only against realistic mocked responses, see the
# project delivery notes. Verify these three real fetches actually
# succeed the first time you run this locally, where you'll have normal
# internet access.)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. Market Data via Yahoo Finance (FREE, no API key) — REAL
# ------------------------------------------------------------------

def fetch_market_data() -> pd.DataFrame:
    """
    Download daily price data for oil-related financial instruments.

    Tickers explained:
      CL=F       -> WTI Crude Oil Futures  (US benchmark price)
      BZ=F       -> Brent Crude Futures    (international benchmark)
      GC=F       -> Gold Futures           (safe-haven / USD hedge)
      NG=F       -> Natural Gas Futures    (energy sector)
      DX-Y.NYB   -> US Dollar Index        (stronger USD = cheaper oil)
      ^GSPC      -> S&P 500 Index          (general economic health)
      UGA        -> US Gasoline ETF        (retail fuel prices)

    Note: newer yfinance versions can return either a flat "Close" column
    or a MultiIndex ("Close", ticker) depending on version/settings. This
    function handles both shapes explicitly instead of assuming one.
    """
    print("  Fetching market data from Yahoo Finance …")

    tickers = {
        "WTI":          "CL=F",
        "Brent":        "BZ=F",
        "Gold":         "GC=F",
        "NaturalGas":   "NG=F",
        "USD_Index":    "DX-Y.NYB",
        "SP500":        "^GSPC",
        "Gasoline_ETF": "UGA",
    }

    frames = {}
    for name, symbol in tickers.items():
        try:
            raw = yf.download(
                symbol, start=START_DATE, end=END_DATE,
                progress=False, auto_adjust=False, threads=False,
            )
            if raw is None or len(raw) <= 100:
                print(f"    -- {name:15s} ({symbol}): too little data - will synthesize")
                continue

            # Handle both flat and MultiIndex column layouts.
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    close = raw["Close"]
                    series = close.iloc[:, 0] if hasattr(close, "columns") else close
                else:
                    print(f"    -- {name:15s} ({symbol}): unexpected column layout - will synthesize")
                    continue
            else:
                series = raw["Close"]

            frames[name] = series.squeeze()
            print(f"    OK {name:15s} ({symbol}): {len(raw):,} rows")
        except Exception as exc:
            print(f"    -- {name:15s} ({symbol}): {exc} - will synthesize")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)

    df_monthly = df.resample("MS").mean()
    df_monthly = _fill_synthetic_prices(df_monthly)

    return df_monthly


def _fill_synthetic_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Yahoo Finance returns incomplete columns, fill them with
    realistic SYNTHETIC time series so the rest of the pipeline works.
    This path should rarely trigger since CL=F/BZ=F are liquid, actively
    quoted futures with a long Yahoo Finance history.
    """
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    n = len(dates)
    ref = pd.DataFrame(index=dates)

    if "WTI" not in df.columns or df["WTI"].isna().mean() > 0.4:
        print("    [FALLBACK] synthesizing WTI prices - real Yahoo Finance data unavailable")
        wti = _synthetic_wti(dates)
        ref["WTI"] = wti

    if "Brent" not in df.columns or df["Brent"].isna().mean() > 0.4:
        print("    [FALLBACK] synthesizing Brent prices - real Yahoo Finance data unavailable")
        base = ref.get("WTI", df.get("WTI", pd.Series(65, index=dates)))
        ref["Brent"] = base + np.random.normal(3, 0.5, n)

    for col in ref.columns:
        if col not in df.columns:
            df[col] = ref[col]

    df = df.reindex(dates)
    df = df.ffill().bfill()
    return df


def _synthetic_wti(dates) -> np.ndarray:
    """FALLBACK ONLY. Approximate historical WTI shape, used only if
    Yahoo Finance is completely unreachable for CL=F."""
    n = len(dates)
    prices = np.zeros(n)
    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y <= 2015:
            base = 55 - (m - 1) * 0.5
        elif y == 2016:
            base = 40 + (m - 1) * 1.5
        elif y == 2017:
            base = 50 + (m - 1) * 0.5
        elif y == 2018:
            base = 55 + (m - 1) * 1.5
        elif y == 2019:
            base = 65 - (m - 1) * 0.2
        elif y == 2020:
            if m <= 3:
                base = 55 - (m - 1) * 10
            elif m == 4:
                base = 15
            elif m <= 8:
                base = 15 + (m - 4) * 8
            else:
                base = 40 + (m - 8) * 1.5
        elif y == 2021:
            base = 50 + (m - 1) * 3
        elif y == 2022:
            if m <= 3:
                base = 80 + (m - 1) * 10
            elif m <= 6:
                base = 100 + (m - 3) * 5
            else:
                base = 110 - (m - 6) * 5
        elif y == 2023:
            base = 80 - (m - 1) * 0.5
        else:
            base = 72 + np.sin(m * 0.3) * 4
        prices[i] = base + np.random.normal(0, 2)
    return np.clip(prices, 10, 150)


# ------------------------------------------------------------------
# 2. Economic Data via FRED public CSV export (FREE, no API key) — REAL
# ------------------------------------------------------------------

def _fetch_fred_series_csv(series_id: str) -> pd.Series:
    """
    Fetch a single FRED series using FRED's public CSV export endpoint.
    This endpoint requires NO API key and is the same underlying data
    you'd get from an authenticated api.stlouisfed.org call.

    Endpoint pattern:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd=1913-01-01&coed=9999-12-31
    """
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd=1913-01-01&coed=9999-12-31"
    )
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
    resp.raise_for_status()

    df = pd.read_csv(BytesIO(resp.content))
    # FRED's CSV header naming has changed over time ("DATE" vs
    # "observation_date"); take columns positionally to be robust to that.
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    series = pd.to_numeric(df[value_col], errors="coerce")
    series.name = series_id
    return series.dropna()


def fetch_economic_data() -> pd.DataFrame:
    """
    Fetch macroeconomic indicators from FRED's public CSV export:
      - CPI (Consumer Price Index, CPIAUCSL)  -> inflation proxy
      - Federal Funds Rate (FEDFUNDS)         -> interest rate proxy

    No API key is needed for this endpoint, so this now runs by default
    instead of only when FRED_API_KEY is set. Falls back to synthetic
    data only if the request itself fails.
    """
    print("  Fetching economic data from FRED (public CSV export, no key needed) …")
    try:
        cpi = _fetch_fred_series_csv("CPIAUCSL")
        fed = _fetch_fred_series_csv("FEDFUNDS")

        df = pd.DataFrame({"CPI": cpi, "FedFundsRate": fed})
        df = df.resample("MS").mean()
        df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]

        if df["CPI"].dropna().empty or df["FedFundsRate"].dropna().empty:
            raise ValueError("FRED CSV export returned no usable rows")

        print(f"    OK CPI: {df['CPI'].notna().sum()} rows, "
              f"FedFundsRate: {df['FedFundsRate'].notna().sum()} rows")
        return df

    except Exception as exc:
        print(f"    [FALLBACK] FRED fetch failed ({exc}) - synthesizing economic data")
        return _synthetic_economic_data()


def _synthetic_economic_data() -> pd.DataFrame:
    """FALLBACK ONLY. Used only if the FRED CSV export is unreachable."""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    n = len(dates)

    cpi = np.linspace(237, 315, n)
    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y == 2021:
            cpi[i] += m * 1.5
        elif y == 2022:
            cpi[i] += max(0, 25 - m * 1.8)
        elif y == 2023:
            cpi[i] += max(0, 8 - m * 0.5)
    cpi += np.random.normal(0, 0.5, n)

    fed = np.zeros(n)
    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y <= 2015:
            fed[i] = 0.1 + (m - 1) * 0.02
        elif y == 2016:
            fed[i] = 0.4 + (m - 1) * 0.02
        elif y == 2017:
            fed[i] = 0.9 + (m - 1) * 0.1
        elif y == 2018:
            fed[i] = 1.5 + (m - 1) * 0.15
        elif y == 2019:
            fed[i] = max(1.5, 2.4 - (m - 1) * 0.08)
        elif y == 2020:
            fed[i] = max(0.1, 1.6 - m * 0.15)
        elif y == 2021:
            fed[i] = 0.1
        elif y == 2022:
            fed[i] = min(4.5, 0.1 + (m - 1) * 0.38)
        elif y == 2023:
            fed[i] = min(5.5, 4.5 + (m - 1) * 0.1)
        elif y >= 2024:
            fed[i] = max(4.25, 5.5 - (d.year - 2024) * 0.5 - (m - 1) * 0.03)

    return pd.DataFrame({"CPI": cpi, "FedFundsRate": fed}, index=dates)


# ------------------------------------------------------------------
# 3. Geopolitical Risk Index — REAL (Caldara & Iacoviello, Federal Reserve)
# ------------------------------------------------------------------

GPR_XLS_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"


def fetch_geopolitical_risk_index() -> pd.DataFrame:
    """
    Fetch the real, published Geopolitical Risk (GPR) Index by Dario
    Caldara and Matteo Iacoviello (Federal Reserve Board). This is a
    news-based index counting the share of newspaper articles referencing
    geopolitical tensions each month, published under a CC-BY license and
    updated monthly with no API key required.

    Source: https://www.matteoiacoviello.com/gpr.htm
    Cite as: Caldara, Dario and Matteo Iacoviello (2022), "Measuring
    Geopolitical Risk," American Economic Review, 112(4), pp.1194-1225.

    The raw GPR index is not naturally bounded at 0-100 the way the old
    synthetic version was (it's normalized to average 100 over 2000-2009).
    We rescale it with a fixed, documented transform to keep it on the
    same 0-100 scale the rest of this app expects, WITHOUT changing its
    relative shape.
    """
    print("  Fetching real Geopolitical Risk Index (Caldara & Iacoviello, Fed) …")
    try:
        resp = requests.get(GPR_XLS_URL, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        resp.raise_for_status()

        raw = pd.read_excel(BytesIO(resp.content))
        raw.columns = [str(c).strip() for c in raw.columns]

        date_col = next((c for c in raw.columns if c.lower() in ("month", "date")), None)
        gpr_col = next((c for c in raw.columns if c.lower() == "gpr"), None)

        if date_col is None or gpr_col is None:
            raise ValueError(f"Expected 'month'/'GPR' columns, got: {list(raw.columns)}")

        df = raw[[date_col, gpr_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df = df.rename(columns={gpr_col: "GPR_Raw"})
        df = df.dropna()

        # Rescale to a 0-100 index using a fixed reference range so the
        # rest of the app's "higher = more risk" 0-100 scale still holds.
        # Reference range chosen from the index's documented historical
        # behavior: ~30 in the calmest post-2000 months, ~250 at its most
        # extreme post-2000 spikes. Values outside this range are clipped,
        # not stretched, so the scale doesn't silently shift if a future
        # event sets a new record.
        GPR_FLOOR, GPR_CEIL = 30.0, 250.0
        df["GeopoliticalRiskIndex"] = (
            (df["GPR_Raw"] - GPR_FLOOR) / (GPR_CEIL - GPR_FLOOR) * 100
        ).clip(0, 100)

        df = df.resample("MS").mean()
        df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]

        if df["GeopoliticalRiskIndex"].dropna().empty:
            raise ValueError("GPR fetch returned no rows in the configured date range")

        print(f"    OK GeopoliticalRiskIndex (real GPR index): {len(df)} rows")
        return df[["GeopoliticalRiskIndex"]]

    except Exception as exc:
        print(f"    [FALLBACK] Real GPR index fetch failed ({exc}) - synthesizing risk index")
        return _synthetic_geopolitical_risk_index()


def _synthetic_geopolitical_risk_index() -> pd.DataFrame:
    """FALLBACK ONLY. Hand-built approximation, used only if the real GPR
    index at matteoiacoviello.com is unreachable."""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    n = len(dates)
    risk = np.full(n, 30.0)

    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y == 2015:
            risk[i] = 38 + 5 * np.sin(m)
        elif y == 2016:
            risk[i] = 36 + 4 * np.sin(m)
        elif y == 2017:
            risk[i] = 40 + 4 * np.sin(m * 2)
        elif y == 2018:
            risk[i] = 42 if m < 5 else 58 + 4 * np.sin(m)
        elif y == 2019:
            risk[i] = 78 if m in (9, 10) else 58
        elif y == 2020:
            risk[i] = 88 if m == 1 else (55 if m <= 3 else 32)
        elif y == 2021:
            risk[i] = 52 if m >= 8 else 44
        elif y == 2022:
            risk[i] = 50 if m == 1 else 82 + 5 * np.sin(m)
        elif y == 2023:
            risk[i] = 82 if m >= 10 else 70
        elif y == 2024:
            risk[i] = (84 + 3 * np.sin(m)) if m >= 4 else 76
        else:
            risk[i] = 78 + 4 * np.sin(m)
        risk[i] += np.random.normal(0, 1.5)
        risk[i] = float(np.clip(risk[i], 5, 100))

    return pd.DataFrame({"GeopoliticalRiskIndex": risk}, index=dates)


# ------------------------------------------------------------------
# 4. OPEC+ Production — PROXY / SYNTHETIC (see module docstring)
# ------------------------------------------------------------------

def generate_opec_production() -> pd.DataFrame:
    """
    PROXY DATA, NOT A LIVE FEED. Monthly OPEC+ crude oil production
    (million barrels/day), hand-built from publicly reported OPEC+ policy
    decisions (production targets, announced cuts). This is NOT pulled
    from a live, machine-readable, no-key data source - no such source
    exists for free. Treat this series as directionally informed by real
    history but not a precise production figure.

    To upgrade this to a real, machine-readable series:
      1. Get a free EIA API key: https://www.eia.gov/opendata/
      2. Pull EIA's international petroleum supply dataset (API v2),
         filtered to OPEC member countries, e.g. series group
         INTL.55-1-OPEC-TBPD.M (total OPEC petroleum supply, monthly).
      3. Set EIA_API_KEY in config.py and wire that series in here.
    This is intentionally left as a documented next step rather than
    implemented with a fabricated "official-looking" API call, since a
    fake integration would be worse than an honestly-labeled proxy.
    """
    print("  Generating OPEC+ production data [PROXY - see docstring for real-data upgrade path] …")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    n = len(dates)
    prod = np.zeros(n)

    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y == 2015:
            prod[i] = 32.0
        elif y == 2016:
            prod[i] = 32.5 if m < 12 else 32.0
        elif y == 2017:
            prod[i] = 31.8
        elif y == 2018:
            prod[i] = 32.5 + m * 0.02
        elif y == 2019:
            prod[i] = 32.2 - m * 0.01
        elif y == 2020:
            if m <= 3:
                prod[i] = 32.0
            elif m == 4:
                prod[i] = 24.0
            elif m <= 12:
                prod[i] = 24.0 + (m - 4) * 0.7
        elif y == 2021:
            prod[i] = 30.0 + m * 0.15
        elif y == 2022:
            prod[i] = 32.5 if m <= 9 else 30.5
        elif y == 2023:
            prod[i] = 30.0 - m * 0.04
        elif y == 2024:
            prod[i] = 30.5 + m * 0.01
        else:
            # Beyond 2024 this is a flat carry-forward assumption, not a
            # reported figure - OPEC+ policy after this point is unknown
            # to this hand-built table.
            prod[i] = 30.8

        prod[i] += np.random.normal(0, 0.2)
        prod[i] = float(np.clip(prod[i], 20, 40))

    return pd.DataFrame({"OPEC_Production_Mbpd": prod}, index=dates)


# ------------------------------------------------------------------
# 5. US Crude Oil Inventories — REAL (EIA public XLS export)
# ------------------------------------------------------------------

EIA_CRUDE_STOCKS_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"


def fetch_crude_inventories() -> pd.DataFrame:
    """
    Fetch REAL weekly US commercial crude oil inventories (excluding SPR)
    directly from the EIA's public "dnav" XLS export, which requires no
    API key. This is the same underlying data the EIA Weekly Petroleum
    Status Report is built from.

    Source: https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTUS1&f=W
    Direct file: WCESTUS1w.xls (thousand barrels, weekly)
    """
    print("  Fetching real US crude oil inventories (EIA public export, no key needed) …")
    try:
        resp = requests.get(EIA_CRUDE_STOCKS_XLS_URL, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        resp.raise_for_status()

        raw = pd.read_excel(BytesIO(resp.content), sheet_name="Data 1", skiprows=2)
        raw.columns = [str(c).strip() for c in raw.columns]

        date_col = raw.columns[0]
        value_col = raw.columns[1]

        df = raw[[date_col, value_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Source is in thousand barrels; app expects million barrels.
        df["US_Crude_Inventory_Mbbl"] = pd.to_numeric(df[value_col], errors="coerce") / 1000.0
        df = df[["US_Crude_Inventory_Mbbl"]].dropna()

        df = df.resample("MS").mean()
        df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]

        if df["US_Crude_Inventory_Mbbl"].dropna().empty:
            raise ValueError("EIA inventory fetch returned no rows in the configured date range")

        print(f"    OK US_Crude_Inventory_Mbbl (real EIA data): {len(df)} rows")
        return df

    except Exception as exc:
        print(f"    [FALLBACK] Real EIA inventory fetch failed ({exc}) - synthesizing inventories")
        return _synthetic_crude_inventories()


def _synthetic_crude_inventories() -> pd.DataFrame:
    """FALLBACK ONLY. Used only if the EIA export is unreachable."""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    n = len(dates)

    seasonal = 25 * np.sin(2 * np.pi * np.arange(n) / 12 + np.pi / 2)
    trend = np.zeros(n)
    for i, d in enumerate(dates):
        y = d.year
        if y <= 2016:
            trend[i] = 40
        elif y <= 2019:
            trend[i] = 10
        elif y == 2020:
            trend[i] = 80 if d.month >= 4 else 15
        elif y <= 2022:
            trend[i] = max(0, 60 - (y - 2020) * 25)
        else:
            trend[i] = -5

    noise = np.random.normal(0, 6, n)
    inventory = 430 + seasonal + trend + noise
    inventory = np.clip(inventory, 300, 620)
    return pd.DataFrame({"US_Crude_Inventory_Mbbl": inventory}, index=dates)


# ------------------------------------------------------------------
# 6. Risk Sentiment Proxy — DERIVED FROM REAL DATA (not raw news feed)
# ------------------------------------------------------------------

def generate_risk_sentiment_proxy(geo_df: pd.DataFrame, wti_series: pd.Series = None) -> pd.DataFrame:
    """
    Builds a "market risk sentiment" proxy from REAL inputs: the real GPR
    index's month-over-month change, plus (if available) real WTI realized
    volatility. This deliberately replaces the old "NewsSentiment" series,
    which was entirely fabricated and mathematically derived from the OLD
    fabricated geopolitical index (fake derived from fake).

    IMPORTANT: this is still a derived PROXY, not independent news data —
    no free, no-key news-sentiment feed exists. It should be read as "is
    real-world risk currently rising or falling, and how volatile is the
    market pricing that in" — not as a sentiment score computed from
    actual news articles. The app UI labels it accordingly.
    """
    print("  Building risk sentiment proxy from real GPR + WTI volatility …")
    geo = geo_df["GeopoliticalRiskIndex"]

    geo_momentum = geo.diff(3).fillna(0) / 100.0  # 3-month change, scaled

    if wti_series is not None and len(wti_series.dropna()) > 6:
        wti_vol = wti_series.pct_change().rolling(3).std().reindex(geo.index).fillna(0)
        sentiment = np.clip(geo_momentum * (1 + wti_vol * 5), -1, 1)
    else:
        sentiment = np.clip(geo_momentum * 2, -1, 1)

    return pd.DataFrame({"NewsSentiment": sentiment}, index=geo.index)


# ------------------------------------------------------------------
# Master Collection Function
# ------------------------------------------------------------------

def collect_all_data(force_refresh: bool = False):
    """
    Orchestrate collection of all data sources and persist to CSV.
    Returns a dict of DataFrames.

    Set force_refresh=True to re-download even if CSVs already exist.
    """
    print("\n" + "=" * 60)
    print("   OIL PRICE PREDICTION — DATA COLLECTION")
    print("=" * 60)

    _ensure_data_dir()

    results = {}

    # --- Market Data (REAL) ---
    path = f"{DATA_DIR}/market_data.csv"
    if force_refresh or not os.path.exists(path):
        df = fetch_market_data()
        df.to_csv(path)
        print(f"  Saved -> {path}")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded (cached) -> {path}")
    results["market"] = df

    # --- Economic Data (REAL, FRED public CSV export) ---
    path = f"{DATA_DIR}/economic_data.csv"
    if force_refresh or not os.path.exists(path):
        df = fetch_economic_data()
        df.to_csv(path)
        print(f"  Saved -> {path}")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded (cached) -> {path}")
    results["economic"] = df

    # --- Geopolitical Risk (REAL, Caldara & Iacoviello GPR index) ---
    path = f"{DATA_DIR}/geopolitical_data.csv"
    if force_refresh or not os.path.exists(path):
        df = fetch_geopolitical_risk_index()
        df.to_csv(path)
        print(f"  Saved -> {path}")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded (cached) -> {path}")
    results["geo"] = df

    # --- OPEC Production (PROXY - see docstring) ---
    path = f"{DATA_DIR}/opec_data.csv"
    if force_refresh or not os.path.exists(path):
        df = generate_opec_production()
        df.to_csv(path)
        print(f"  Saved -> {path}")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded (cached) -> {path}")
    results["opec"] = df

    # --- Inventories (REAL, EIA public export) ---
    path = f"{DATA_DIR}/inventory_data.csv"
    if force_refresh or not os.path.exists(path):
        df = fetch_crude_inventories()
        df.to_csv(path)
        print(f"  Saved -> {path}")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded (cached) -> {path}")
    results["inventory"] = df

    # --- Risk Sentiment Proxy (derived from real GPR + real WTI) ---
    path = f"{DATA_DIR}/sentiment_data.csv"
    if force_refresh or not os.path.exists(path):
        wti_series = results["market"]["WTI"] if "WTI" in results["market"].columns else None
        df = generate_risk_sentiment_proxy(results["geo"], wti_series)
        df.to_csv(path)
        print(f"  Saved -> {path}")
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded (cached) -> {path}")
    results["sentiment"] = df

    print("\n  Data collection complete!\n")
    print("  Data authenticity summary:")
    print("    REAL:   WTI/Brent/Gold/NatGas/USD/S&P500 (Yahoo Finance)")
    print("    REAL:   CPI, Fed Funds Rate (FRED public export)")
    print("    REAL:   US crude inventories (EIA public export)")
    print("    REAL:   Geopolitical Risk Index (Caldara & Iacoviello / Fed)")
    print("    PROXY:  OPEC+ production (hand-built from public policy history)")
    print("    PROXY:  News sentiment (derived from real GPR + WTI volatility)")
    print()

    return results


# Run standalone for testing
if __name__ == "__main__":
    data = collect_all_data(force_refresh=True)
    for key, df in data.items():
        print(f"  {key:12s}: {df.shape}")
