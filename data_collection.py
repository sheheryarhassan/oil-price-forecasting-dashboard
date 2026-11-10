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
# A source outage keeps saved observations and reports their age in the app.
# ============================================================

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from io import BytesIO

warnings.filterwarnings("ignore")

from config import (
    START_DATE, DATA_DIR
)

REQUEST_TIMEOUT = 30  # seconds, for all external HTTP calls below

# Sent on every external request below. Some government/data sites are
# known to reject requests carrying Python's default User-Agent as basic
# anti-bot protection, so a standard browser-like one is used defensively.
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

def fetch_daily_market() -> pd.DataFrame:
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
                symbol, start=START_DATE, end=pd.Timestamp.now(tz="America/New_York").date().isoformat(),
                progress=False, auto_adjust=False, threads=False, timeout=10,
            )
            if raw is None or len(raw) <= 100:
                print(f"    -- {name:15s} ({symbol}): too little data")
                continue

            # Handle both flat and MultiIndex column layouts.
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    close = raw["Close"]
                    series = close.iloc[:, 0] if hasattr(close, "columns") else close
                else:
                    print(f"    -- {name:15s} ({symbol}): unexpected column layout")
                    continue
            else:
                series = raw["Close"]

            frames[name] = series.squeeze()
            print(f"    OK {name:15s} ({symbol}): {len(raw):,} rows")
        except Exception as exc:
            print(f"    -- {name:15s} ({symbol}): {exc}")

    daily = pd.DataFrame(frames).sort_index().replace([np.inf, -np.inf], np.nan)
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    for name in ("WTI", "Brent"):
        if name not in daily or daily[name].count() < 100:
            raise ValueError(f"No usable {name} futures history from Yahoo Finance")
    return daily


def save_frame(frame, path):
    # Replace only after the complete CSV has been written.
    with NamedTemporaryFile(mode="w", dir=Path(path).parent, delete=False) as tmp:
        try:
            frame.to_csv(tmp)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def fetch_market_data():
    return fetch_daily_market().resample("MS").mean().dropna(subset=["WTI"])


def load_market(force_refresh):
    daily_path = Path(DATA_DIR, "market_daily.csv")
    monthly_path = Path(DATA_DIR, "market_data.csv")
    daily = None
    warnings_list = []
    if daily_path.exists():
        daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    today = pd.Timestamp.now(tz="America/New_York").date()
    checked_today = daily_path.exists() and (
        pd.Timestamp(daily_path.stat().st_mtime, unit="s", tz="UTC")
        .tz_convert("America/New_York").date() == today
    )
    cache_recent = daily is not None and all(
        name in daily and daily[name].notna().any()
        and daily[name].last_valid_index() >= pd.Timestamp(today) - pd.offsets.BDay(1)
        for name in ("WTI", "Brent")
    )
    if force_refresh or not (checked_today and cache_recent):
        try:
            fresh = fetch_daily_market()
            for name in ("WTI", "Brent"):
                last = fresh[name].last_valid_index()
                if (pd.Timestamp(today) - last).days > 7:
                    raise ValueError(f"{name} download ends on {last.date()}")
                if daily is not None and last < daily[name].last_valid_index():
                    raise ValueError(f"{name} download is older than the saved close")
            updated = fresh if daily is None else fresh.combine_first(daily).sort_index()
            months = updated["WTI"].resample("MS").count()
            if len(months) < 36 or (months == 0).any():
                raise ValueError("WTI history has missing months or fewer than three years of observations")
            save_frame(updated, daily_path)
            daily = updated
        except Exception as exc:
            warnings_list.append(f"Market refresh failed: {exc}. Showing saved history.")
    if daily is None:
        if not monthly_path.exists():
            raise RuntimeError("No saved oil prices are available. " + " ".join(warnings_list))
        monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
        warnings_list.append("Using the bundled monthly snapshot. Daily closes are unavailable.")
        quotes = {}
    else:
        monthly = daily.resample("MS").mean().dropna(subset=["WTI"])
        quotes = {}
        for name in ("WTI", "Brent"):
            series = daily[name].dropna()
            quotes[name] = (series.index[-1], series.iloc[-1])
            if (pd.Timestamp(today) - series.index[-1]).days > 7:
                warnings_list.append(f"Last saved {name} close is over a week old.")
        save_frame(monthly, monthly_path)
    monthly.attrs.update(quotes=quotes, warnings=warnings_list)
    return monthly


# ------------------------------------------------------------------
# 2. Economic Data via FRED public CSV export (FREE, no API key) — REAL
# ------------------------------------------------------------------

def fetch_economic_data() -> pd.DataFrame:
    """Download both monthly indicators in one FRED request."""
    names = {"CPIAUCSL": "CPI", "FEDFUNDS": "FedFundsRate"}
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": ",".join(names), "cosd": START_DATE,
              "coed": pd.Timestamp.today().date().isoformat()}
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=(5, 10))
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 1:
                raise

    frame = pd.read_csv(BytesIO(response.content), index_col=0, parse_dates=True)
    frame = frame[list(names)].rename(columns=names).apply(pd.to_numeric, errors="coerce")
    frame = frame.resample("MS").mean().dropna(how="all")
    if frame.count().min() == 0:
        raise ValueError("FRED returned an empty economic indicator")
    return frame


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
        df = df.loc[(df.index >= START_DATE) & (df.index <= pd.Timestamp.today())]

        if df["GeopoliticalRiskIndex"].dropna().empty:
            raise ValueError("GPR fetch returned no rows in the configured date range")

        print(f"    OK GeopoliticalRiskIndex (real GPR index): {len(df)} rows")
        return df[["GeopoliticalRiskIndex"]]

    except Exception as exc:
        raise ValueError(str(exc)) from exc


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
    dates = pd.date_range(start=START_DATE, end=pd.Timestamp.today(), freq="MS")
    n = len(dates)
    rng = np.random.default_rng(42)
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

        prod[i] += rng.normal(0, 0.2)
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
        df = df.loc[(df.index >= START_DATE) & (df.index <= pd.Timestamp.today())]

        if df["US_Crude_Inventory_Mbbl"].dropna().empty:
            raise ValueError("EIA inventory fetch returned no rows in the configured date range")

        print(f"    OK US_Crude_Inventory_Mbbl (real EIA data): {len(df)} rows")
        return df

    except Exception as exc:
        raise ValueError(str(exc)) from exc


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

    market = load_market(force_refresh)
    results = {"market": market}
    sources = [
        ("economic", "economic_data", fetch_economic_data),
        ("geo", "geopolitical_data", fetch_geopolitical_risk_index),
        ("opec", "opec_data", generate_opec_production),
        ("inventory", "inventory_data", fetch_crude_inventories),
    ]
    for key, filename, fetch in sources:
        path = Path(DATA_DIR, filename + ".csv")
        cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
        if key == "economic":
            # Monthly indicators have different publication lags. Check once per day.
            due = cached is None or pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").date() < pd.Timestamp.now(tz="UTC").date()
        else:
            due = cached is None or cached.index.max() < market.index.max()
        if force_refresh or due:
            try:
                frame = fetch()
                if frame.empty:
                    raise ValueError("source returned no data")
                if cached is not None:
                    frame = frame.combine_first(cached)
                save_frame(frame, path)
            except Exception as exc:
                if cached is None:
                    raise
                print(f"{key} refresh failed: {exc}")
                if key == "economic":
                    market.attrs["warnings"].append("FRED is unavailable. Showing saved CPI and Fed funds observations.")
                else:
                    market.attrs["warnings"].append(f"{key} refresh failed ({exc}). Using saved inputs.")
                frame = cached
        else:
            frame = cached
        results[key] = frame
    results["sentiment"] = generate_risk_sentiment_proxy(results["geo"], market["WTI"])
    save_frame(results["sentiment"], Path(DATA_DIR, "sentiment_data.csv"))
    return results


# Run standalone for testing
if __name__ == "__main__":
    data = collect_all_data(force_refresh=True)
    for key, df in data.items():
        print(f"  {key:12s}: {df.shape}")
