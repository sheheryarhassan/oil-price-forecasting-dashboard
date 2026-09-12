# ============================================================
# config.py - Configuration Settings
# ============================================================
# --- API Keys (OPTIONAL) ---
# As of this version, CPI, Fed Funds Rate, US crude inventories, and the
# Geopolitical Risk Index are all pulled from real public sources that
# need NO API key (see data_collection.py for the exact endpoints).
#
# These two keys are only needed if you want to upgrade the two remaining
# proxy/synthetic series (OPEC+ production, news sentiment) to real data:
#   FRED key (optional, not required for current features):
#     https://fred.stlouisfed.org/docs/api/api_key.html
#   EIA key (required only to pull real OPEC/world-supply series via the
#   EIA API v2 international dataset, which is NOT the same as the
#   no-key inventory download already used below):
#     https://www.eia.gov/opendata/
# ============================================================

FRED_API_KEY = None   # Optional — not required for CPI/FedFundsRate anymore
EIA_API_KEY  = None   # Optional — only used for a future OPEC-data upgrade

# --- Date Range for Historical Data ---
# END_DATE is computed as "today" every run so the historical dataset never
# goes stale. The previous hardcoded "2025-12-31" meant the model was
# always missing the most recent months of real price action — a real
# source of forecast inaccuracy, since the forecast's own starting point
# was out of date.
import datetime as _dt

START_DATE = "2015-01-01"
END_DATE   = _dt.date.today().strftime("%Y-%m-%d")

# --- Forecast Horizon ---
# Starts the month AFTER the last available historical month, and always
# covers 3 full years forward from today.
_today = _dt.date.today()
_forecast_start_year = _today.year if _today.month < 12 else _today.year + 1
_forecast_start_month = _today.month + 1 if _today.month < 12 else 1
FORECAST_START = f"{_forecast_start_year}-{_forecast_start_month:02d}-01"
FORECAST_END   = f"{_today.year + 3}-12-31"

# --- File Paths ---
DATA_DIR = "data"

# --- Model Settings ---
RANDOM_STATE = 42       # Reproducible results
TEST_SIZE    = 0.2      # 20% of data for testing

# --- Scenario Parameters ---
# These multipliers are applied to the base geopolitical risk
# to create different forecast scenarios
SCENARIOS = {
    "Optimistic": {
        "label": "Peace / Stability",
        "geo_risk_multiplier": 0.5,      # Risk drops 50%
        "opec_cut_factor": -0.05,        # OPEC increases supply 5%
        "iran_conflict_level": 0,        # No conflict
        "hormuz_disruption": 0.0,        # No disruption
        "color": "#00cc44",
        "description": "Peace deal in Middle East, OPEC increases supply, strong global economy"
    },
    "Base": {
        "label": "Moderate Tensions",
        "geo_risk_multiplier": 1.0,      # Current risk level
        "opec_cut_factor": 0.0,          # No change
        "iran_conflict_level": 1,        # Sanctions only
        "hormuz_disruption": 0.1,        # 10% disruption risk
        "color": "#ffaa00",
        "description": "Current tensions continue, gradual normalization, moderate growth"
    },
    "Worst Case": {
        "label": "Regional Conflict",
        "geo_risk_multiplier": 1.8,      # Risk increases 80%
        "opec_cut_factor": 0.15,         # OPEC cuts 15%
        "iran_conflict_level": 3,        # Full military conflict
        "hormuz_disruption": 0.6,        # 60% disruption
        "color": "#ff4444",
        "description": "US-Iran war escalation, Strait of Hormuz partially closed, supply crisis"
    }
}

# --- Geopolitical Events Timeline ---
# Used for marking key events on charts
KEY_EVENTS = [
    {"date": "2016-11-30", "event": "OPEC+ Deal",            "impact": "positive"},
    {"date": "2018-05-08", "event": "US exits Iran Deal",    "impact": "negative"},
    {"date": "2019-09-14", "event": "Aramco Attack",         "impact": "negative"},
    {"date": "2020-01-03", "event": "Soleimani Killed",      "impact": "negative"},
    {"date": "2020-04-20", "event": "COVID: Prices Go Neg.", "impact": "negative"},
    {"date": "2021-11-26", "event": "Omicron Variant",       "impact": "positive"},
    {"date": "2022-02-24", "event": "Russia Invades Ukraine","impact": "negative"},
    {"date": "2022-10-05", "event": "OPEC+ Cuts 2M bpd",    "impact": "negative"},
    {"date": "2023-10-07", "event": "Hamas Attacks Israel",  "impact": "negative"},
    {"date": "2024-04-13", "event": "Iran Attacks Israel",   "impact": "negative"},
    {"date": "2024-12-01", "event": "OPEC+ Extends Cuts",   "impact": "negative"},
]
