# ============================================================
# app.py - Main Streamlit Dashboard
# ============================================================
# Run with:  streamlit run app.py
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, KEY_EVENTS, SCENARIOS,
    START_DATE, END_DATE, FORECAST_START, FORECAST_END
)

# ============================================================
# Page Configuration (must be FIRST Streamlit call)
# ============================================================
st.set_page_config(
    page_title = "Oil Price Prediction 2026-2028",
    page_icon  = "🛢️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ============================================================
# Custom CSS for professional dark theme
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* KPI card style */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .kpi-label { color: #8892b0; font-size: 13px; font-weight: 500; }
    .kpi-value { color: #e6f1ff; font-size: 28px; font-weight: 700; }
    .kpi-delta-pos { color: #00e676; font-size: 14px; }
    .kpi-delta-neg { color: #ff5252; font-size: 14px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #8892b0;
    }
    .stTabs [aria-selected="true"] {
        background: #252840;
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff;
    }

    /* Section headers */
    h2 { color: #ccd6f6 !important; }
    h3 { color: #a8b2d8 !important; }

    /* Sidebar */
    .css-1d391kg { background: #0a0d14; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Loading (cached for performance)
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_all_data():
    """Load or generate all data. Cached for 1 hour."""
    from data_collection     import collect_all_data
    from data_cleaning       import clean_and_merge_data
    from feature_engineering import engineer_features
    from models              import train_all_models
    from forecasting         import generate_forecasts, summarize_forecast

    # Collect / load raw data
    raw   = collect_all_data(force_refresh=False)
    clean = clean_and_merge_data(raw)
    feat, feat_cols = engineer_features(clean)

    # Train models
    results = train_all_models(feat, feat_cols)

    # Generate 2026-2028 forecasts
    forecasts = generate_forecasts(clean, feat_cols, results["models"])
    summary   = summarize_forecast(forecasts)

    return {
        "clean":     clean,
        "feat":      feat,
        "feat_cols": feat_cols,
        "results":   results,
        "forecasts": forecasts,
        "summary":   summary,
    }


# ============================================================
# Helper: Plotly Figure Defaults
# ============================================================

PLOT_BG    = "#0e1117"
PAPER_BG   = "#0e1117"
GRID_COLOR = "#1e2130"
TEXT_COLOR = "#ccd6f6"

def apply_dark_theme(fig):
    """Apply consistent dark theme to any plotly figure."""
    fig.update_layout(
        plot_bgcolor  = PLOT_BG,
        paper_bgcolor = PAPER_BG,
        font          = {"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        xaxis = {"gridcolor": GRID_COLOR, "zeroline": False,
                 "showgrid": True, "linecolor": GRID_COLOR},
        yaxis = {"gridcolor": GRID_COLOR, "zeroline": False,
                 "showgrid": True, "linecolor": GRID_COLOR},
        legend = {"bgcolor": "#1e2130", "bordercolor": GRID_COLOR},
        margin = {"t": 40, "b": 40, "l": 60, "r": 20},
    )
    return fig


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB color to Plotly-compatible rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def kpi_card(label: str, value: str, delta: str = "", positive: bool = True):
    """Render a styled KPI metric card."""
    delta_class = "kpi-delta-pos" if positive else "kpi-delta-neg"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.markdown("""
<div style="text-align:center; padding: 10px 0 20px 0;">
    <h1 style="color:#00d4ff; font-size:2.4rem; font-weight:800;">
        🛢️ US Oil Price Prediction System
    </h1>
    <p style="color:#8892b0; font-size:1.1rem;">
        AI-Powered WTI & Brent Forecasting · 2026–2028 · Geopolitical Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar Controls
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")

    st.markdown("---")
    st.markdown("### 📅 Historical Date Range")
    hist_start = st.date_input("From", value=pd.Timestamp("2019-01-01"),
                                min_value=pd.Timestamp(START_DATE),
                                max_value=pd.Timestamp(END_DATE))
    hist_end   = st.date_input("To",   value=pd.Timestamp(END_DATE),
                                min_value=pd.Timestamp(START_DATE),
                                max_value=pd.Timestamp(END_DATE))

    st.markdown("---")
    st.markdown("### 🎭 Forecast Scenario")
    scenario_choice = st.selectbox(
        "Select Scenario",
        options=list(SCENARIOS.keys()),
        index=1,
        help="Choose the geopolitical/economic scenario for 2026-2028 forecast"
    )

    st.markdown("---")
    st.markdown("### 📊 Chart Options")
    show_events = st.checkbox("Show Key Events", value=True)
    show_ci     = st.checkbox("Show Confidence Interval", value=True)
    show_ma     = st.checkbox("Show Moving Averages", value=True)

    st.markdown("---")
    st.markdown("### 🔄 Data Refresh")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Data: Yahoo Finance · FRED · EIA\nModels: LR · RF · XGBoost · Prophet · ARIMA")


# ============================================================
# Load Data (with spinner)
# ============================================================

with st.spinner("Loading data and training models — please wait ~30 seconds on first run …"):
    try:
        data = load_all_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Try refreshing the page or clicking '🔄 Refresh Data' in the sidebar.")
        st.stop()

clean     = data["clean"]
feat      = data["feat"]
feat_cols = data["feat_cols"]
results   = data["results"]
forecasts = data["forecasts"]
summary   = data["summary"]

# Date-filtered historical data
hist_mask = (clean.index >= str(hist_start)) & (clean.index <= str(hist_end))
hist_df   = clean[hist_mask]


# ============================================================
# KPI Row at Top
# ============================================================

st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    last_wti = clean["WTI"].iloc[-1]
    prev_wti = clean["WTI"].iloc[-2]
    delta    = last_wti - prev_wti
    kpi_card("WTI Crude (Latest)",
             f"${last_wti:.2f}",
             f"{'▲' if delta >= 0 else '▼'} ${abs(delta):.2f} MoM",
             delta >= 0)

with col2:
    last_brent = clean["Brent"].iloc[-1] if "Brent" in clean.columns else last_wti + 3
    kpi_card("Brent Crude (Latest)",
             f"${last_brent:.2f}",
             f"Spread: ${last_brent - last_wti:.1f}")

with col3:
    fc_base = forecasts["Base"]["WTI_Ensemble"].mean()
    kpi_card("2026-28 Base Forecast",
             f"${fc_base:.0f}/bbl",
             "Average WTI")

with col4:
    geo_val = clean["GeopoliticalRiskIndex"].iloc[-1] if "GeopoliticalRiskIndex" in clean.columns else 75
    level   = "HIGH" if geo_val > 70 else "MED" if geo_val > 45 else "LOW"
    kpi_card("Geopolitical Risk",
             f"{geo_val:.0f}/100",
             f"Level: {level}",
             geo_val < 60)

with col5:
    cpi_val = clean["CPI"].iloc[-1] if "CPI" in clean.columns else 320
    kpi_card("US CPI (Inflation)",
             f"{cpi_val:.1f}",
             "Monthly Index")

with col6:
    rate_val = clean["FedFundsRate"].iloc[-1] if "FedFundsRate" in clean.columns else 5.0
    kpi_card("Fed Funds Rate",
             f"{rate_val:.2f}%",
             "Interest Rate")

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📈 Live Monitor",
    "📜 Historical Trends",
    "🔮 Prediction Charts",
    "🌍 Geopolitical Risk",
    "💹 Inflation Tracker",
    "🏭 OPEC Analysis",
    "📰 News Sentiment",
    "🎲 Scenario Sim",
    "📊 Forecast Compare",
    "🏆 Model Metrics",
])


# ============================================================
# TAB 1: Live Oil Price Monitor
# ============================================================

with tabs[0]:
    st.subheader("Live Oil Price Monitor")

    # Price chart: last 12 months
    last_12m = clean.last("12ME") if hasattr(clean, "last") else clean.tail(12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_12m.index, y=last_12m["WTI"],
        name="WTI Crude", line={"color": "#00d4ff", "width": 2.5},
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"
    ))
    if "Brent" in last_12m.columns:
        fig.add_trace(go.Scatter(
            x=last_12m.index, y=last_12m["Brent"],
            name="Brent Crude", line={"color": "#ff6b35", "width": 2.5}
        ))
    fig.update_layout(title="Oil Prices – Last 12 Months ($/barrel)",
                      height=380, hovermode="x unified")
    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Price table
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Recent Monthly Prices")
        display_df = clean[["WTI","Brent"]].tail(12).copy()
        display_df.index = display_df.index.strftime("%b %Y")
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True)

    with col_b:
        st.markdown("#### WTI Price Statistics")
        stats = pd.DataFrame({
            "Metric": ["Current", "3-Month Avg", "6-Month Avg",
                       "12-Month Avg", "YTD High", "YTD Low",
                       "52-Week Chg %"],
            "Value": [
                f"${clean['WTI'].iloc[-1]:.2f}",
                f"${clean['WTI'].tail(3).mean():.2f}",
                f"${clean['WTI'].tail(6).mean():.2f}",
                f"${clean['WTI'].tail(12).mean():.2f}",
                f"${clean['WTI'].tail(12).max():.2f}",
                f"${clean['WTI'].tail(12).min():.2f}",
                f"{((clean['WTI'].iloc[-1] / clean['WTI'].iloc[-13] - 1)*100):.1f}%" if len(clean) > 13 else "N/A",
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2: Historical Trends
# ============================================================

with tabs[1]:
    st.subheader("Historical Oil Price Trends")

    # Full history chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df["WTI"],
        name="WTI Crude", line={"color": "#00d4ff", "width": 2}
    ))

    if "Brent" in hist_df.columns:
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["Brent"],
            name="Brent Crude", line={"color": "#ff6b35", "width": 2}
        ))

    if show_ma and "WTI_12M_Avg" in hist_df.columns:
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["WTI_12M_Avg"],
            name="12-Month MA", line={"color": "#ffd700", "width": 1.5, "dash": "dash"}
        ))

    # Add geopolitical events
    if show_events:
        for ev in KEY_EVENTS:
            ev_date = pd.Timestamp(ev["date"])
            if ev_date < pd.Timestamp(str(hist_start)) or ev_date > pd.Timestamp(str(hist_end)):
                continue
            color = "#ff4444" if ev["impact"] == "negative" else "#00cc44"
            fig.add_vline(x=ev_date, line_dash="dot",
                          line_color=color, opacity=0.5)
            fig.add_annotation(
                x=ev_date, y=hist_df["WTI"].max() * 0.9,
                text=ev["event"][:20], showarrow=False,
                font={"color": color, "size": 9},
                textangle=-45, yanchor="top"
            )

    fig.update_layout(
        title="Historical WTI & Brent Crude Oil Prices",
        yaxis_title="Price (USD/barrel)",
        height=450, hovermode="x unified"
    )
    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Matrix: What Drives Oil Prices?")
    corr_cols = ["WTI", "Brent", "Gold", "NaturalGas",
                 "SP500", "USD_Index", "CPI", "FedFundsRate",
                 "GeopoliticalRiskIndex", "OPEC_Production_Mbpd",
                 "US_Crude_Inventory_Mbbl"]
    avail_corr = [c for c in corr_cols if c in hist_df.columns]
    corr_df    = hist_df[avail_corr].corr().round(2)

    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale="RdBu",
        zmid=0,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))
    fig_heat.update_layout(title="Correlation Heatmap",
                           height=450, width=700)
    apply_dark_theme(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Candlestick-style year-over-year box plot
    st.markdown("#### Annual Price Distribution (Box Plot)")
    hist_df2 = hist_df.copy()
    hist_df2["Year"] = hist_df2.index.year
    fig_box = px.box(hist_df2, x="Year", y="WTI",
                     title="WTI Crude Oil Price Distribution by Year",
                     color="Year",
                     color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_box.update_layout(height=380, showlegend=False)
    apply_dark_theme(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)


# ============================================================
# TAB 3: Prediction Charts
# ============================================================

with tabs[2]:
    st.subheader(f"2026–2028 Oil Price Forecast — {scenario_choice} Scenario")

    sc_info = SCENARIOS[scenario_choice]
    st.info(f"**{sc_info['label']}:** {sc_info['description']}")

    fc_df = forecasts[scenario_choice]

    fig = go.Figure()

    # Historical tail (last 2 years)
    hist_tail = clean.last("24ME") if hasattr(clean, "last") else clean.tail(24)
    fig.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail["WTI"],
        name="Historical WTI",
        line={"color": "#8892b0", "width": 2}
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["WTI_Ensemble"],
        name=f"Forecast ({scenario_choice})",
        line={"color": sc_info["color"], "width": 2.5}
    ))

    # Confidence interval
    if show_ci and "WTI_Lower" in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=list(fc_df.index) + list(fc_df.index[::-1]),
            y=list(fc_df["WTI_Upper"]) + list(fc_df["WTI_Lower"][::-1]),
            fill="toself",
            fillcolor=f"rgba(0,212,255,0.10)",
            line={"color": "rgba(0,0,0,0)"},
            name="95% Confidence Interval",
            showlegend=True
        ))

    # Individual model lines
    for model_col, color in [
        ("WTI_LR",  "#ff6b35"),
        ("WTI_RF",  "#7ecfc0"),
        ("WTI_XGB", "#f7c59f"),
    ]:
        if model_col in fc_df.columns:
            fig.add_trace(go.Scatter(
                x=fc_df.index, y=fc_df[model_col],
                name=model_col.replace("WTI_", ""),
                line={"width": 1, "dash": "dash", "color": color},
                opacity=0.6
            ))

    # Add a divider at forecast start
    fig.add_vline(x=pd.Timestamp(FORECAST_START),
                  line_dash="dash", line_color="#ffd700",
                  annotation_text="Forecast Start", annotation_font_color="#ffd700")

    fig.update_layout(
        title="WTI Oil Price Forecast 2026–2028",
        yaxis_title="Price (USD/barrel)",
        height=480, hovermode="x unified"
    )
    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Quarterly bar chart
    st.markdown("#### Quarterly Average Forecast Prices")
    forecast_price_cols = ["WTI_Ensemble", "WTI_Lower", "WTI_Upper"]
    fc_q = fc_df[forecast_price_cols].apply(pd.to_numeric, errors="coerce").resample("QE").mean()
    fc_q.index = [f"Q{idx.quarter} {idx.year}" for idx in fc_q.index]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=fc_q.index, y=fc_q["WTI_Ensemble"],
        name="Ensemble Forecast",
        marker_color=sc_info["color"],
        error_y={"type": "data",
                 "array": (fc_q["WTI_Upper"] - fc_q["WTI_Ensemble"]).values,
                 "arrayminus": (fc_q["WTI_Ensemble"] - fc_q["WTI_Lower"]).values,
                 "color": "#8892b0"},
        text=fc_q["WTI_Ensemble"].round(1),
        textposition="outside",
        textfont={"color": "#ccd6f6"}
    ))
    fig_bar.update_layout(title="Quarterly WTI Forecast",
                           yaxis_title="$/barrel", height=350)
    apply_dark_theme(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Download button
    csv = fc_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=csv,
        file_name=f"wti_forecast_{scenario_choice.lower().replace(' ','_')}.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 4: Geopolitical Risk Meter
# ============================================================

with tabs[3]:
    st.subheader("Geopolitical Risk Analysis")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        # Gauge chart
        current_risk = clean["GeopoliticalRiskIndex"].iloc[-1] if "GeopoliticalRiskIndex" in clean.columns else 75
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_risk,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Current Geopolitical Risk", "font": {"color": TEXT_COLOR}},
            delta={"reference": clean["GeopoliticalRiskIndex"].iloc[-13] if len(clean) > 13 else 60,
                   "increasing": {"color": "#ff5252"}, "decreasing": {"color": "#00e676"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR},
                "bar":  {"color": "#00d4ff"},
                "bgcolor": PLOT_BG,
                "bordercolor": GRID_COLOR,
                "steps": [
                    {"range": [0, 33],   "color": "#1a472a"},  # Low risk – green
                    {"range": [33, 66],  "color": "#7d5a00"},  # Medium – yellow
                    {"range": [66, 100], "color": "#7a0000"},  # High – red
                ],
                "threshold": {
                    "line": {"color": "#ff6b35", "width": 4},
                    "thickness": 0.75, "value": 75
                }
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor=PAPER_BG,
                                 font={"color": TEXT_COLOR})
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk factor breakdown
        st.markdown("#### Risk Factor Breakdown")
        risk_factors = {
            "US-Iran Tensions":      82,
            "Middle East Conflict":  78,
            "Russia Sanctions":      65,
            "OPEC Political Risk":   55,
            "Hormuz Disruption":     48,
            "China Demand Risk":     40,
            "Trade War Risk":        35,
            "Energy Transition":     25,
        }
        rf_df = pd.DataFrame(list(risk_factors.items()),
                             columns=["Factor", "Score"])
        fig_rf = px.bar(rf_df, x="Score", y="Factor", orientation="h",
                        color="Score",
                        color_continuous_scale=["#00cc44", "#ffaa00", "#ff4444"],
                        range_color=[0, 100])
        fig_rf.update_layout(height=280, showlegend=False,
                             yaxis={"categoryorder": "total ascending"})
        apply_dark_theme(fig_rf)
        st.plotly_chart(fig_rf, use_container_width=True)

    with col_r:
        # Geopolitical risk over time
        if "GeopoliticalRiskIndex" in hist_df.columns:
            fig_geo = go.Figure()
            fig_geo.add_trace(go.Scatter(
                x=hist_df.index,
                y=hist_df["GeopoliticalRiskIndex"],
                name="Geo Risk Index",
                line={"color": "#ff6b35", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(255,107,53,0.10)"
            ))
            # Overlay WTI (secondary axis)
            fig_geo.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["WTI"],
                name="WTI Price", line={"color": "#00d4ff", "width": 1.5},
                yaxis="y2"
            ))
            if show_events:
                for ev in KEY_EVENTS:
                    ev_date = pd.Timestamp(ev["date"])
                    if ev_date < pd.Timestamp(str(hist_start)): continue
                    color = "#ff4444" if ev["impact"] == "negative" else "#00cc44"
                    fig_geo.add_vline(x=ev_date, line_dash="dot",
                                      line_color=color, opacity=0.4)
            fig_geo.update_layout(
                title="Geopolitical Risk vs WTI Price",
                yaxis={"title": "Geo Risk (0-100)", "side": "left"},
                yaxis2={"title": "WTI ($/bbl)", "overlaying": "y",
                        "side": "right", "showgrid": False},
                height=380, hovermode="x unified",
                legend={"x": 0.02, "y": 0.95}
            )
            apply_dark_theme(fig_geo)
            st.plotly_chart(fig_geo, use_container_width=True)

        # Geopolitical scenario impact table
        st.markdown("#### Scenario Impact on Oil Price")
        impact_data = {
            "Scenario":          ["US-Iran Peace Deal", "Iran Sanctions Continue",
                                  "Limited US-Iran Conflict", "Strait of Hormuz Closure",
                                  "OPEC+ 2 mbpd Cut", "Russia Ceasefire"],
            "Probability (%)":   [15, 40, 30, 15, 55, 20],
            "Price Impact ($/bbl)": [-15, 0, +25, +55, +15, -10],
            "Duration":          ["Permanent", "Ongoing", "6-18 months",
                                  "1-3 months", "6 months", "Permanent"],
        }
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 5: Inflation Tracker
# ============================================================

with tabs[4]:
    st.subheader("Inflation & Interest Rate Impact on Oil Prices")

    col_a, col_b = st.columns(2)

    with col_a:
        # CPI vs WTI
        if "CPI" in hist_df.columns:
            fig_infl = make_subplots(specs=[[{"secondary_y": True}]])
            fig_infl.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["WTI"],
                name="WTI Price", line={"color": "#00d4ff", "width": 2}
            ), secondary_y=False)
            fig_infl.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["CPI"],
                name="CPI Inflation", line={"color": "#ff6b35", "width": 2}
            ), secondary_y=True)
            fig_infl.update_layout(title="WTI vs CPI Inflation", height=320,
                                    hovermode="x unified")
            fig_infl.update_yaxes(title_text="WTI ($/bbl)", secondary_y=False)
            fig_infl.update_yaxes(title_text="CPI Index", secondary_y=True)
            apply_dark_theme(fig_infl)
            st.plotly_chart(fig_infl, use_container_width=True)

    with col_b:
        # Interest rate vs WTI
        if "FedFundsRate" in hist_df.columns:
            fig_rate = make_subplots(specs=[[{"secondary_y": True}]])
            fig_rate.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["WTI"],
                name="WTI Price", line={"color": "#00d4ff", "width": 2}
            ), secondary_y=False)
            fig_rate.add_trace(go.Bar(
                x=hist_df.index, y=hist_df["FedFundsRate"],
                name="Fed Funds Rate", marker_color="#ffd700", opacity=0.6
            ), secondary_y=True)
            fig_rate.update_layout(title="WTI vs Federal Funds Rate", height=320,
                                    hovermode="x unified")
            apply_dark_theme(fig_rate)
            st.plotly_chart(fig_rate, use_container_width=True)

    # Scatter: Inflation vs Oil
    if "CPI" in hist_df.columns:
        st.markdown("#### Inflation vs Oil Price Relationship")
        fig_sc = px.scatter(hist_df.reset_index(),
                            x="CPI", y="WTI",
                            color=hist_df.index.year,
                            trendline="ols",
                            labels={"CPI": "CPI Index", "WTI": "WTI ($/bbl)",
                                    "color": "Year"},
                            title="Inflation vs WTI — each dot = one month")
        fig_sc.update_layout(height=370)
        apply_dark_theme(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Key insights
    st.markdown("#### Key Insights: How Inflation & Rates Affect Oil")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**High Inflation → Higher Oil**\nOil is priced in USD. When inflation rises, the purchasing power of USD falls, pushing oil prices higher in nominal terms.")
    with col2:
        st.warning("**Rate Hikes → Lower Oil**\nWhen the Fed raises rates, the USD strengthens and economic activity slows — both depress oil demand and prices.")
    with col3:
        st.error("**Real Rate Matters Most**\nNegative real rates (Fed rate < inflation) are the most bullish environment for commodities including oil.")


# ============================================================
# TAB 6: OPEC Analysis
# ============================================================

with tabs[5]:
    st.subheader("OPEC+ Production Analysis")
    st.caption(
        "⚠️ No free, no-key official OPEC+ production feed exists. This series is a "
        "hand-built approximation of publicly reported OPEC+ policy decisions, not a "
        "live production feed — treat trends as directionally informed, not precise."
    )

    if "OPEC_Production_Mbpd" in hist_df.columns:
        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig_opec = make_subplots(specs=[[{"secondary_y": True}]])
            fig_opec.add_trace(go.Bar(
                x=hist_df.index,
                y=hist_df["OPEC_Production_Mbpd"],
                name="OPEC+ Production (mbpd)",
                marker_color="#7ecfc0", opacity=0.8
            ), secondary_y=False)
            fig_opec.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["WTI"],
                name="WTI Price",
                line={"color": "#00d4ff", "width": 2}
            ), secondary_y=True)
            fig_opec.update_yaxes(title_text="Prod (Mbpd)", secondary_y=False)
            fig_opec.update_yaxes(title_text="WTI ($/bbl)", secondary_y=True,
                                   showgrid=False)
            fig_opec.update_layout(
                title="OPEC+ Production vs WTI Price",
                height=380, hovermode="x unified"
            )
            apply_dark_theme(fig_opec)
            st.plotly_chart(fig_opec, use_container_width=True)

        with col_b:
            st.markdown("#### OPEC+ Key Decisions")
            decisions = pd.DataFrame({
                "Date":       ["Nov 2016", "Apr 2020", "Oct 2022",
                               "Jun 2023", "Nov 2023", "Jun 2024"],
                "Decision":   ["1.8 mbpd cut", "9.7 mbpd cut (COVID)",
                               "2 mbpd cut", "Saudi 1 mbpd voluntary cut",
                               "Extended cuts", "Phase-out delayed"],
                "Price Impact": ["+12%", "-40%", "+8%", "+6%", "+4%", "+3%"],
            })
            st.dataframe(decisions, use_container_width=True, hide_index=True)

            st.markdown("#### OPEC+ vs US Shale")
            st.metric("OPEC+ Market Share", "~43%", "-2% vs 2020")
            st.metric("US Production (mbpd)", "~13.2", "+1.1 YoY")
            st.metric("Break-even Price", "~$70/bbl", "Saudi Arabia")

        # OPEC Production forecast
        st.markdown("#### OPEC+ Production Forecast 2026-2028")
        for sc_name, fc_df in forecasts.items():
            pass  # just use Base
        fc_df = forecasts["Base"]
        if "OPEC_Production_Mbpd" in fc_df.columns:
            opec_hist = hist_df["OPEC_Production_Mbpd"].tail(24)
            opec_fc   = fc_df["OPEC_Production_Mbpd"]
            fig_opec_fc = go.Figure()
            fig_opec_fc.add_trace(go.Scatter(
                x=opec_hist.index, y=opec_hist,
                name="Historical", line={"color": "#8892b0"}
            ))
            fig_opec_fc.add_trace(go.Scatter(
                x=opec_fc.index, y=opec_fc,
                name="Forecast", line={"color": "#7ecfc0", "dash": "dash"}
            ))
            fig_opec_fc.add_vline(x=pd.Timestamp(FORECAST_START),
                                   line_dash="dash", line_color="#ffd700")
            fig_opec_fc.update_layout(title="OPEC+ Production Forecast",
                                       yaxis_title="Mbpd", height=320)
            apply_dark_theme(fig_opec_fc)
            st.plotly_chart(fig_opec_fc, use_container_width=True)


# ============================================================
# TAB 7: News Sentiment Analysis
# ============================================================

with tabs[6]:
    st.subheader("Oil Market News Sentiment Analysis")
    st.caption(
        "⚠️ No free, no-key news-sentiment feed exists. This is a proxy derived from "
        "the real Geopolitical Risk Index's momentum and real WTI volatility — it is "
        "NOT computed from actual news articles and should be read as a second view "
        "on the same real risk signal, not independent sentiment data."
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        if "NewsSentiment" in hist_df.columns:
            # Sentiment over time with color
            sentiment = hist_df["NewsSentiment"]
            colors     = ["#ff4444" if v < 0 else "#00cc44" for v in sentiment]
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                x=sentiment.index, y=sentiment,
                name="Sentiment Score",
                marker_color=colors, opacity=0.8
            ))
            fig_sent.add_trace(go.Scatter(
                x=sentiment.index,
                y=sentiment.rolling(3).mean(),
                name="3-Month MA",
                line={"color": "#ffd700", "width": 2}
            ))
            fig_sent.add_hline(y=0, line_color="#8892b0",
                                line_dash="dash", opacity=0.5)
            fig_sent.update_layout(
                title="Oil Market News Sentiment (-1 Bearish → +1 Bullish)",
                yaxis_title="Sentiment Score",
                height=370, hovermode="x unified"
            )
            apply_dark_theme(fig_sent)
            st.plotly_chart(fig_sent, use_container_width=True)

    with col_b:
        # Current sentiment meter
        curr_sent = clean["NewsSentiment"].iloc[-1] if "NewsSentiment" in clean.columns else 0.3
        label = "Bullish 📈" if curr_sent > 0.2 else ("Bearish 📉" if curr_sent < -0.2 else "Neutral ➡️")
        fig_sm = go.Figure(go.Indicator(
            mode="gauge+number",
            value=(curr_sent + 1) * 50,  # scale -1..1 → 0..100
            title={"text": f"Market Sentiment\n{label}", "font": {"size": 14, "color": TEXT_COLOR}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100],
                         "ticktext": ["Very\nBearish", "Bearish", "Neutral",
                                      "Bullish", "Very\nBullish"]},
                "bar": {"color": "#00e676" if curr_sent > 0 else "#ff5252"},
                "bgcolor": PLOT_BG,
                "steps": [
                    {"range": [0, 25],  "color": "#3d0000"},
                    {"range": [25, 50], "color": "#4a3000"},
                    {"range": [50, 75], "color": "#1a3d1a"},
                    {"range": [75, 100],"color": "#0a2e0a"},
                ],
            }
        ))
        fig_sm.update_layout(height=280, paper_bgcolor=PAPER_BG,
                              font={"color": TEXT_COLOR})
        st.plotly_chart(fig_sm, use_container_width=True)

        st.markdown("#### Sentiment Drivers")
        drivers = {
            "🔴 Iran nuclear talks stalled":      -0.3,
            "🔴 OPEC+ extends supply cuts":       +0.4,
            "🟡 US recession concerns":           -0.2,
            "🟢 China demand recovery":           +0.3,
            "🔴 USD strengthening":               -0.2,
            "🟢 Demand beats estimates":          +0.2,
        }
        for d, s in drivers.items():
            color = "#00e676" if s > 0 else "#ff5252"
            st.markdown(f"<span style='color:{color}'>{d}: {'+' if s>0 else ''}{s:.1f}</span>",
                        unsafe_allow_html=True)


# ============================================================
# TAB 8: Scenario Simulation
# ============================================================

with tabs[7]:
    st.subheader("Interactive Scenario Simulation")
    st.markdown("Adjust the sliders below to simulate different geopolitical and economic conditions and see the estimated oil price impact.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Geopolitical Controls")
        iran_level   = st.slider("US-Iran Conflict Level",        0, 10, 4,
                                  help="0=Peace, 5=Sanctions, 10=Military War")
        hormuz_prob  = st.slider("Strait of Hormuz Disruption %", 0, 100, 15,
                                  help="% of shipping disrupted")
        me_conflict  = st.slider("Middle East Conflict Index",    0, 10, 6)
        russia_sanc  = st.slider("Russia Sanctions Intensity",    0, 10, 7)

    with col2:
        st.markdown("### 🏭 Market Controls")
        opec_cut_pct = st.slider("OPEC+ Production Cut %",       0, 30, 5,
                                  help="% cut from current production")
        us_inv_change= st.slider("US Inventory Change (Mbbl)",   -50, 50, -10)
        fed_rate_chg = st.slider("Fed Rate Change (bps)",        -200, 200, -50)
        china_demand = st.slider("China Demand Growth (%)",       -10, 20, 5)

    # Compute estimated price
    base_price   = clean["WTI"].iloc[-1]

    iran_impact    = iran_level   * 3.5          # Each conflict level = +$3.50
    hormuz_impact  = hormuz_prob  * 0.55         # Each 1% disruption = +$0.55
    me_impact      = me_conflict  * 1.5          # Each index point = +$1.50
    russia_impact  = russia_sanc  * 0.8          # Each sanction level = +$0.80
    opec_impact    = opec_cut_pct * 1.5          # Each 1% cut = +$1.50
    inv_impact     = -us_inv_change * 0.3        # Lower inventory → higher price
    rate_impact    = -fed_rate_chg  * 0.01       # Rate cut → slightly bullish
    china_impact   = china_demand   * 0.8        # More China demand → higher price

    total_impact = (iran_impact + hormuz_impact + me_impact + russia_impact
                    + opec_impact + inv_impact + rate_impact + china_impact)
    simulated_price = base_price + total_impact

    st.markdown("---")
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        kpi_card("Current WTI Price", f"${base_price:.2f}")
    with col_res2:
        direction = "▲" if total_impact >= 0 else "▼"
        kpi_card("Total Price Impact",
                 f"{direction} ${abs(total_impact):.2f}",
                 "Scenario vs. current",
                 total_impact >= 0)
    with col_res3:
        kpi_card("Simulated WTI Price",
                 f"${simulated_price:.2f}",
                 f"{'Bullish' if total_impact > 0 else 'Bearish'} scenario",
                 total_impact >= 0)

    # Impact breakdown chart
    st.markdown("#### Price Impact Breakdown ($)")
    impact_dict = {
        "Iran Conflict":      iran_impact,
        "Hormuz Disruption":  hormuz_impact,
        "ME Conflict":        me_impact,
        "Russia Sanctions":   russia_impact,
        "OPEC+ Cuts":         opec_impact,
        "Inventory Change":   inv_impact,
        "Fed Rate Change":    rate_impact,
        "China Demand":       china_impact,
    }
    colors_impact = ["#ff4444" if v < 0 else "#00cc44"
                     for v in impact_dict.values()]
    fig_impact = go.Figure(go.Bar(
        x=list(impact_dict.keys()),
        y=list(impact_dict.values()),
        marker_color=colors_impact,
        text=[f"${v:+.1f}" for v in impact_dict.values()],
        textposition="outside",
        textfont={"color": "#ccd6f6"}
    ))
    fig_impact.add_hline(y=0, line_color="#8892b0", line_dash="dash")
    fig_impact.update_layout(
        title="Price Impact by Factor",
        yaxis_title="Price Impact ($/bbl)",
        height=350
    )
    apply_dark_theme(fig_impact)
    st.plotly_chart(fig_impact, use_container_width=True)


# ============================================================
# TAB 9: Forecast Comparison
# ============================================================

with tabs[8]:
    st.subheader("All Scenarios Forecast Comparison")

    # Three-scenario overlay
    fig_comp = go.Figure()

    # Historical tail
    hist_tail_24 = clean.tail(24)
    fig_comp.add_trace(go.Scatter(
        x=hist_tail_24.index, y=hist_tail_24["WTI"],
        name="Historical", line={"color": "#8892b0", "width": 2}
    ))

    for sc_name, sc_conf in SCENARIOS.items():
        if sc_name not in forecasts:
            continue
        fc = forecasts[sc_name]
        fig_comp.add_trace(go.Scatter(
            x=fc.index, y=fc["WTI_Ensemble"],
            name=f"{sc_name} ({sc_conf['label']})",
            line={"color": sc_conf["color"], "width": 2.5}
        ))
        if show_ci and "WTI_Lower" in fc.columns and "WTI_Upper" in fc.columns:
            fig_comp.add_trace(go.Scatter(
                x=list(fc.index) + list(fc.index[::-1]),
                y=list(fc["WTI_Upper"]) + list(fc["WTI_Lower"][::-1]),
                fill="toself",
                fillcolor=hex_to_rgba(sc_conf["color"], 0.08),
                line={"color": "rgba(0,0,0,0)"},
                name=f"{sc_name} CI",
                showlegend=False
            ))

    fig_comp.add_vline(x=pd.Timestamp(FORECAST_START),
                        line_dash="dash", line_color="#ffd700",
                        annotation_text="Forecast Start →",
                        annotation_font_color="#ffd700")
    fig_comp.update_layout(
        title="WTI Oil Price: All Scenarios 2026–2028",
        yaxis_title="Price (USD/barrel)",
        height=480, hovermode="x unified"
    )
    apply_dark_theme(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Summary table
    st.markdown("#### Forecast Summary by Year & Scenario")
    pivot = summary.pivot(index="Year", columns="Scenario", values="Avg WTI ($)")
    st.dataframe(pivot.style.format("${:.2f}").background_gradient(
        cmap="RdYlGn", axis=None
    ), use_container_width=True)

    # Year by year comparison
    col_a, col_b, col_c = st.columns(3)
    for i, year in enumerate([2026, 2027, 2028]):
        with [col_a, col_b, col_c][i]:
            st.markdown(f"#### {year} Forecast")
            yr_summary = summary[summary["Year"] == year].copy()
            yr_summary = yr_summary.set_index("Scenario")[
                ["Avg WTI ($)", "Min ($)", "Max ($)"]
            ]
            st.dataframe(yr_summary, use_container_width=True)

    # Download combined forecast
    try:
        combined_df = pd.read_csv(f"{DATA_DIR}/forecast_combined.csv",
                                   index_col=0, parse_dates=True)
        csv_combined = combined_df.to_csv().encode("utf-8")
        st.download_button("⬇️ Download All Scenarios CSV",
                            data=csv_combined,
                            file_name="oil_price_forecast_2026_2028.csv",
                            mime="text/csv")
    except Exception:
        pass


# ============================================================
# TAB 10: Model Accuracy Metrics
# ============================================================

with tabs[9]:
    st.subheader("Model Performance & Accuracy")

    metrics_df = results["metrics"]

    # Metrics table with highlighting
    st.markdown("#### Model Comparison Table")
    st.markdown("_Lower RMSE/MAE = better accuracy. Higher R² = better fit._")
    styled = metrics_df.style.format({
        "RMSE": "${:.2f}", "MAE": "${:.2f}",
        "R2": "{:.4f}", "MAPE": "{:.1f}%"
    }).background_gradient(subset=["R2"], cmap="Greens") \
      .background_gradient(subset=["RMSE"], cmap="Reds_r")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Metrics bar charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig_rmse = px.bar(metrics_df, x="Model", y="RMSE",
                          color="RMSE", color_continuous_scale="Reds_r",
                          title="RMSE by Model (lower is better)",
                          text="RMSE")
        fig_rmse.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
        fig_rmse.update_layout(height=320, showlegend=False)
        apply_dark_theme(fig_rmse)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_b:
        fig_r2 = px.bar(metrics_df, x="Model", y="R2",
                         color="R2", color_continuous_scale="Greens",
                         title="R² Score by Model (higher is better)",
                         text="R2")
        fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_r2.update_layout(height=320, showlegend=False)
        apply_dark_theme(fig_r2)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Feature Importance
    st.markdown("#### Top Features Driving WTI Price (XGBoost)")
    try:
        xgb_imp = results["importance"]["xgb"].head(15)
        fig_imp  = px.bar(xgb_imp, x="Importance_Pct", y="Feature",
                          orientation="h",
                          color="Importance_Pct",
                          color_continuous_scale="Blues",
                          title="Feature Importance — Which factors predict oil prices?",
                          text="Importance_Pct")
        fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_imp.update_layout(height=480,
                               yaxis={"categoryorder": "total ascending"})
        apply_dark_theme(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance chart unavailable: {e}")

    # Actual vs Predicted (best model)
    st.markdown("#### Actual vs Predicted Prices (Test Set)")
    best_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    model_key_map = {
        "Random Forest":    "RandomForest",
        "XGBoost":          "XGBoost",
        "Linear Regression":"LinearRegression",
        "Prophet":          "Prophet",
        "ARIMA":            "ARIMA",
    }
    best_key = model_key_map.get(best_model, "XGBoost")

    if best_key in results["forecasts"]:
        dates_te, preds_te, actuals_te = results["forecasts"][best_key]
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=dates_te, y=actuals_te,
            name="Actual WTI", line={"color": "#00d4ff", "width": 2}
        ))
        fig_avp.add_trace(go.Scatter(
            x=dates_te, y=preds_te,
            name=f"Predicted ({best_model})",
            line={"color": "#ff6b35", "width": 2, "dash": "dash"}
        ))
        fig_avp.update_layout(
            title=f"Actual vs. Predicted — {best_model} (Test Set)",
            yaxis_title="WTI ($/barrel)",
            height=350, hovermode="x unified"
        )
        apply_dark_theme(fig_avp)
        st.plotly_chart(fig_avp, use_container_width=True)

    # Metric explanations
    st.markdown("#### Understanding the Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**RMSE** (Root Mean Squared Error)\n\nAverage prediction error in $/barrel. An RMSE of $5 means predictions are off by about $5 on average.")
    with col2:
        st.info("**MAE** (Mean Absolute Error)\n\nSimilar to RMSE but less sensitive to large errors. More intuitive: average absolute dollar error.")
    with col3:
        st.success("**R² Score** (R-squared)\n\n% of price variation explained. R²=0.95 means the model explains 95% of oil price movements.")
    with col4:
        st.warning("**MAPE** (Mean Absolute % Error)\n\nError as % of actual price. A MAPE of 5% means predictions are within 5% of actual prices.")


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.85rem; padding:20px 0;">
    🛢️ <b>Oil Price Prediction System</b> · Built with Python, Streamlit, Plotly, XGBoost & Prophet<br>
    Real data: Yahoo Finance (prices) · FRED (CPI, rates) · EIA (inventories) · Caldara-Iacoviello GPR Index (Fed)<br>
    Proxy/estimated data: OPEC+ production (policy history, not a live feed) · Risk sentiment (derived from real GPR + volatility, not scraped news)<br>
    <i>⚠️ This is for educational and research purposes only. Not financial advice.</i>
</div>
""", unsafe_allow_html=True)
