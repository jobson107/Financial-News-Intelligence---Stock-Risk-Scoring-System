"""
Financial News Intelligence System
Phase 7: Streamlit Dashboard

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Financial News Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────
API_BASE = "http://localhost:8000"
RISK_COLORS = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
SENTIMENT_COLORS = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#F44336"}
PRICE_COLORS = {"bullish": "#4CAF50", "neutral": "#9E9E9E", "bearish": "#F44336"}


def load_latest_csv():
    """Load the largest NLP results CSV."""
    csvs = glob.glob("outputs/nlp_results_*.csv")
    if not csvs:
        return None
    return pd.read_csv(max(csvs, key=os.path.getsize))


def load_enriched_csv():
    """Load the latest enriched CSV with stock data."""
    csvs = glob.glob("outputs/enriched_*.csv")
    if not csvs:
        return None
    return pd.read_csv(max(csvs, key=os.path.getsize))


def load_forecast_json():
    """Load latest forecast JSON."""
    jsons = glob.glob("outputs/forecast_*.json")
    if not jsons:
        return None
    with open(max(jsons, key=os.path.getctime), encoding="utf-8") as f:
        return json.load(f)


def api_predict(title: str, summary: str):
    """Call FastAPI /predict endpoint."""
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"title": title, "summary": summary},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def api_stats():
    """Call FastAPI /stats endpoint."""
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
    st.title("Financial News\nIntelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Article Analyzer", "📈 Forecasting",
         "🏦 Stock Analysis", "📰 Article Explorer"]
    )

    st.markdown("---")
    st.caption("Built with Python, MongoDB, VADER, XGBoost, LSTM, FastAPI")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── Load Data ─────────────────────────────────────────────
df = load_latest_csv()
df_enriched = load_enriched_csv()


# ════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Financial News Intelligence Dashboard")
    st.markdown("Real-time NLP analysis of financial news articles")

    if df is None:
        st.warning("No data found. Run Phase 1 and Phase 2 first.")
        st.stop()

    # ── KPI Row ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Articles", f"{len(df):,}")

    with col2:
        avg_sentiment = df["sentiment_compound"].mean()
        delta_color = "normal" if avg_sentiment >= 0 else "inverse"
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}",
                  delta="Positive" if avg_sentiment > 0 else "Negative")

    with col3:
        avg_risk = df["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")

    with col4:
        high_risk = len(df[df["risk_label"] == "high"]) if "risk_label" in df.columns else 0
        st.metric("High Risk Articles", high_risk)

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        if "risk_label" in df.columns:
            risk_counts = df["risk_label"].value_counts().reset_index()
            risk_counts.columns = ["label", "count"]
            fig = px.pie(
                risk_counts, values="count", names="label",
                color="label",
                color_discrete_map=RISK_COLORS,
                hole=0.4
            )
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Phase 3 to see risk distribution")

    with col2:
        st.subheader("Sentiment Distribution")
        if "sentiment_label" in df.columns:
            sent_counts = df["sentiment_label"].value_counts().reset_index()
            sent_counts.columns = ["label", "count"]
            fig = px.bar(
                sent_counts, x="label", y="count",
                color="label",
                color_discrete_map=SENTIMENT_COLORS
            )
            fig.update_layout(height=300, margin=dict(t=0, b=0),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Phase 2 to see sentiment distribution")

    # ── Charts Row 2 ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Articles by Sector")
        if "sector" in df.columns:
            sector_counts = df["sector"].value_counts().reset_index()
            sector_counts.columns = ["sector", "count"]
            fig = px.bar(
                sector_counts, x="count", y="sector",
                orientation="h",
                color="count",
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment Trend Over Time")
        if "published" in df.columns:
            df_time = df.copy()
            df_time["published"] = pd.to_datetime(df_time["published"], errors="coerce")
            df_time = df_time.dropna(subset=["published"])
            df_time["date"] = df_time["published"].dt.date
            daily = df_time.groupby("date")["sentiment_compound"].mean().reset_index()
            fig = px.line(
                daily, x="date", y="sentiment_compound",
                color_discrete_sequence=["#2196F3"]
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # ── Source Breakdown ──────────────────────────────────
    st.subheader("Articles by Source")
    if "source" in df.columns:
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig = px.bar(
            source_counts, x="source", y="count",
            color="count", color_continuous_scale="Viridis"
        )
        fig.update_layout(height=250, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 2: ARTICLE ANALYZER
# ════════════════════════════════════════════════════════
elif page == "🔍 Article Analyzer":
    st.title("🔍 Article Risk Analyzer")
    st.markdown("Paste any financial news article to get instant risk and sentiment analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        title = st.text_input(
            "Article Headline",
            placeholder="e.g. Tesla faces supply chain disruption amid chip shortage"
        )
        summary = st.text_area(
            "Article Summary (optional)",
            placeholder="Paste the article body here...",
            height=150
        )
        analyze_btn = st.button("Analyze Article", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### How it works")
        st.markdown("""
        1. **VADER** scores sentiment (-1 to +1)
        2. **TF-IDF** extracts keywords
        3. **Risk model** combines 3 signals
        4. **XGBoost** predicts price movement
        5. **LSTM** gives sequence-based prediction
        """)

    if analyze_btn and title:
        with st.spinner("Analyzing..."):
            result = api_predict(title, summary)

        if result:
            st.markdown("---")
            st.subheader("Analysis Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                sentiment = result["sentiment"]
                color = SENTIMENT_COLORS.get(sentiment["label"], "#9E9E9E")
                st.markdown(f"""
                <div style='background:{color}20; padding:15px; border-radius:8px; border-left:4px solid {color}'>
                <h4>Sentiment</h4>
                <h2>{sentiment['compound']:.3f}</h2>
                <p>{sentiment['label'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                risk = result["risk"]
                color = RISK_COLORS.get(risk["label"], "#9E9E9E")
                st.markdown(f"""
                <div style='background:{color}20; padding:15px; border-radius:8px; border-left:4px solid {color}'>
                <h4>Risk Score</h4>
                <h2>{risk['score']:.3f}</h2>
                <p>{risk['label'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                pm = result.get("price_movement_xgboost", "N/A")
                conf = result.get("price_movement_confidence", 0)
                color = PRICE_COLORS.get(pm, "#9E9E9E")
                st.markdown(f"""
                <div style='background:{color}20; padding:15px; border-radius:8px; border-left:4px solid {color}'>
                <h4>XGBoost Signal</h4>
                <h2>{pm}</h2>
                <p>Confidence: {conf:.0%}</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                lstm = result.get("price_movement_lstm")
                if lstm:
                    color = PRICE_COLORS.get(lstm["label"], "#9E9E9E")
                    st.markdown(f"""
                    <div style='background:{color}20; padding:15px; border-radius:8px; border-left:4px solid {color}'>
                    <h4>LSTM Signal</h4>
                    <h2>{lstm['label']}</h2>
                    <p>Confidence: {lstm['confidence']:.0%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background:#f5f5f5; padding:15px; border-radius:8px'>
                    <h4>LSTM Signal</h4>
                    <p>Not available<br>(Run Phase 3B)</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Detected Sector:**", )
                st.info(result.get("sector", "N/A").upper())
                if result.get("ticker_detected"):
                    st.markdown("**Ticker Detected:**")
                    st.success(result["ticker_detected"])

            with col2:
                if risk.get("neg_keywords"):
                    st.markdown("**Risk Keywords Found:**")
                    for kw in risk["neg_keywords"]:
                        st.markdown(f"- `{kw}`")

            # Sentiment breakdown
            st.markdown("---")
            st.subheader("Sentiment Breakdown")
            sent = result["sentiment"]
            fig = go.Figure(go.Bar(
                x=["Positive", "Negative", "Neutral"],
                y=[sent["positive"], sent["negative"], sent["neutral"]],
                marker_color=["#4CAF50", "#F44336", "#9E9E9E"]
            ))
            fig.update_layout(height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Could not connect to API. Make sure FastAPI is running on port 8000.")
            st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")


# ════════════════════════════════════════════════════════
# PAGE 3: FORECASTING
# ════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.title("📈 Sentiment & Risk Forecasting")
    st.markdown("ARIMA vs Prophet — 7-day forward predictions")

    forecast_data = load_forecast_json()

    if forecast_data is None:
        st.warning("No forecast data found. Run Phase 4 first.")
        st.stop()

    for target, result in forecast_data.items():
        st.subheader(f"{target} Forecast")

        comp = result.get("comparison", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ARIMA MAE", comp.get("arima_mae", "N/A"))
        with col2:
            st.metric("Prophet MAE", comp.get("prophet_mae", "N/A"))
        with col3:
            st.metric("ARIMA RMSE", comp.get("arima_rmse", "N/A"))
        with col4:
            winner = comp.get("winner_mae", "N/A")
            st.metric("Winner (MAE)", winner)

        # Show chart if saved
        chart_path = result.get("chart", "")
        if chart_path and os.path.exists(chart_path):
            st.image(chart_path, use_column_width=True)

        st.markdown("---")


# ════════════════════════════════════════════════════════
# PAGE 4: STOCK ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "🏦 Stock Analysis":
    st.title("🏦 Stock Price Movement Analysis")
    st.markdown("Articles matched to real stock price movements via yfinance")

    if df_enriched is None:
        st.warning("No enriched data found. Run Phase 4 first.")
        st.stop()

    df_stock = df_enriched.dropna(subset=["price_label"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Articles Matched", len(df_stock))
    with col2:
        bullish = len(df_stock[df_stock["price_label"] == "bullish"])
        st.metric("Bullish", bullish)
    with col3:
        bearish = len(df_stock[df_stock["price_label"] == "bearish"])
        st.metric("Bearish", bearish)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Movement Distribution")
        price_counts = df_stock["price_label"].value_counts().reset_index()
        price_counts.columns = ["label", "count"]
        fig = px.pie(
            price_counts, values="count", names="label",
            color="label",
            color_discrete_map=PRICE_COLORS,
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment vs Price Movement")
        fig = px.box(
            df_stock, x="price_label", y="sentiment_compound",
            color="price_label",
            color_discrete_map=PRICE_COLORS
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Matched Tickers")
    if "ticker" in df_stock.columns:
        ticker_counts = df_stock["ticker"].value_counts().head(10).reset_index()
        ticker_counts.columns = ["ticker", "count"]
        fig = px.bar(ticker_counts, x="ticker", y="count",
                     color="count", color_continuous_scale="Blues")
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Article Details")
    cols_to_show = ["title", "source", "ticker", "price_label",
                    "pct_change", "sentiment_compound", "risk_score"]
    cols_available = [c for c in cols_to_show if c in df_stock.columns]
    st.dataframe(df_stock[cols_available].head(50), use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 5: ARTICLE EXPLORER
# ════════════════════════════════════════════════════════
elif page == "📰 Article Explorer":
    st.title("📰 Article Explorer")
    st.markdown("Browse and filter all ingested articles")

    if df is None:
        st.warning("No data found. Run Phase 1 and Phase 2 first.")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        sources = ["All"] + sorted(df["source"].unique().tolist()) if "source" in df.columns else ["All"]
        source_filter = st.selectbox("Source", sources)

    with col2:
        if "sentiment_label" in df.columns:
            sent_options = ["All"] + df["sentiment_label"].unique().tolist()
            sent_filter = st.selectbox("Sentiment", sent_options)
        else:
            sent_filter = "All"

    with col3:
        if "risk_label" in df.columns:
            risk_options = ["All"] + df["risk_label"].unique().tolist()
            risk_filter = st.selectbox("Risk Level", risk_options)
        else:
            risk_filter = "All"

    # Apply filters
    filtered = df.copy()
    if source_filter != "All" and "source" in df.columns:
        filtered = filtered[filtered["source"] == source_filter]
    if sent_filter != "All" and "sentiment_label" in df.columns:
        filtered = filtered[filtered["sentiment_label"] == sent_filter]
    if risk_filter != "All" and "risk_label" in df.columns:
        filtered = filtered[filtered["risk_label"] == risk_filter]

    st.markdown(f"Showing **{len(filtered)}** articles")

    cols_to_show = ["title", "source", "published", "sector",
                    "sentiment_compound", "sentiment_label",
                    "risk_score", "risk_label"]
    cols_available = [c for c in cols_to_show if c in filtered.columns]
    st.dataframe(filtered[cols_available], use_container_width=True, height=500)
