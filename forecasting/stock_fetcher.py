import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Company → Ticker Mapping ─────────────────────────────
COMPANY_TICKER_MAP = {
    # Tech
    "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
    "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "intel": "INTC", "amd": "AMD", "netflix": "NFLX",
    "salesforce": "CRM", "oracle": "ORCL", "ibm": "IBM",
    "qualcomm": "QCOM", "broadcom": "AVGO", "tsmc": "TSM",

    # Finance
    "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS",
    "bank of america": "BAC", "wells fargo": "WFC", "citigroup": "C",
    "blackrock": "BLK", "visa": "V", "mastercard": "MA",
    "american express": "AXP", "paypal": "PYPL",

    # Energy
    "exxon": "XOM", "chevron": "CVX", "shell": "SHEL",
    "bp": "BP", "conocophillips": "COP", "halliburton": "HAL",

    # Healthcare
    "pfizer": "PFE", "johnson": "JNJ", "moderna": "MRNA",
    "abbvie": "ABBV", "merck": "MRK", "unitedhealth": "UNH",

    # Macro proxies
    "s&p": "SPY", "nasdaq": "QQQ", "dow jones": "DIA",
    "federal reserve": "SPY", "fed": "SPY", "treasury": "TLT",
    "bitcoin": "BTC-USD", "crypto": "BTC-USD", "ethereum": "ETH-USD",

    # Retail
    "walmart": "WMT", "target": "TGT", "costco": "COST",
    "starbucks": "SBUX", "mcdonalds": "MCD", "nike": "NKE",
}


def extract_ticker(title: str, summary: str) -> Optional[str]:
    """Scans article text for known company names, returns first ticker found."""
    text = f"{title} {summary}".lower()
    for company, ticker in COMPANY_TICKER_MAP.items():
        if company in text:
            return ticker
    return None


def get_price_movement(ticker: str, article_date: datetime, days: int = 3) -> Optional[Dict]:
    """
    Fetches 3-day average price change after article publication.
    Returns label: bullish / bearish / neutral
    """
    try:
        start = article_date - timedelta(days=2)
        end = article_date + timedelta(days=days + 5)

        hist = yf.Ticker(ticker).history(start=start, end=end)

        if hist.empty or len(hist) < 2:
            return None

        before = hist[hist.index.date <= article_date.date()]
        after = hist[hist.index.date > article_date.date()]

        if before.empty or after.empty:
            return None

        price_before = before["Close"].iloc[-1]
        price_after = after.head(days)["Close"].mean()
        pct_change = ((price_after - price_before) / price_before) * 100

        if pct_change > 1.0:
            label = "bullish"
        elif pct_change < -1.0:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "ticker":       ticker,
            "price_before": round(price_before, 4),
            "price_after":  round(price_after, 4),
            "pct_change":   round(pct_change, 4),
            "label":        label
        }

    except Exception as e:
        logger.warning(f"Price fetch failed for {ticker}: {e}")
        return None


def enrich_with_stock_data(df: pd.DataFrame, days: int = 3) -> pd.DataFrame:
    """
    Adds stock price columns to Phase 2 CSV dataframe.
    New columns: ticker, price_before, price_after, pct_change, price_label
    """
    logger.info(f"Enriching {len(df)} articles with stock data...")

    results = {"ticker": [], "price_before": [], "price_after": [],
               "pct_change": [], "price_label": []}
    matched = 0

    for _, row in df.iterrows():
        ticker = extract_ticker(str(row.get("title", "")), str(row.get("summary", "")))

        if not ticker:
            for k in results:
                results[k].append(None)
            continue

        try:
            article_date = pd.to_datetime(row["published"])
            if article_date.tzinfo is not None:
                article_date = article_date.tz_localize(None)
        except Exception:
            article_date = datetime.utcnow() - timedelta(days=7)

        data = get_price_movement(ticker, article_date, days=days)

        if data:
            matched += 1
            for k in results:
                results[k].append(data[k] if k != "price_label" else data["label"])
        else:
            results["ticker"].append(ticker)
            for k in ["price_before", "price_after", "pct_change", "price_label"]:
                results[k].append(None)

    df = df.copy()
    for col, vals in results.items():
        df[col] = vals

    logger.info(f"Matched: {matched}/{len(df)} articles with stock data")
    if not df["price_label"].dropna().empty:
        for label, count in df["price_label"].value_counts().items():
            logger.info(f"  {label:<12} {count}")

    return df
