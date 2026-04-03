import glob
import os
import json
import pandas as pd
from datetime import datetime
from forecasting.stock_fetcher import enrich_with_stock_data
from forecasting.forecaster import prepare_time_series, run_forecasting
from ml.feature_engineering import load_and_prepare
from ml.trainer import train_and_evaluate
from ml.evaluator import generate_report
from utils.logger import get_logger

logger = get_logger(__name__)


class Phase4Pipeline:
    """
    Phase 4 Pipeline:
        4A: Stock price integration → real target variable → retrain models
        4B: ARIMA + Prophet forecasting on sentiment/risk trends
    """

    def run(self, csv_path: str = None) -> None:
        logger.info("=" * 50)
        logger.info("PHASE 4: Stock Integration + Forecasting")
        logger.info("=" * 50)

        # ── Find latest Phase 2 CSV ───────────────────────
        if not csv_path:
            csvs = glob.glob("outputs/nlp_results_*.csv")
            if not csvs:
                logger.error("No Phase 2 CSV found. Run Phase 2 first.")
                return
            csv_path = max(csvs, key=os.path.getsize)
            logger.info(f"Using CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} articles")

        # ════════════════════════════════════════
        # PHASE 4A: Stock Price Integration
        # ════════════════════════════════════════
        logger.info("\n" + "─" * 40)
        logger.info("PHASE 4A: Stock Price Integration")
        logger.info("─" * 40)

        df_enriched = enrich_with_stock_data(df, days=3)

        # Save enriched CSV
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        enriched_path = f"outputs/enriched_{timestamp}.csv"
        df_enriched.to_csv(enriched_path, index=False)
        logger.info(f"Enriched CSV saved: {enriched_path}")

        # ── Retrain models with real stock labels ─
        df_with_labels = df_enriched.dropna(subset=["price_label"])
        logger.info(f"Articles with stock labels: {len(df_with_labels)}")

        if len(df_with_labels) >= 30:
            logger.info("Retraining models with real stock price labels...")
            X, y = load_and_prepare_stock(df_with_labels)

            if X is not None and len(X) >= 30:
                stock_results = train_and_evaluate(X, y["price_label"], "price_movement")

                report_data = [stock_results]
                generate_report(report_data)
                logger.info("Stock-based model training complete")
            else:
                logger.warning("Not enough matched articles for retraining")
        else:
            logger.warning(f"Only {len(df_with_labels)} articles matched — need 30+ for retraining")
            logger.info("Tip: Run Phase 1 again to ingest more articles, then re-run Phase 4")

        # ════════════════════════════════════════
        # PHASE 4B: Time Series Forecasting
        # ════════════════════════════════════════
        logger.info("\n" + "─" * 40)
        logger.info("PHASE 4B: ARIMA + Prophet Forecasting")
        logger.info("─" * 40)

        daily_df = prepare_time_series(df)

        if len(daily_df) < 5:
            logger.error("Need at least 5 days of data for forecasting. Run Phase 1 over multiple days.")
            return

        forecast_results = run_forecasting(daily_df, forecast_days=7)

        # Save forecast results as JSON
        forecast_path = f"outputs/forecast_{timestamp}.json"
        with open(forecast_path, "w") as f:
            json.dump(forecast_results, f, indent=2, default=str)
        logger.info(f"Forecast results saved: {forecast_path}")

        # ── Summary ───────────────────────────────
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 4 SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Enriched CSV     : {enriched_path}")
        logger.info(f"Forecast JSON    : {forecast_path}")

        for target, result in forecast_results.items():
            comp = result.get("comparison", {})
            logger.info(f"\n{target}:")
            logger.info(f"  ARIMA   MAE: {comp.get('arima_mae')} | RMSE: {comp.get('arima_rmse')}")
            logger.info(f"  Prophet MAE: {comp.get('prophet_mae')} | RMSE: {comp.get('prophet_rmse')}")
            logger.info(f"  Winner (MAE): {comp.get('winner_mae')}")
            logger.info(f"  Chart: {result.get('chart')}")

        logger.info("=" * 50)
        logger.info("Phase 4 complete.")


def load_and_prepare_stock(df: pd.DataFrame):
    """
    Feature engineering for stock price prediction.
    Uses NLP features to predict price_label (bullish/bearish/neutral).
    No leakage — price_label is external data from yfinance.
    """
    from collections import Counter

    required_cols = ["sentiment_compound", "risk_score", "keyword_risk",
                     "sector", "price_label"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return None, None

    df = df.dropna(subset=required_cols)

    # Remove classes with fewer than 2 samples
    counts = Counter(df["price_label"])
    valid = [cls for cls, cnt in counts.items() if cnt >= 2]
    removed = [cls for cls, cnt in counts.items() if cnt < 2]
    if removed:
        logger.warning(f"Removing price_label classes with < 2 samples: {removed}")
    df = df[df["price_label"].isin(valid)]

    if len(df) < 10:
        logger.error("Not enough data after filtering")
        return None, None

    # Feature set — NO leakage since price_label is from yfinance
    feature_cols = [
        "sentiment_compound", "sentiment_positive", "sentiment_negative",
        "sentiment_neutral", "risk_score", "sentiment_risk", "keyword_risk"
    ]

    sector_dummies = pd.get_dummies(df["sector"], prefix="sector")
    df = pd.concat([df.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
    sector_cols = [c for c in df.columns if c.startswith("sector_")]
    feature_cols.extend(sector_cols)

    df["sentiment_strength"] = df["sentiment_compound"].abs()
    df["neg_dominance"] = df["sentiment_negative"] - df["sentiment_positive"]
    df["combined_risk"] = (df["risk_score"] + df["sentiment_risk"]) / 2
    feature_cols.extend(["sentiment_strength", "neg_dominance", "combined_risk"])

    X = df[feature_cols].fillna(0)
    y = df[["price_label"]].copy()

    logger.info(f"Stock model features: {len(feature_cols)} columns")
    logger.info(f"Price label distribution:\n{y['price_label'].value_counts().to_string()}")

    return X, y
