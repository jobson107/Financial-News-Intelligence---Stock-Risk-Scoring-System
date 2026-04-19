import glob
import os
import json
import pandas as pd
from datetime import datetime
from ml.lstm_trainer import train_lstm
from utils.logger import get_logger

logger = get_logger(__name__)


class Phase3BPipeline:
    """
    Phase 3B: PyTorch LSTM training.

    Runs after Phase 4 (needs price_label from yfinance).
    Compares LSTM against XGBoost/RF/LR from Phase 3.
    """

    def run(self, csv_path: str = None) -> None:
        logger.info("=" * 50)
        logger.info("PHASE 3B: PyTorch LSTM Training")
        logger.info("=" * 50)

        # Use enriched CSV (has price_label from yfinance)
        if not csv_path:
            csvs = glob.glob("outputs/enriched_*.csv")
            if not csvs:
                logger.error("No enriched CSV found. Run Phase 4 first.")
                return
            csv_path = max(csvs, key=os.path.getsize)
            logger.info(f"Using CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} articles")

        matched = df.dropna(subset=["price_label"])
        logger.info(f"Articles with price labels: {len(matched)}")

        if len(matched) < 10:
            logger.error("Need at least 10 articles with price labels. Run Phase 4 first.")
            return

        # Train LSTM
        result = train_lstm(matched)

        if not result.get("success"):
            logger.error(f"LSTM training failed: {result.get('error')}")
            return

        # Save results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = f"outputs/reports/lstm_evaluation_{timestamp}.json"
        os.makedirs("outputs/reports", exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"\nLSTM Report saved: {report_path}")
        logger.info("=" * 50)
        logger.info("PHASE 3B SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model     : LSTM (PyTorch)")
        logger.info(f"Accuracy  : {result['accuracy']}")
        logger.info(f"F1        : {result['f1_weighted']}")
        logger.info(f"Compare vs XGBoost Phase 3: F1=0.5136")
        logger.info("=" * 50)
        logger.info("Phase 3B complete.")
