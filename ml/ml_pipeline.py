import glob
import os
from ml.feature_engineering import load_and_prepare
from ml.trainer import train_and_evaluate
from ml.evaluator import generate_report
from utils.logger import get_logger

logger = get_logger(__name__)


class MLPipeline:
    """
    Phase 3 Pipeline:
        1. Load latest Phase 2 CSV from outputs/
        2. Engineer features
        3. Train Logistic Regression, Random Forest, XGBoost
        4. Evaluate all models on risk_label + sentiment_label
        5. Save best models as .pkl
        6. Generate evaluation report
    """

    def run(self, csv_path: str = None) -> None:
        logger.info("=" * 50)
        logger.info("PHASE 3: ML Pipeline Starting")
        logger.info("=" * 50)

        # ── Find latest CSV if not specified ──────────────
        if not csv_path:
            csvs = glob.glob("outputs/nlp_results_*.csv")
            if not csvs:
                logger.error("No Phase 2 CSV found in outputs/. Run Phase 2 first.")
                return
            csv_path = max(csvs, key=os.path.getctime)
            logger.info(f"Using CSV: {csv_path}")

        # ── Feature Engineering ───────────────────────────
        X, y = load_and_prepare(csv_path)

        if len(X) < 50:
            logger.error(f"Not enough data: only {len(X)} rows. Need at least 50.")
            return

        # ── Train both targets ────────────────────────────
        all_results = []

        # Target 1: Risk label (low / medium / high)
        risk_results = train_and_evaluate(X, y["risk_label"], "risk_label")
        all_results.append(risk_results)

        # Target 2: Sentiment label (positive / negative / neutral)
        sentiment_results = train_and_evaluate(X, y["sentiment_label"], "sentiment_label")
        all_results.append(sentiment_results)

        # ── Generate Report ───────────────────────────────
        report_path = generate_report(all_results)

        # ── Summary ───────────────────────────────────────
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 3 SUMMARY")
        logger.info("=" * 50)
        for result in all_results:
            logger.info(
                f"{result['target']:<20} "
                f"Best: {result['best_model']:<25} "
                f"F1: {result['best_f1']:.4f}"
            )
        logger.info(f"Report: {report_path}")
        logger.info(f"Models saved in: models/")
        logger.info("=" * 50)
        logger.info("Phase 3 complete.")
