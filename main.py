"""
Financial News Intelligence System

Usage:
    python main.py --phase 1              # Ingest RSS feeds
    python main.py --phase 2              # NLP pipeline
    python main.py --phase 3              # ML models (LR, RF, XGBoost)
    python main.py --phase 3b             # PyTorch LSTM
    python main.py --phase 4              # Stock data + forecasting
    python main.py --phase all            # Run all phases
"""

import argparse
from data_ingestion.pipeline import IngestionPipeline
from nlp.nlp_pipeline import NLPPipeline
from ml.ml_pipeline import MLPipeline
from ml.phase3b_pipeline import Phase3BPipeline
from forecasting.phase4_pipeline import Phase4Pipeline


def main():
    parser = argparse.ArgumentParser(description="Financial News Intelligence System")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "3b", "4", "all"],
        default="1",
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        IngestionPipeline().run(use_csv=args.csv)

    if args.phase in ("2", "all"):
        NLPPipeline().run()

    if args.phase in ("3", "all"):
        MLPipeline().run(csv_path=args.csv)

    if args.phase in ("3b", "all"):
        Phase3BPipeline().run(csv_path=args.csv)

    if args.phase in ("4", "all"):
        Phase4Pipeline().run(csv_path=args.csv)


if __name__ == "__main__":
    main()
