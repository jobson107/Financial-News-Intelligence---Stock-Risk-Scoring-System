"""
Financial News Intelligence System

Usage:
    python main.py --phase 1              # Ingest from RSS feeds
    python main.py --phase 1 --csv path   # Ingest from CSV
    python main.py --phase 2              # NLP pipeline
    python main.py --phase 3              # ML models
    python main.py --phase all            # Run all phases
"""

import argparse
from data_ingestion.pipeline import IngestionPipeline
from nlp.nlp_pipeline import NLPPipeline
from ml.ml_pipeline import MLPipeline


def main():
    parser = argparse.ArgumentParser(description="Financial News Intelligence System")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "all"],
        default="1",
        help="Which phase to run: 1 (ingestion), 2 (NLP), 3 (ML), all (all phases)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (Phase 1: Kaggle dataset | Phase 3: override NLP output)"
    )
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        pipeline = IngestionPipeline()
        pipeline.run(use_csv=args.csv)

    if args.phase in ("2", "all"):
        nlp = NLPPipeline()
        nlp.run()

    if args.phase in ("3", "all"):
        ml = MLPipeline()
        ml.run(csv_path=args.csv)


if __name__ == "__main__":
    main()
