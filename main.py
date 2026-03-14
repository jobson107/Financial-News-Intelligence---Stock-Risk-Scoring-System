"""
Financial News Intelligence System

Usage:
    # Phase 1 - Ingest from RSS feeds
    python main.py --phase 1

    # Phase 1 - Ingest from CSV
    python main.py --phase 1 --csv data/all_the_news.csv

    # Phase 2 - NLP Pipeline
    python main.py --phase 2

    # Run both phases sequentially
    python main.py --phase all
"""

import argparse
from data_ingestion.pipeline import IngestionPipeline
from nlp.nlp_pipeline import NLPPipeline


def main():
    parser = argparse.ArgumentParser(description="Financial News Intelligence System")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all"],
        default="1",
        help="Which phase to run: 1 (ingestion), 2 (NLP), all (both)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to Kaggle CSV file for Phase 1 (optional)"
    )
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        pipeline = IngestionPipeline()
        pipeline.run(use_csv=args.csv)

    if args.phase in ("2", "all"):
        nlp = NLPPipeline()
        nlp.run()


if __name__ == "__main__":
    main()
