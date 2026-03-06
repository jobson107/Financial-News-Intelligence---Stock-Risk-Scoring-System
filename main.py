"""
Financial News Intelligence System
Phase 1: Data Ingestion & MongoDB Storage

Usage:
    # Ingest from RSS feeds (default)
    python main.py

    # Ingest from a Kaggle CSV file
    python main.py --csv data/all_the_news.csv
"""

import argparse
from data_ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Financial News Ingestion Pipeline")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to Kaggle CSV file (optional). Defaults to RSS feeds."
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    pipeline.run(use_csv=args.csv)


if __name__ == "__main__":
    main()
