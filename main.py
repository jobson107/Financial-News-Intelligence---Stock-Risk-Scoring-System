"""
Financial News Intelligence System

Usage:
    python main.py --phase 1
    python main.py --phase 1 --csv data/all_the_news.csv
    python main.py --phase 2
    python main.py --phase 3
    python main.py --phase 4
    python main.py --phase all
"""

import argparse
from data_ingestion.pipeline import IngestionPipeline
from nlp.nlp_pipeline import NLPPipeline
from ml.ml_pipeline import MLPipeline
from forecasting.phase4_pipeline import Phase4Pipeline


def main():
    parser = argparse.ArgumentParser(description="Financial News Intelligence System")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="1",
        help="Which phase to run"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file"
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

    if args.phase in ("4", "all"):
        p4 = Phase4Pipeline()
        p4.run(csv_path=args.csv)


if __name__ == "__main__":
    main()
