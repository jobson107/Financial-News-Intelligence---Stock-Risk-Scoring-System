from data_ingestion.fetcher import fetch_all_rss, fetch_from_csv
from data_ingestion.mongo_handler import MongoHandler
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full Phase 1 pipeline:
        fetch → validate → insert → report

    This is the only class main.py needs to know about.
    """

    def __init__(self):
        self.mongo = MongoHandler()

    def run(self, use_csv: str = None) -> None:
        """
        Main entry point.

        Args:
            use_csv: Optional path to a Kaggle CSV file.
                     If None, ingests from RSS feeds.
        """
        logger.info("=" * 50)
        logger.info("PHASE 1: Ingestion Pipeline Starting")
        logger.info("=" * 50)

        # Step 1: Connect to MongoDB
        self.mongo.connect()

        # Step 2: Fetch articles
        if use_csv:
            logger.info(f"Source: CSV file → {use_csv}")
            articles = fetch_from_csv(use_csv)
        else:
            logger.info("Source: RSS feeds")
            articles = fetch_all_rss()

        if not articles:
            logger.warning("No articles fetched. Check feed URLs or CSV path.")
            return

        # Step 3: Insert into MongoDB
        logger.info(f"Inserting {len(articles)} articles into MongoDB...")
        result = self.mongo.insert_articles(articles)

        # Step 4: Report results
        logger.info("-" * 40)
        logger.info(f"RESULTS:")
        logger.info(f"  Inserted  : {result['inserted']}")
        logger.info(f"  Duplicates: {result['duplicates']} (skipped)")
        logger.info(f"  Errors    : {result['errors']}")
        logger.info("-" * 40)

        # Step 5: Collection stats
        stats = self.mongo.get_stats()
        logger.info(f"COLLECTION STATS:")
        logger.info(f"  Total in DB   : {stats['total_articles']}")
        logger.info(f"  Processed     : {stats['processed']}")
        logger.info(f"  Unprocessed   : {stats['unprocessed']} (ready for Phase 2 NLP)")
        logger.info(f"  By source     :")
        for source, count in stats["by_source"].items():
            logger.info(f"    {source:<25} {count}")

        logger.info("=" * 50)
        logger.info("Phase 1 complete.")

        # Disconnect cleanly
        self.mongo.disconnect()
