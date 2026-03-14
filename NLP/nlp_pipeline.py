from data_ingestion.mongo_handler import MongoHandler
from nlp.nlp_processor import NLPProcessor
from nlp.exporter import export_to_csv
from utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 50  # process 50 articles at a time to avoid memory issues


class NLPPipeline:
    """
    Phase 2 Pipeline:
        1. Pull unprocessed articles from MongoDB
        2. Run NLP (VADER + TF-IDF + Risk Score)
        3. Write results back to MongoDB
        4. Export all results to CSV
    """

    def __init__(self):
        self.mongo = MongoHandler()
        self.processor = NLPProcessor()

    def run(self) -> None:
        logger.info("=" * 50)
        logger.info("PHASE 2: NLP Pipeline Starting")
        logger.info("=" * 50)

        # Step 1: Connect to MongoDB
        self.mongo.connect()

        # Step 2: Get all unprocessed articles
        articles = self.mongo.get_unprocessed(limit=1000)
        total = len(articles)

        if total == 0:
            logger.info("No unprocessed articles found. Run Phase 1 first.")
            self.mongo.disconnect()
            return

        logger.info(f"Found {total} unprocessed articles")

        # Step 3: Process in batches
        all_results = []
        for i in range(0, total, BATCH_SIZE):
            batch = articles[i: i + BATCH_SIZE]
            logger.info(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} articles)...")
            results = self.processor.process_batch(batch)
            all_results.extend(results)

            # Step 4: Write batch results back to MongoDB immediately
            for result in results:
                self.mongo.mark_processed(
                    doc_id=result["_id"],
                    sentiment_score=result["sentiment"]["compound"],
                    keywords=result["keywords"],
                    extra_fields={
                        "risk_score":       result["risk_score"],
                        "risk_label":       result["risk_label"],
                        "sentiment_label":  result["sentiment"]["label"],
                        "sentiment_pos":    result["sentiment"]["positive"],
                        "sentiment_neg":    result["sentiment"]["negative"],
                        "sector":           result["sector"],
                        "neg_keywords_hit": result["neg_keywords_hit"],
                    }
                )

        logger.info("-" * 40)
        logger.info(f"NLP complete: {len(all_results)} articles processed")

        # Step 5: Export to CSV
        csv_path = export_to_csv(all_results)
        logger.info(f"CSV saved to: {csv_path}")

        # Step 6: Final stats
        stats = self.mongo.get_stats()
        logger.info("-" * 40)
        logger.info(f"COLLECTION STATS:")
        logger.info(f"  Total in DB  : {stats['total_articles']}")
        logger.info(f"  Processed    : {stats['processed']}")
        logger.info(f"  Unprocessed  : {stats['unprocessed']}")
        logger.info("=" * 50)
        logger.info("Phase 2 complete.")

        self.mongo.disconnect()
