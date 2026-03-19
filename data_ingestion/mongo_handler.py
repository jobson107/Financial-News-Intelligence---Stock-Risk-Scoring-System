from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, BulkWriteError
from datetime import datetime
from typing import List, Dict, Optional
from config.settings import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_RAW
from utils.logger import get_logger

logger = get_logger(__name__)


class MongoHandler:
    """
    Handles all MongoDB operations for the ingestion layer.
    Keeps all DB logic here — pipeline.py and fetcher.py never
    touch pymongo directly. This makes it easy to swap the DB later.
    """

    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._db = None
        self._collection = None
        self.connect()

    def connect(self) -> None:
        """
        Establishes connection to MongoDB Atlas (or local).
        Call this once at startup.
        """
        try:
            self._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self._client.admin.command("ping")

            self._db = self._client[MONGO_DB_NAME]
            self._collection = self._db[MONGO_COLLECTION_RAW]

            self._collection.create_index(
                [("source", ASCENDING), ("title", ASCENDING)],
                unique=True,
                background=True
            )
            self._collection.create_index([("processed", ASCENDING)], background=True)
            self._collection.create_index([("ingested_at", ASCENDING)], background=True)

            logger.info(f"Connected to MongoDB | DB: {MONGO_DB_NAME} | Collection: {MONGO_COLLECTION_RAW}")

        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            logger.error("Check your MONGO_URI in .env and that your Atlas cluster is running.")
            raise

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed.")

    def insert_articles(self, articles: List[Dict]) -> Dict:
        """
        Bulk insert articles. Skips duplicates silently.
        Returns: {"inserted": int, "duplicates": int, "errors": int}
        """
        if not articles:
            return {"inserted": 0, "duplicates": 0, "errors": 0}

        # Validate before inserting
        articles = self._validate_articles(articles)

        inserted = 0
        duplicates = 0
        errors = 0

        try:
            result = self._collection.insert_many(articles, ordered=False)
            inserted = len(result.inserted_ids)

        except BulkWriteError as bwe:
            inserted = bwe.details.get("nInserted", 0)
            for error in bwe.details.get("writeErrors", []):
                if error.get("code") == 11000:
                    duplicates += 1
                else:
                    errors += 1
                    logger.warning(f"Insert error (non-duplicate): {error.get('errmsg')}")

        return {"inserted": inserted, "duplicates": duplicates, "errors": errors}

    def get_unprocessed(self, limit: int = 500) -> List[Dict]:
        """
        Returns articles not yet processed by the NLP pipeline.
        Phase 2 calls this.
        """
        cursor = self._collection.find(
            {"processed": False},
            {"_id": 1, "title": 1, "summary": 1, "source": 1, "published": 1}
        ).limit(limit)
        return list(cursor)

    def mark_processed(self, doc_id, sentiment_score: float, keywords: List[str], extra_fields: dict = None) -> None:
        """
        Called by Phase 2 NLP pipeline after processing an article.
        Saves sentiment, keywords, risk score and all NLP results back to MongoDB.
        """
        update_data = {
            "processed": True,
            "sentiment": sentiment_score,
            "keywords": keywords,
            "processed_at": datetime.utcnow()
        }

        # Phase 2 extra fields: risk_score, risk_label, sector, etc.
        if extra_fields:
            update_data.update(extra_fields)

        self._collection.update_one(
            {"_id": doc_id},
            {"$set": update_data}
        )

    def get_stats(self) -> Dict:
        """Returns collection-level stats — useful for monitoring."""
        total = self._collection.count_documents({})
        processed = self._collection.count_documents({"processed": True})
        unprocessed = total - processed

        pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        by_source = {doc["_id"]: doc["count"] for doc in self._collection.aggregate(pipeline)}

        return {
            "total_articles": total,
            "processed": processed,
            "unprocessed": unprocessed,
            "by_source": by_source
        }

    def _validate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filters out documents missing required fields before insert."""
        valid = []
        for doc in articles:
            if not doc.get("title") or len(doc["title"]) < 5:
                continue
            if not doc.get("source"):
                continue
            valid.append(doc)
        logger.info(f"Validation: {len(valid)}/{len(articles)} passed")
        return valid