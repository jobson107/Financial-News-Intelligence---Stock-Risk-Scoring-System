import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ──────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "financial_news")
MONGO_COLLECTION_RAW = os.getenv("MONGO_COLLECTION_RAW", "raw_articles")

# ── RSS Feed Sources ──────────────────────────────────────
# These are free, no API key needed
RSS_FEEDS = {
    "yahoo_finance":       "https://finance.yahoo.com/news/rssindex",
    "reuters_business":    "https://feeds.reuters.com/reuters/businessNews",
    "marketwatch":         "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "seeking_alpha":       "https://seekingalpha.com/feed.xml",
    "investing_com":       "https://www.investing.com/rss/news.rss",
}

# ── Ingestion Settings ────────────────────────────────────
INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", 100))

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
