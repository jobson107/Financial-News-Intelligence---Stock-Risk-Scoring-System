import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ────────────────────────────────────────────────────────
from urllib.parse import quote_plus

_user = quote_plus(os.getenv("MONGO_USERNAME", "jobsonjoby7_db_user"))
_pass = quote_plus(os.getenv("MONGO_PASSWORD", "KingJobson@9637"))
_host = os.getenv("MONGO_HOST", "cluster1.vajk63h.mongodb.net")

MONGODB_URI = (
    f"mongodb+srv://{_user}:{_pass}@{_host}/"
    f"?retryWrites=true&w=majority&appName=Cluster1"
)

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
