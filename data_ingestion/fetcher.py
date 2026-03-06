import feedparser
import pandas as pd
from datetime import datetime
from typing import List, Dict
from config.settings import RSS_FEEDS, INGESTION_BATCH_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)


def _parse_published_date(entry: dict) -> datetime:
    """
    RSS feeds return dates in inconsistent formats.
    feedparser normalizes to a time.struct_time — convert to datetime.
    Falls back to utcnow() if missing.
    """
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6])  # unpack year,month,day,hour,min,sec
    return datetime.utcnow()


def _clean_text(text: str) -> str:
    """Strip HTML tags and extra whitespace from summaries."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)   # remove HTML tags
    text = re.sub(r"\s+", " ", text)         # collapse whitespace
    return text.strip()


def fetch_from_rss(source_name: str, url: str) -> List[Dict]:
    """
    Fetches articles from a single RSS feed.

    Returns a list of dicts — each dict is one MongoDB document.
    All fields are explicit so downstream code can rely on them.
    """
    logger.info(f"Fetching: {source_name}")

    try:
        # feedparser handles HTTP, encoding, and malformed XML gracefully
        feed = feedparser.parse(url)

        if feed.bozo:
            # bozo=True means feedparser encountered a malformed feed
            # Still returns partial data, so we continue with a warning
            logger.warning(f"{source_name}: malformed feed (bozo=True), proceeding anyway")

        articles = []
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if not title:
                continue  # skip entries with no title

            doc = {
                # Core fields
                "title": title,
                "summary": _clean_text(entry.get("summary", "")),
                "source": source_name,
                "link": entry.get("link", ""),

                # Dates
                "published": _parse_published_date(entry),
                "ingested_at": datetime.utcnow(),

                # Phase 2 NLP pipeline flags — set defaults here
                "processed": False,
                "sentiment": None,
                "keywords": [],

                # Metadata
                "tags": [t.get("term", "") for t in entry.get("tags", [])],
            }
            articles.append(doc)

        logger.info(f"{source_name}: fetched {len(articles)} articles")
        return articles

    except Exception as e:
        # Don't crash the whole pipeline if one feed fails
        logger.error(f"{source_name}: fetch failed → {e}")
        return []


def fetch_from_csv(filepath: str) -> List[Dict]:
    """
    Alternative ingestion path: load from a Kaggle CSV dataset.

    Expected CSV columns (flexible — maps what's available):
        title, description/summary, source, publishedAt/date, url/link

    Use this if RSS feeds are blocked or you want historical data.
    Kaggle dataset: 'All the News' or 'Financial News Dataset'
    """
    logger.info(f"Loading CSV: {filepath}")

    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=["title"])  # title is mandatory

        # Flexible column mapping — handles different Kaggle CSV schemas
        column_map = {
            "description": "summary",
            "content": "summary",
            "publishedAt": "published",
            "date": "published",
            "url": "link",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Fill missing columns with defaults
        for col, default in [("summary", ""), ("source", "csv_import"), ("link", "")]:
            if col not in df.columns:
                df[col] = default

        articles = []
        for _, row in df.iterrows():
            doc = {
                "title": str(row["title"]).strip(),
                "summary": _clean_text(str(row.get("summary", ""))),
                "source": str(row.get("source", "csv_import")),
                "link": str(row.get("link", "")),
                "published": pd.to_datetime(row.get("published", datetime.utcnow()), errors="coerce") or datetime.utcnow(),
                "ingested_at": datetime.utcnow(),
                "processed": False,
                "sentiment": None,
                "keywords": [],
                "tags": [],
            }
            articles.append(doc)

        logger.info(f"CSV: loaded {len(articles)} articles from {filepath}")
        return articles

    except Exception as e:
        logger.error(f"CSV ingestion failed: {e}")
        return []


def fetch_all_rss() -> List[Dict]:
    """Fetches from all configured RSS feeds and returns combined list."""
    all_articles = []
    for source_name, url in RSS_FEEDS.items():
        articles = fetch_from_rss(source_name, url)
        all_articles.extend(articles)

    logger.info(f"Total fetched across all feeds: {len(all_articles)}")
    return all_articles
