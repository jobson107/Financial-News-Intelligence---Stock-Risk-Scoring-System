import feedparser
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict
from config.settings import RSS_FEEDS, INGESTION_BATCH_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)

# Only ingest articles from the last 30 days
CUTOFF_DAYS = 30


def _parse_published_date(entry: dict) -> datetime:
    """
    RSS feeds return dates in inconsistent formats.
    feedparser normalizes to a time.struct_time — convert to datetime.
    Falls back to utcnow() if missing.
    """
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6])
    return datetime.utcnow()


def _clean_text(text: str) -> str:
    """Strip HTML tags and extra whitespace from summaries."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_recent(published: datetime) -> bool:
    """
    Returns True if the article is within the CUTOFF_DAYS window.
    Filters out stale articles that would skew the NLP pipeline.
    """
    cutoff = datetime.utcnow() - timedelta(days=CUTOFF_DAYS)
    return published >= cutoff


def fetch_from_rss(source_name: str, url: str, retries: int = 3) -> List[Dict]:
    """
    Fetches articles from a single RSS feed with retry logic.

    Args:
        source_name : key from RSS_FEEDS dict (used as MongoDB 'source' field)
        url         : RSS feed URL
        retries     : number of attempts before giving up (default 3)

    Returns a list of dicts — each dict is one MongoDB document.
    """
    logger.info(f"Fetching: {source_name}")

    feed = None
    for attempt in range(retries):
        try:
            feed = feedparser.parse(url)

            # feed.bozo = True means malformed XML — still has partial data
            if feed.bozo:
                logger.warning(f"{source_name}: malformed feed (bozo=True), proceeding anyway")

            # If we got entries, no need to retry
            if feed.entries:
                break
            else:
                logger.warning(f"{source_name}: empty feed on attempt {attempt + 1}")

        except Exception as e:
            logger.warning(f"{source_name}: attempt {attempt + 1}/{retries} failed → {e}")
            if attempt < retries - 1:
                time.sleep(2)  # wait 2 seconds before retrying

    if not feed or not feed.entries:
        logger.error(f"{source_name}: all {retries} attempts failed, skipping")
        return []

    articles = []
    skipped_old = 0

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        published = _parse_published_date(entry)

        # Skip articles older than CUTOFF_DAYS
        if not _is_recent(published):
            skipped_old += 1
            continue

        doc = {
            "title": title,
            "summary": _clean_text(entry.get("summary", "")),
            "source": source_name,
            "link": entry.get("link", ""),
            "published": published,
            "ingested_at": datetime.utcnow(),
            "processed": False,
            "sentiment": None,
            "keywords": [],
            "tags": [t.get("term", "") for t in entry.get("tags", [])],
        }
        articles.append(doc)

    if skipped_old:
        logger.info(f"{source_name}: skipped {skipped_old} articles older than {CUTOFF_DAYS} days")

    logger.info(f"{source_name}: fetched {len(articles)} recent articles")
    return articles


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
        df = df.dropna(subset=["title"])

        column_map = {
            "description": "summary",
            "content": "summary",
            "publishedAt": "published",
            "date": "published",
            "url": "link",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        for col, default in [("summary", ""), ("source", "csv_import"), ("link", "")]:
            if col not in df.columns:
                df[col] = default

        articles = []
        skipped_old = 0

        for _, row in df.iterrows():
            published = pd.to_datetime(row.get("published", datetime.utcnow()), errors="coerce")
            if pd.isna(published):
                published = datetime.utcnow()
            else:
                published = published.to_pydatetime()

            # Apply same date filter to CSV imports
            if not _is_recent(published):
                skipped_old += 1
                continue

            doc = {
                "title": str(row["title"]).strip(),
                "summary": _clean_text(str(row.get("summary", ""))),
                "source": str(row.get("source", "csv_import")),
                "link": str(row.get("link", "")),
                "published": published,
                "ingested_at": datetime.utcnow(),
                "processed": False,
                "sentiment": None,
                "keywords": [],
                "tags": [],
            }
            articles.append(doc)

        if skipped_old:
            logger.info(f"CSV: skipped {skipped_old} articles older than {CUTOFF_DAYS} days")

        logger.info(f"CSV: loaded {len(articles)} recent articles from {filepath}")
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