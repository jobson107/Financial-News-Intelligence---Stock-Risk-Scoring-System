import pandas as pd
import os
from datetime import datetime
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "outputs"


def export_to_csv(results: List[Dict]) -> str:
    """
    Exports NLP-processed articles to a CSV file.

    Flattens nested sentiment dict into individual columns
    so it's directly usable by sklearn/pandas in Phase 3.

    Output columns:
        title, source, published, sector,
        sentiment_compound, sentiment_label,
        sentiment_positive, sentiment_negative, sentiment_neutral,
        risk_score, risk_label, sentiment_risk, keyword_risk,
        keywords, neg_keywords_hit
    """
    if not results:
        logger.warning("No results to export")
        return ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for r in results:
        sentiment = r.get("sentiment", {})
        rows.append({
            "title":               r.get("title", ""),
            "source":              r.get("source", ""),
            "published":           r.get("published", ""),
            "sector":              r.get("sector", ""),

            # Sentiment columns (flattened from nested dict)
            "sentiment_compound":  sentiment.get("compound", 0),
            "sentiment_label":     sentiment.get("label", ""),
            "sentiment_positive":  sentiment.get("positive", 0),
            "sentiment_negative":  sentiment.get("negative", 0),
            "sentiment_neutral":   sentiment.get("neutral", 0),

            # Risk columns
            "risk_score":          r.get("risk_score", 0),
            "risk_label":          r.get("risk_label", ""),
            "sentiment_risk":      r.get("sentiment_risk", 0),
            "keyword_risk":        r.get("keyword_risk", 0),

            # Keyword columns (stored as pipe-separated strings)
            "keywords":            "|".join(r.get("keywords", [])),
            "neg_keywords_hit":    "|".join(r.get("neg_keywords_hit", [])),
        })

    df = pd.DataFrame(rows)

    # Filename includes timestamp so each run creates a new file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f"nlp_results_{timestamp}.csv")

    df.to_csv(filepath, index=False)
    logger.info(f"Exported {len(df)} articles to {filepath}")

    # Print summary stats
    logger.info(f"Risk distribution:")
    risk_counts = df["risk_label"].value_counts()
    for label, count in risk_counts.items():
        logger.info(f"  {label:<10} {count}")

    logger.info(f"Sector distribution:")
    sector_counts = df["sector"].value_counts()
    for sector, count in sector_counts.items():
        logger.info(f"  {sector:<12} {count}")

    logger.info(f"Avg sentiment compound: {df['sentiment_compound'].mean():.4f}")
    logger.info(f"Avg risk score:         {df['risk_score'].mean():.4f}")

    return filepath
