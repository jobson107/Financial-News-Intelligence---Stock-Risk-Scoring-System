import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def load_and_prepare(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the Phase 2 CSV and engineers features for ML models.

    Returns two dataframes:
        X → feature matrix
        y → target labels (risk_label + sentiment_label)
    """
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Drop rows with missing targets ───────────────────
    df = df.dropna(subset=["risk_label", "sentiment_label"])
    logger.info(f"After dropping nulls: {len(df)} rows")

    # ── Remove classes with fewer than 2 samples ─────────
    # Classes with only 1 sample crash train/test split and CV
    # They are also statistically meaningless for training
    for col in ["risk_label", "sentiment_label"]:
        counts = Counter(df[col])
        valid_classes = [cls for cls, count in counts.items() if count >= 2]
        removed = [cls for cls, count in counts.items() if count < 2]
        if removed:
            logger.warning(f"{col}: removing classes with < 2 samples: {removed}")
        df = df[df[col].isin(valid_classes)]

    logger.info(f"After class filtering: {len(df)} rows")

    # ── Feature Engineering ───────────────────────────────

    # 1. Core NLP scores (already computed in Phase 2)
    feature_cols = [
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "risk_score",
        "sentiment_risk",
        "keyword_risk",
    ]

    # 2. Sector one-hot encoding
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector")
    df = pd.concat([df, sector_dummies], axis=1)
    sector_cols = [c for c in df.columns if c.startswith("sector_")]
    feature_cols.extend(sector_cols)

    # 3. Derived features
    df["sentiment_strength"] = df["sentiment_compound"].abs()
    df["neg_dominance"] = df["sentiment_negative"] - df["sentiment_positive"]
    df["combined_risk"] = (df["risk_score"] + df["sentiment_risk"]) / 2

    feature_cols.extend(["sentiment_strength", "neg_dominance", "combined_risk"])

    # ── Fill missing values ───────────────────────────────
    X = df[feature_cols].fillna(0)

    # ── Target labels ─────────────────────────────────────
    y = df[["risk_label", "sentiment_label"]].copy()

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Risk label distribution:\n{y['risk_label'].value_counts().to_string()}")
    logger.info(f"Sentiment label distribution:\n{y['sentiment_label'].value_counts().to_string()}")

    return X, y
