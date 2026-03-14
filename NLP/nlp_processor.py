from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Risk Keywords ─────────────────────────────────────────
# Negative keywords that amplify risk score regardless of sentiment
NEGATIVE_RISK_KEYWORDS = {
    "crisis", "crash", "fraud", "bankruptcy", "default", "collapse",
    "scandal", "investigation", "lawsuit", "recall", "layoffs", "downgrade",
    "loss", "deficit", "debt", "inflation", "recession", "tariff", "sanction",
    "hack", "breach", "fine", "penalty", "warning", "risk", "concern", "fear",
    "decline", "drop", "plunge", "selloff", "volatility", "uncertainty"
}

# Sector-specific keywords mapped to risk signals
SECTOR_KEYWORDS = {
    "tech":     {"ai", "semiconductor", "cloud", "software", "hardware", "cybersecurity", "chip", "data"},
    "finance":  {"bank", "fed", "interest rate", "bond", "credit", "lending", "mortgage", "hedge"},
    "energy":   {"oil", "gas", "opec", "renewable", "solar", "pipeline", "crude", "barrel"},
    "health":   {"fda", "drug", "clinical", "trial", "pharma", "vaccine", "approval", "biotech"},
    "macro":    {"gdp", "inflation", "unemployment", "federal reserve", "treasury", "cpi", "ppi"},
}


class NLPProcessor:
    """
    Runs the full NLP pipeline on a batch of articles:
        1. VADER sentiment scoring
        2. TF-IDF keyword extraction
        3. Risk score calculation
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(
            max_features=50,        # top 50 terms across the corpus
            stop_words="english",   # remove common English words
            ngram_range=(1, 2),     # unigrams + bigrams (e.g. "interest rate")
            min_df=1                # include terms that appear at least once
        )

    def get_sentiment(self, text: str) -> Dict:
        """
        VADER sentiment analysis.

        Returns:
            {
                "compound": float,  # -1 (most negative) to +1 (most positive)
                "positive": float,
                "negative": float,
                "neutral": float,
                "label": str        # "positive" | "negative" | "neutral"
            }

        VADER compound score thresholds:
            >= 0.05  → positive
            <= -0.05 → negative
            between  → neutral
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "compound": round(compound, 4),
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral":  round(scores["neu"], 4),
            "label":    label
        }

    def extract_keywords_tfidf(self, texts: List[str]) -> List[List[str]]:
        """
        TF-IDF keyword extraction across a batch of articles.

        TF-IDF works on a corpus (multiple documents) — it finds words
        that are important in one article but rare across others.
        Running on a single article gives poor results.

        Returns: list of keyword lists, one per article
        """
        if not texts:
            return []

        # Handle edge case: single article
        if len(texts) == 1:
            # fallback to simple frequency for single doc
            words = texts[0].lower().split()
            words = [w for w in words if len(w) > 4]
            return [list(set(words))[:10]]

        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()

            keywords_per_doc = []
            for i in range(tfidf_matrix.shape[0]):
                # Get TF-IDF scores for this document
                row = tfidf_matrix.getrow(i).toarray()[0]

                # Sort by score descending, take top 10
                top_indices = row.argsort()[::-1][:10]
                top_keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                keywords_per_doc.append(top_keywords)

            return keywords_per_doc

        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            return [[] for _ in texts]

    def detect_sector(self, text: str) -> str:
        """
        Detects which financial sector an article belongs to
        based on keyword presence.
        Returns the sector with most keyword matches, or 'general'.
        """
        text_lower = text.lower()
        sector_scores = {}

        for sector, keywords in SECTOR_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                sector_scores[sector] = score

        if not sector_scores:
            return "general"

        return max(sector_scores, key=sector_scores.get)

    def calculate_risk_score(
        self,
        sentiment: Dict,
        keywords: List[str],
        text: str
    ) -> Dict:
        """
        Combines three signals into a final risk score (0.0 to 1.0):

        1. Sentiment signal    (40% weight) — negative sentiment = higher risk
        2. Negative keywords   (40% weight) — crisis/fraud/crash keywords
        3. Sector signal       (20% weight) — volatile sectors get slight boost

        Risk score interpretation:
            0.0 - 0.3 → Low risk
            0.3 - 0.6 → Medium risk
            0.6 - 1.0 → High risk
        """
        # Signal 1: Sentiment (inverted — negative sentiment = high risk)
        # compound ranges -1 to +1, we want 0 to 1 where 1 = high risk
        sentiment_risk = (1 - sentiment["compound"]) / 2  # maps to 0-1

        # Signal 2: Negative keyword density
        text_lower = text.lower()
        all_words = text_lower.split()
        if all_words:
            neg_hits = sum(1 for kw in NEGATIVE_RISK_KEYWORDS if kw in text_lower)
            keyword_risk = min(neg_hits / 5, 1.0)  # cap at 1.0 (5+ hits = max risk)
        else:
            keyword_risk = 0.0

        # Signal 3: Sector volatility boost
        sector = self.detect_sector(text)
        sector_boost = {
            "finance": 0.1,
            "energy":  0.1,
            "macro":   0.15,
            "tech":    0.05,
            "health":  0.05,
            "general": 0.0
        }.get(sector, 0.0)

        # Weighted combination
        raw_score = (
            0.40 * sentiment_risk +
            0.40 * keyword_risk +
            0.20 * (sector_boost / 0.15)  # normalize sector boost to 0-1
        )

        final_score = round(min(raw_score, 1.0), 4)

        # Risk label
        if final_score >= 0.6:
            risk_label = "high"
        elif final_score >= 0.3:
            risk_label = "medium"
        else:
            risk_label = "low"

        return {
            "risk_score":      final_score,
            "risk_label":      risk_label,
            "sentiment_risk":  round(sentiment_risk, 4),
            "keyword_risk":    round(keyword_risk, 4),
            "sector":          sector,
            "neg_keywords_hit": [kw for kw in NEGATIVE_RISK_KEYWORDS if kw in text_lower]
        }

    def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Runs the full NLP pipeline on a batch of articles.

        Args:
            articles: list of dicts with 'title', 'summary', '_id'

        Returns:
            list of enriched dicts with sentiment, keywords, risk_score
        """
        if not articles:
            return []

        logger.info(f"Processing batch of {len(articles)} articles")

        # Combine title + summary for richer NLP signal
        texts = [
            f"{a.get('title', '')} {a.get('summary', '')}".strip()
            for a in articles
        ]

        # TF-IDF runs on the whole batch at once
        keywords_list = self.extract_keywords_tfidf(texts)

        results = []
        for i, (article, text, keywords) in enumerate(zip(articles, texts, keywords_list)):
            sentiment = self.get_sentiment(text)
            risk = self.calculate_risk_score(sentiment, keywords, text)

            result = {
                "_id":             article["_id"],
                "title":           article.get("title", ""),
                "source":          article.get("source", ""),
                "published":       article.get("published"),
                "sentiment":       sentiment,
                "keywords":        keywords,
                "risk_score":      risk["risk_score"],
                "risk_label":      risk["risk_label"],
                "sentiment_risk":  risk["sentiment_risk"],
                "keyword_risk":    risk["keyword_risk"],
                "sector":          risk["sector"],
                "neg_keywords_hit": risk["neg_keywords_hit"],
            }
            results.append(result)

        logger.info(f"Batch processed: {len(results)} articles")
        return results
