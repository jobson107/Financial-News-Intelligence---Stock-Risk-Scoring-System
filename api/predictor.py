import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict
from nlp.nlp_processor import NLPProcessor
from forecasting.stock_fetcher import extract_ticker
from ml.lstm_predictor import LSTMPredictor
from api.schemas import PredictResponse, SentimentDetail, RiskDetail, LSTMDetail
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATHS = {
    "price_movement": {
        "model":   "models/price_movement_best_model.pkl",
        "encoder": "models/price_movement_label_encoder.pkl",
    },
    "risk_label": {
        "model":   "models/risk_label_best_model.pkl",
        "encoder": "models/risk_label_label_encoder.pkl",
    },
}

ALL_SECTORS = ["sector_energy", "sector_finance", "sector_general",
               "sector_health", "sector_macro", "sector_tech"]


class Predictor:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.nlp = NLPProcessor()
        self.lstm = LSTMPredictor()
        self._ready = False

    def load_models(self) -> None:
        for target, paths in MODEL_PATHS.items():
            try:
                with open(paths["model"], "rb") as f:
                    self.models[target] = pickle.load(f)
                with open(paths["encoder"], "rb") as f:
                    self.encoders[target] = pickle.load(f)
                logger.info(f"Loaded: {target}")
            except FileNotFoundError:
                logger.warning(f"Model not found: {paths['model']}")

        # Load LSTM (optional — won't crash if not trained yet)
        self.lstm.load()

        self._ready = len(self.models) > 0

    def is_ready(self) -> bool:
        return self._ready

    def _build_features(self, sentiment: Dict, risk: Dict, sector: str) -> pd.DataFrame:
        features = {
            "sentiment_compound":  sentiment["compound"],
            "sentiment_positive":  sentiment["positive"],
            "sentiment_negative":  sentiment["negative"],
            "sentiment_neutral":   sentiment["neutral"],
            "risk_score":          risk["risk_score"],
            "sentiment_risk":      risk["sentiment_risk"],
            "keyword_risk":        risk["keyword_risk"],
            "sentiment_strength":  abs(sentiment["compound"]),
            "neg_dominance":       sentiment["negative"] - sentiment["positive"],
            "combined_risk":       (risk["risk_score"] + risk["sentiment_risk"]) / 2,
        }
        for sec_col in ALL_SECTORS:
            features[sec_col] = 1 if sec_col == f"sector_{sector}" else 0
        return pd.DataFrame([features])

    def predict(self, title: str, summary: str) -> PredictResponse:
        text = f"{title} {summary}".strip()

        # NLP
        sentiment = self.nlp.get_sentiment(text)
        sector = self.nlp.detect_sector(text)
        risk = self.nlp.calculate_risk_score(sentiment, [], text)
        ticker = extract_ticker(title, summary)

        # Feature vector
        X = self._build_features(sentiment, risk, sector)

        # XGBoost price prediction
        price_movement = None
        price_confidence = None
        if "price_movement" in self.models:
            model = self.models["price_movement"]
            encoder = self.encoders["price_movement"]
            model_features = getattr(model, "feature_names_in_", X.columns)
            X_aligned = X.reindex(columns=model_features, fill_value=0)
            pred = model.predict(X_aligned)[0]
            price_movement = encoder.inverse_transform([pred])[0]
            try:
                proba = model.predict_proba(X_aligned)[0]
                price_confidence = round(float(np.max(proba)), 4)
            except Exception:
                pass

        # LSTM price prediction
        lstm_result = None
        if self.lstm.is_ready():
            lstm_output = self.lstm.predict(
                sentiment["compound"],
                risk["risk_score"],
                risk["keyword_risk"]
            )
            if lstm_output:
                lstm_result = LSTMDetail(
                    label=lstm_output["label"],
                    confidence=lstm_output["confidence"],
                    probabilities=lstm_output["probabilities"]
                )

        return PredictResponse(
            title=title,
            sector=sector,
            sentiment=SentimentDetail(
                compound=sentiment["compound"],
                label=sentiment["label"],
                positive=sentiment["positive"],
                negative=sentiment["negative"],
                neutral=sentiment["neutral"],
            ),
            risk=RiskDetail(
                score=risk["risk_score"],
                label=risk["risk_label"],
                keyword_risk=risk["keyword_risk"],
                neg_keywords=risk["neg_keywords_hit"][:5],
            ),
            price_movement_xgboost=price_movement,
            price_movement_confidence=price_confidence,
            price_movement_lstm=lstm_result,
            ticker_detected=ticker,
        )
