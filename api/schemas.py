from pydantic import BaseModel, Field
from typing import Optional, Dict, List


# ── POST /predict ─────────────────────────────────────────

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=5, description="Article headline")
    summary: str = Field("", description="Article body or summary (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Tesla faces supply chain disruption amid chip shortage",
                "summary": "Tesla reported significant production delays due to semiconductor shortage."
            }
        }


class SentimentDetail(BaseModel):
    compound: float
    label: str
    positive: float
    negative: float
    neutral: float


class RiskDetail(BaseModel):
    score: float
    label: str
    keyword_risk: float
    neg_keywords: List[str]


class LSTMDetail(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    title: str
    sector: str
    sentiment: SentimentDetail
    risk: RiskDetail
    price_movement_xgboost: Optional[str] = None
    price_movement_confidence: Optional[float] = None
    price_movement_lstm: Optional[LSTMDetail] = None
    ticker_detected: Optional[str] = None


# ── GET /forecast ─────────────────────────────────────────

class ForecastPoint(BaseModel):
    date: str
    value: float
    lower: float
    upper: float


class ModelForecast(BaseModel):
    model_name: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    points: List[ForecastPoint]


class ForecastResponse(BaseModel):
    target: str
    days_forecasted: int
    arima: Optional[ModelForecast] = None
    prophet: Optional[ModelForecast] = None
    winner_mae: Optional[str] = None
    winner_rmse: Optional[str] = None


# ── GET /stats ────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_articles: int
    processed: int
    unprocessed: int
    by_source: Dict[str, int]
    risk_distribution: Optional[Dict[str, int]] = None
    sentiment_distribution: Optional[Dict[str, int]] = None
