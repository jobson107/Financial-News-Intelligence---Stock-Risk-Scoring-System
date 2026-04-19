"""
Financial News Intelligence System
Phase 5: FastAPI REST API

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict   — article text → sentiment + risk + price prediction (XGBoost + LSTM)
    GET  /forecast  — 7-day ARIMA + Prophet forecast
    GET  /stats     — MongoDB collection statistics
    GET  /health    — health check
    GET  /docs      — Swagger UI
"""

import glob
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    PredictRequest, PredictResponse,
    ForecastResponse, ModelForecast, ForecastPoint,
    StatsResponse
)
from api.predictor import Predictor
from data_ingestion.mongo_handler import MongoHandler
from forecasting.forecaster import prepare_time_series, run_forecasting
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Financial News Intelligence API",
    description="NLP-powered financial news risk scoring, sentiment analysis, and price movement prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()


@app.on_event("startup")
async def startup_event():
    predictor.load_models()
    logger.info("API started — all models loaded")


# ── GET /health ───────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Health check — confirms API and models are running."""
    return {
        "status":        "ok",
        "models_loaded": predictor.is_ready(),
        "lstm_loaded":   predictor.lstm.is_ready(),
        "version":       "1.0.0"
    }


# ── POST /predict ─────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Given article title and summary, returns:
    - Sentiment score + label
    - Risk score + label + negative keywords detected
    - Price movement prediction from XGBoost
    - Price movement prediction from LSTM (if trained)
    - Stock ticker detected from text

    Example input:
        { "title": "Tesla faces supply chain issues", "summary": "..." }
    """
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        return predictor.predict(request.title, request.summary)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /forecast ─────────────────────────────────────────

@app.get("/forecast", tags=["Forecasting"])
def forecast():
    """
    Runs ARIMA and Prophet forecasting on historical sentiment
    and risk trends from your article database.

    Returns 7-day predictions with confidence intervals.
    Compares ARIMA vs Prophet on MAE and RMSE.
    """
    try:
        csvs = glob.glob("outputs/nlp_results_*.csv")
        if not csvs:
            raise HTTPException(
                status_code=404,
                detail="No Phase 2 CSV found. Run Phase 2 first."
            )

        csv_path = max(csvs, key=os.path.getsize)
        df = pd.read_csv(csv_path)
        daily_df = prepare_time_series(df)

        if len(daily_df) < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(daily_df)} days of data. Need at least 5."
            )

        results = run_forecasting(daily_df, forecast_days=7)
        response = []

        for target_label, result in results.items():
            arima = result.get("arima", {})
            prophet = result.get("prophet", {})
            comp = result.get("comparison", {})
            forecast_dates = prophet.get("forecast_dates", [])

            def build_points(res, dates):
                points = []
                for i, val in enumerate(res.get("forecast", [])):
                    lowers = res.get("conf_int_lower", [])
                    uppers = res.get("conf_int_upper", [])
                    points.append(ForecastPoint(
                        date=dates[i] if i < len(dates) else f"Day {i+1}",
                        value=round(float(val), 4),
                        lower=round(float(lowers[i]) if i < len(lowers) else val, 4),
                        upper=round(float(uppers[i]) if i < len(uppers) else val, 4),
                    ))
                return points

            response.append(ForecastResponse(
                target=target_label,
                days_forecasted=7,
                arima=ModelForecast(
                    model_name="ARIMA(1,1,1)",
                    mae=arima.get("mae"),
                    rmse=arima.get("rmse"),
                    points=build_points(arima, forecast_dates)
                ) if arima.get("success") else None,
                prophet=ModelForecast(
                    model_name="Prophet",
                    mae=prophet.get("mae"),
                    rmse=prophet.get("rmse"),
                    points=build_points(prophet, forecast_dates)
                ) if prophet.get("success") else None,
                winner_mae=comp.get("winner_mae"),
                winner_rmse=comp.get("winner_rmse"),
            ))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /stats ────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
def stats():
    """
    Returns MongoDB collection statistics:
    - Total articles, processed/unprocessed counts
    - Breakdown by source
    - Risk and sentiment label distributions
    """
    try:
        mongo = MongoHandler()
        base = mongo.get_stats()

        risk_dist = {}
        sentiment_dist = {}

        try:
            for doc in mongo._collection.aggregate([
                {"$match": {"risk_label": {"$exists": True}}},
                {"$group": {"_id": "$risk_label", "count": {"$sum": 1}}}
            ]):
                if doc["_id"]:
                    risk_dist[doc["_id"]] = doc["count"]

            for doc in mongo._collection.aggregate([
                {"$match": {"sentiment_label": {"$exists": True}}},
                {"$group": {"_id": "$sentiment_label", "count": {"$sum": 1}}}
            ]):
                if doc["_id"]:
                    sentiment_dist[doc["_id"]] = doc["count"]
        except Exception:
            pass

        mongo.disconnect()

        return StatsResponse(
            total_articles=base["total_articles"],
            processed=base["processed"],
            unprocessed=base["unprocessed"],
            by_source=base["by_source"],
            risk_distribution=risk_dist or None,
            sentiment_distribution=sentiment_dist or None,
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
