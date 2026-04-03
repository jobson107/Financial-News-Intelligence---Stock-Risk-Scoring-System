import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)

CHARTS_DIR = "outputs/charts"


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates article-level data into daily time series.
    Returns daily averages of sentiment_compound and risk_score.
    """
    df = df.copy()
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "sentiment_compound", "risk_score"])
    df["date"] = df["published"].dt.date

    daily = df.groupby("date").agg(
        avg_sentiment=("sentiment_compound", "mean"),
        avg_risk=("risk_score", "mean"),
        article_count=("title", "count")
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    logger.info(f"Time series: {len(daily)} daily data points")
    logger.info(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

    return daily


def run_arima(series: pd.Series, forecast_days: int = 7) -> Dict:
    """
    Fits ARIMA model on a time series and forecasts next N days.

    Uses auto-selected order (1,1,1) as a sensible default for
    short financial time series. In production you'd use auto_arima.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Fit ARIMA(1,1,1) — order=(p,d,q)
        # p=1: one autoregressive term
        # d=1: first difference (removes trend)
        # q=1: one moving average term
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()

        forecast = fitted.forecast(steps=forecast_days)
        conf_int = fitted.get_forecast(steps=forecast_days).conf_int()

        # In-sample metrics
        residuals = fitted.resid
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))

        logger.info(f"ARIMA fitted | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        return {
            "model": "ARIMA(1,1,1)",
            "forecast": forecast.tolist(),
            "conf_int_lower": conf_int.iloc[:, 0].tolist(),
            "conf_int_upper": conf_int.iloc[:, 1].tolist(),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "success": True
        }

    except Exception as e:
        logger.error(f"ARIMA failed: {e}")
        return {"success": False, "error": str(e)}


def run_prophet(daily_df: pd.DataFrame, target_col: str, forecast_days: int = 7) -> Dict:
    """
    Fits Facebook Prophet model and forecasts next N days.

    Prophet requires columns named 'ds' (date) and 'y' (value).
    Handles missing dates, holidays, and seasonality automatically.
    """
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        prophet_df = pd.DataFrame({
            "ds": daily_df["date"],
            "y": daily_df[target_col]
        }).dropna()

        model = Prophet(
            yearly_seasonality=False,  # not enough data for yearly
            weekly_seasonality=True,   # financial news has weekly patterns
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Get only future predictions
        future_forecast = forecast.tail(forecast_days)

        # In-sample MAE
        in_sample = forecast[forecast["ds"].isin(prophet_df["ds"])]
        mae = np.mean(np.abs(in_sample["yhat"].values - prophet_df["y"].values))
        rmse = np.sqrt(np.mean((in_sample["yhat"].values - prophet_df["y"].values) ** 2))

        logger.info(f"Prophet fitted | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        return {
            "model": "Prophet",
            "forecast": future_forecast["yhat"].tolist(),
            "conf_int_lower": future_forecast["yhat_lower"].tolist(),
            "conf_int_upper": future_forecast["yhat_upper"].tolist(),
            "forecast_dates": future_forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "success": True
        }

    except Exception as e:
        logger.error(f"Prophet failed: {e}")
        return {"success": False, "error": str(e)}


def plot_forecasts(daily_df: pd.DataFrame, arima_result: Dict,
                   prophet_result: Dict, target_col: str, label: str) -> str:
    """
    Creates a comparison chart: historical data + ARIMA + Prophet forecasts.
    Saves as PNG and returns filepath.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(CHARTS_DIR, f"{label}_forecast_{timestamp}.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Financial News {label} Forecast\nARIMA vs Prophet", fontsize=14)

    forecast_days = 7
    last_date = daily_df["date"].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq="D")[1:]

    for ax, (model_name, result) in zip(axes, [("ARIMA", arima_result), ("Prophet", prophet_result)]):
        if not result.get("success"):
            ax.text(0.5, 0.5, f"{model_name} failed", ha="center", va="center")
            continue

        # Historical
        ax.plot(daily_df["date"], daily_df[target_col],
                color="#2196F3", linewidth=1.5, label="Historical", alpha=0.8)

        # Forecast
        forecast_vals = result["forecast"]
        dates = pd.to_datetime(result.get("forecast_dates", future_dates))

        ax.plot(dates, forecast_vals,
                color="#FF5722", linewidth=2, linestyle="--", label=f"{model_name} Forecast")

        # Confidence interval
        ax.fill_between(dates,
                         result["conf_int_lower"],
                         result["conf_int_upper"],
                         alpha=0.2, color="#FF5722", label="95% CI")

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"{model_name} | MAE: {result['mae']:.4f} | RMSE: {result['rmse']:.4f}")
        ax.set_ylabel(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Chart saved: {filepath}")
    return filepath


def run_forecasting(daily_df: pd.DataFrame, forecast_days: int = 7) -> Dict:
    """
    Runs ARIMA + Prophet on both sentiment and risk trends.
    Returns all results and chart paths.
    """
    results = {}

    for target_col, label in [("avg_sentiment", "Sentiment"), ("avg_risk", "Risk Score")]:
        logger.info(f"\nForecasting: {label}")
        series = daily_df[target_col].ffill()
        if len(series) < 5:
            logger.warning(f"Not enough data points for {label} forecasting (need 5+)")
            continue

        arima_result = run_arima(series, forecast_days)
        prophet_result = run_prophet(daily_df, target_col, forecast_days)

        chart_path = plot_forecasts(
            daily_df, arima_result, prophet_result,
            target_col, label
        )

        results[label] = {
            "arima": arima_result,
            "prophet": prophet_result,
            "chart": chart_path,
            "comparison": {
                "arima_mae":   arima_result.get("mae"),
                "prophet_mae": prophet_result.get("mae"),
                "arima_rmse":  arima_result.get("rmse"),
                "prophet_rmse": prophet_result.get("rmse"),
                "winner_mae":  "ARIMA" if (arima_result.get("mae", 999) < prophet_result.get("mae", 999)) else "Prophet",
                "winner_rmse": "ARIMA" if (arima_result.get("rmse", 999) < prophet_result.get("rmse", 999)) else "Prophet",
            }
        }

        logger.info(f"{label} — ARIMA MAE: {arima_result.get('mae')} | Prophet MAE: {prophet_result.get('mae')}")
        logger.info(f"{label} — Winner (MAE): {results[label]['comparison']['winner_mae']}")

    return results
