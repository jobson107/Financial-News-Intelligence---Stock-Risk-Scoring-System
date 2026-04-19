# Financial News Intelligence & Stock Risk Scoring System

An end-to-end machine learning system that ingests financial news, scores sentiment and risk using NLP, predicts stock price movement, and serves predictions via a REST API with a real-time dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA FLOW                            │
│                                                             │
│  RSS Feeds → MongoDB → NLP Pipeline → Feature Store (CSV)  │
│                 ↓                           ↓               │
│           yfinance API            ML Models (pkl)           │
│                 ↓                           ↓               │
│          Stock Labels              ARIMA + Prophet          │
│                 ↓                           ↓               │
│              FastAPI (/predict /forecast /stats)            │
│                              ↓                              │
│                    Streamlit Dashboard                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── config/
│   └── settings.py              # MongoDB URI, RSS feeds, constants
├── data_ingestion/
│   ├── fetcher.py               # RSS + CSV ingestion with retry logic
│   ├── mongo_handler.py         # All MongoDB CRUD operations
│   └── pipeline.py              # Phase 1 orchestrator
├── nlp/
│   ├── nlp_processor.py         # VADER sentiment + TF-IDF + risk scoring
│   ├── nlp_pipeline.py          # Phase 2 orchestrator
│   └── exporter.py              # CSV export
├── ml/
│   ├── feature_engineering.py   # Feature matrix + class filtering
│   ├── trainer.py               # LR + RF + XGBoost training + evaluation
│   ├── ml_pipeline.py           # Phase 3 orchestrator
│   ├── lstm_model.py            # PyTorch LSTM architecture
│   ├── lstm_trainer.py          # LSTM sequence builder + training loop
│   ├── lstm_predictor.py        # LSTM inference
│   └── phase3b_pipeline.py      # Phase 3B orchestrator
├── forecasting/
│   ├── stock_fetcher.py         # yfinance integration + ticker matching
│   ├── forecaster.py            # ARIMA + Prophet + chart generation
│   └── phase4_pipeline.py       # Phase 4 orchestrator
├── api/
│   ├── main.py                  # FastAPI app + all endpoints
│   ├── schemas.py               # Pydantic request/response models
│   └── predictor.py             # Model inference layer
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── models/                      # Saved .pkl and .pt model files
├── outputs/
│   ├── charts/                  # Forecast PNG charts
│   ├── reports/                 # Evaluation JSON + TXT reports
│   └── nlp_results_*.csv        # Phase 2 output
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Phases

| Phase | Description | Key Technologies |
|-------|-------------|-----------------|
| 1 | RSS ingestion → MongoDB storage | feedparser, pymongo, MongoDB Atlas |
| 2 | Sentiment scoring + risk scoring | VADER, TF-IDF, scikit-learn |
| 3 | ML classification models | Logistic Regression, Random Forest, XGBoost |
| 3B | Sequential risk prediction | PyTorch LSTM |
| 4A | Real stock label integration | yfinance, pandas |
| 4B | Time series forecasting | ARIMA, Prophet, matplotlib |
| 5 | REST API deployment | FastAPI, uvicorn, Pydantic |
| 7 | Interactive dashboard | Streamlit, Plotly |

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/jobson107/Financial-News-Intelligence---Stock-Risk-Scoring-System.git
cd Financial-News-Intelligence---Stock-Risk-Scoring-System

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your MongoDB Atlas connection string
```

### 3. Run pipeline phases
```bash
# Phase 1 — Ingest articles
python main.py --phase 1

# Phase 2 — NLP pipeline
python main.py --phase 2

# Phase 3 — ML models
python main.py --phase 3

# Phase 3B — PyTorch LSTM
python main.py --phase 3b

# Phase 4 — Stock data + forecasting
python main.py --phase 4

# All phases sequentially
python main.py --phase all
```

### 4. Start FastAPI
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
API docs: http://localhost:8000/docs

### 5. Start Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard: http://localhost:8501

---

## API Endpoints

### POST /predict
Given article title and summary, returns sentiment, risk score, and price movement prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Tesla faces supply chain disruption", "summary": "Production delays expected..."}'
```

Response:
```json
{
  "title": "Tesla faces supply chain disruption",
  "sector": "tech",
  "sentiment": {
    "compound": -0.6249,
    "label": "negative",
    "positive": 0.0,
    "negative": 0.263,
    "neutral": 0.737
  },
  "risk": {
    "score": 0.612,
    "label": "high",
    "keyword_risk": 0.6,
    "neg_keywords": ["disruption", "delay"]
  },
  "price_movement_xgboost": "bearish",
  "price_movement_confidence": 0.612,
  "price_movement_lstm": {
    "label": "bearish",
    "confidence": 0.581,
    "probabilities": {"bearish": 0.581, "bullish": 0.22, "neutral": 0.199}
  },
  "ticker_detected": "TSLA"
}
```

### GET /forecast
Returns 7-day ARIMA and Prophet forecasts for sentiment and risk trends.

### GET /stats
Returns MongoDB collection statistics including article counts and label distributions.

### GET /health
Returns API health status and model loading confirmation.

---

## Key Results

| Target | Model | F1 Score | Notes |
|--------|-------|----------|-------|
| Risk Label | Random Forest | 99.08% | Data leakage — target from Phase 2 |
| Sentiment Label | Random Forest | 100% | Data leakage — target from Phase 2 |
| Price Movement | XGBoost | 51.36% | Real yfinance labels — no leakage |
| Price Movement | LSTM | TBD | Sequential model — needs more data |

**Note on accuracy:** The 99-100% scores on risk and sentiment labels are due to data leakage — those labels were derived from the same features used for training. The 51% F1 on price movement prediction uses real external stock data and is the honest, production-relevant metric.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.9 |
| Data Storage | MongoDB Atlas |
| NLP | VADER, TF-IDF (scikit-learn) |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| Deep Learning | PyTorch LSTM |
| Time Series | ARIMA (statsmodels), Prophet |
| Stock Data | yfinance |
| API | FastAPI, uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib, Plotly |

---

## Certifications
- Oracle Generative AI Professional (2025)
- Deloitte Data Analytics Job Simulation (2025)
- IBM Data Analysis with Python

---

## Author
**Jobson K Joby**
B.Tech Computer Science — SRM Institute of Science and Technology, Delhi NCR
GitHub: [@jobson107](https://github.com/jobson107)
