# Financial News Intelligence System
## Phase 1: Data Ingestion & Storage

### Project Structure
```
financial_news_intelligence/
├── config/
│   └── settings.py          # All config: MongoDB URI, feed URLs, etc.
├── data_ingestion/
│   ├── __init__.py
│   ├── fetcher.py           # Pulls articles from RSS feeds + Kaggle CSV
│   ├── mongo_handler.py     # All MongoDB read/write operations
│   └── pipeline.py          # Orchestrates fetcher → mongo
├── utils/
│   ├── __init__.py
│   └── logger.py            # Centralized logging
├── tests/
│   └── test_phase1.py       # Validation tests
├── main.py                  # Entry point
├── requirements.txt
└── .env.example             # Template for credentials
```

### Setup Instructions

**Step 1: Install dependencies**
```bash
pip install pymongo feedparser pandas python-dotenv requests
```

**Step 2: MongoDB Atlas (free)**
1. Go to https://www.mongodb.com/atlas
2. Create free account → Create free cluster (M0)
3. Database Access → Add user → username/password
4. Network Access → Add IP → 0.0.0.0/0 (allow all for dev)
5. Connect → Drivers → Copy connection string

**Step 3: Configure environment**
```bash
cp .env.example .env
# Edit .env with your MongoDB Atlas URI
```

**Step 4: Run ingestion**
```bash
python main.py
```

### What Phase 1 Delivers
- Raw articles stored in MongoDB (`raw_articles` collection)
- Deduplication via (source, title) unique index
- Ingestion logs with counts per source
- `processed: False` flag ready for Phase 2 NLP pipeline
- MS SQL feature table schema (ready for Phase 3)
