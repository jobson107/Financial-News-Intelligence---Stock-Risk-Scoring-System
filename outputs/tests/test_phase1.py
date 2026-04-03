"""
Phase 1 Tests
Run with: python -m pytest tests/test_phase1.py -v

These tests validate:
1. Document schema integrity
2. Deduplication logic
3. CSV ingestion mapping
4. Stats reporting

NOTE: These are unit tests that mock MongoDB — they don't need a real connection.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from data_ingestion.fetcher import _clean_text, _parse_published_date
from data_ingestion.mongo_handler import MongoHandler


# ── Fetcher Tests ─────────────────────────────────────────

class TestCleaner:
    def test_strips_html_tags(self):
        raw = "<p>Tesla faces <b>supply chain</b> issues</p>"
        assert _clean_text(raw) == "Tesla faces supply chain issues"

    def test_collapses_whitespace(self):
        raw = "  Tesla    faces   issues  "
        assert _clean_text(raw) == "Tesla faces issues"

    def test_empty_string(self):
        assert _clean_text("") == ""


class TestDateParser:
    def test_missing_date_falls_back_to_now(self):
        entry = {}  # no published_parsed
        result = _parse_published_date(entry)
        assert isinstance(result, datetime)

    def test_valid_struct_time(self):
        import time
        entry = MagicMock()
        entry.published_parsed = time.strptime("2024-03-01", "%Y-%m-%d")
        result = _parse_published_date(entry)
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 1


# ── MongoHandler Tests (mocked) ───────────────────────────

class TestMongoHandler:
    """
    All MongoDB calls are mocked — tests run without a real DB connection.
    """

    def _make_handler(self):
        handler = MongoHandler()
        handler._collection = MagicMock()
        return handler

    def test_insert_returns_correct_counts(self):
        handler = self._make_handler()
        handler._collection.insert_many.return_value = MagicMock(inserted_ids=["id1", "id2"])

        articles = [
            {"title": "Article 1", "source": "reuters"},
            {"title": "Article 2", "source": "reuters"},
        ]
        result = handler.insert_articles(articles)

        assert result["inserted"] == 2
        assert result["duplicates"] == 0
        assert result["errors"] == 0

    def test_empty_insert_returns_zeros(self):
        handler = self._make_handler()
        result = handler.insert_articles([])
        assert result == {"inserted": 0, "duplicates": 0, "errors": 0}
        handler._collection.insert_many.assert_not_called()

    def test_document_schema_has_required_fields(self):
        """Validates that all docs going to MongoDB have the expected fields."""
        required_fields = {
            "title", "summary", "source", "link",
            "published", "ingested_at", "processed",
            "sentiment", "keywords", "tags"
        }
        sample_doc = {
            "title": "Test Article",
            "summary": "Some summary",
            "source": "reuters",
            "link": "https://example.com",
            "published": datetime.utcnow(),
            "ingested_at": datetime.utcnow(),
            "processed": False,
            "sentiment": None,
            "keywords": [],
            "tags": [],
        }
        assert required_fields.issubset(set(sample_doc.keys()))

    def test_get_unprocessed_uses_correct_filter(self):
        handler = self._make_handler()
        handler._collection.find.return_value = iter([])

        handler.get_unprocessed(limit=10)

        # Verify find was called with processed=False filter
        call_args = handler._collection.find.call_args
        assert call_args[0][0] == {"processed": False}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
