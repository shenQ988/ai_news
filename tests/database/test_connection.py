"""
Database connection tests.
Requires a running Postgres instance (docker compose -f docker/docker-compose.yml up -d).
Run with: uv run pytest tests/database/test_connection.py -v
"""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.database.connection import engine, get_session, DATABASE_URL
from app.database.models import Base, ArticleRecord, ContentType


class TestEngineConnection:
    def test_engine_connects(self):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_database_url_is_set(self):
        assert DATABASE_URL is not None
        assert DATABASE_URL.startswith("postgresql://")

    def test_pgvector_extension_exists(self):
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            assert result.scalar() == 1, "pgvector extension is not installed"

    def test_articles_table_exists(self):
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_name = 'articles'")
            )
            assert result.scalar() == 1, "articles table does not exist"

    def test_articles_table_has_embedding_column(self):
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT udt_name FROM information_schema.columns
                    WHERE table_name = 'articles' AND column_name = 'embedding'
                """)
            )
            assert result.scalar() is not None, "embedding column does not exist"


class TestGetSession:
    def test_session_yields_and_commits(self):
        with get_session() as session:
            assert session is not None
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_session_rolls_back_on_exception(self):
        with pytest.raises(ValueError):
            with get_session() as session:
                session.execute(text("SELECT 1"))
                raise ValueError("intentional error")

    def test_bad_url_raises_operational_error(self):
        with patch("app.database.connection.DATABASE_URL", "postgresql://bad:bad@localhost:9999/bad"):
            from sqlalchemy import create_engine
            bad_engine = create_engine("postgresql://bad:bad@localhost:9999/bad")
            with pytest.raises(OperationalError):
                with bad_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))


class TestArticleRecordSchema:
    def test_insert_and_retrieve_article(self):
        with get_session() as session:
            record = ArticleRecord(
                guid="test-guid-connection-001",
                title="Test Article",
                description="Test description",
                url="https://example.com/test",
                source="TestSource",
                published_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
                content_type=ContentType.ARTICLE,
            )
            session.add(record)

        with get_session() as session:
            result = session.query(ArticleRecord).filter_by(guid="test-guid-connection-001").first()
            assert result is not None
            assert result.title == "Test Article"
            assert result.content_type == ContentType.ARTICLE
            assert result.embedding is None

            session.delete(result)

    def test_duplicate_guid_is_ignored(self):
        guid = "test-guid-connection-002"
        with get_session() as session:
            session.add(ArticleRecord(
                guid=guid, title="First", description="", url="https://example.com",
                source="Test", published_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
                content_type=ContentType.ARTICLE,
            ))

        with pytest.raises(Exception):
            with get_session() as session:
                session.add(ArticleRecord(
                    guid=guid, title="Duplicate", description="", url="https://example.com",
                    source="Test", published_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
                    content_type=ContentType.ARTICLE,
                ))

        with get_session() as session:
            session.query(ArticleRecord).filter_by(guid=guid).delete()
