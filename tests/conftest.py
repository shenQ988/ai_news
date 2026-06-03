"""
Shared fixtures for the AI News Aggregator test suite.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.database.models import ArticleRecord, ContentType
from app.scrapers.models import Article


# ── Article factories ────────────────────────────────────────────────────────


def _make_record(
    title: str = "Test Article",
    source: str = "TestSource",
    guid: str | None = None,
    published_hours_ago: float = 1.0,
    description: str = "A test description.",
    url: str | None = None,
    content: str | None = None,
    content_type: ContentType = ContentType.ARTICLE,
) -> ArticleRecord:
    _guid = guid or f"guid-{title.lower().replace(' ', '-')}"
    published_at = datetime.now(timezone.utc) - timedelta(hours=published_hours_ago)
    return ArticleRecord(
        guid=_guid,
        title=title,
        description=description,
        url=url or f"https://example.com/{_guid}",
        source=source,
        published_at=published_at,
        content_type=content_type,
        content=content,
    )


def _make_article(
    title: str = "Test Article",
    source: str = "TestSource",
    guid: str | None = None,
    published_hours_ago: float = 1.0,
    description: str = "A test description.",
    url: str | None = None,
    content: str | None = None,
) -> Article:
    _guid = guid or f"guid-{title.lower().replace(' ', '-')}"
    published_at = datetime.now(timezone.utc) - timedelta(hours=published_hours_ago)
    return Article(
        guid=_guid,
        title=title,
        description=description,
        url=url or f"https://example.com/{_guid}",
        source=source,
        published_at=published_at,
        content=content,
    )


@pytest.fixture
def make_record():
    return _make_record


@pytest.fixture
def make_article():
    return _make_article


@pytest.fixture
def sample_records():
    """A realistic set of ArticleRecords from multiple sources."""
    return [
        _make_record("Claude 4 Released", source="Anthropic", published_hours_ago=2),
        _make_record("GPT-5 Capabilities", source="OpenAI", published_hours_ago=5),
        _make_record("Gemini Ultra Beats Benchmarks", source="Google DeepMind", published_hours_ago=10),
        _make_record("Claude 4 vs GPT-5 Comparison", source="TechCrunch", published_hours_ago=8),
        _make_record("LLM Agents in Production", source="Anthropic", published_hours_ago=20),
        _make_record("Open Source LLaMA 4 Released", source="Meta AI", published_hours_ago=36),
        _make_record("AI Coding Tools Market 2026", source="Cursor", published_hours_ago=48),
    ]


@pytest.fixture
def sample_digest():
    """A pre-built digest dict as DigestAgent.generate_digest() would return."""
    return {
        "subject": "🤖 AI News Digest — May 31, 2026",
        "sections": {
            "highlights": "## Claude 4\nWhy this matters: Revolutionary capabilities.",
            "cross_source": "Multiple sources cover the Claude vs GPT debate.",
            "action_items": "🔴 Try Claude 4 API today.\n🟡 Evaluate for production.",
            "quick_mentions": "- [Article A](https://a.com) — *Source A*",
            "trends": "**Trend 1**: Agentic AI — High confidence.",
        },
        "html": "<html><body><h1>Digest</h1></body></html>",
    }
