from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.scrapers.feeds import FEEDS
from app.scrapers.models import Article
from app.scrapers.rss_scraper import RSSScraper


MOCK_ENTRY = {
    "title": "Test Article",
    "description": "A test description",
    "link": "https://openai.com/blog/test",
    "id": "https://openai.com/blog/test",
    "published_parsed": (2026, 5, 29, 12, 0, 0, 3, 149, 0),
    "tags": [{"term": "research"}],
}


def make_mock_feed(entries=None):
    feed = MagicMock()
    feed.entries = entries or [MagicMock(**MOCK_ENTRY, get=lambda k, d=None: MOCK_ENTRY.get(k, d))]
    return feed


class TestRSSScraperInit:
    def test_loads_all_feeds_by_default(self):
        scraper = RSSScraper()
        assert scraper.feeds == FEEDS

    def test_filters_to_specified_sources(self):
        scraper = RSSScraper(sources=["OpenAI"])
        assert list(scraper.feeds.keys()) == ["OpenAI"]

    def test_ignores_unknown_sources(self):
        scraper = RSSScraper(sources=["OpenAI", "NonExistent"])
        assert "NonExistent" not in scraper.feeds
        assert "OpenAI" in scraper.feeds

    def test_add_feed(self):
        scraper = RSSScraper(sources=[])
        scraper.add_feed("My Blog", "https://myblog.com/rss.xml")
        assert scraper.feeds["My Blog"] == "https://myblog.com/rss.xml"

    def test_remove_feed(self):
        scraper = RSSScraper(sources=["OpenAI"])
        scraper.remove_feed("OpenAI")
        assert "OpenAI" not in scraper.feeds

    def test_remove_nonexistent_feed_does_not_raise(self):
        scraper = RSSScraper(sources=[])
        scraper.remove_feed("NonExistent")


class TestScrapeArticles:
    @patch("app.scrapers.rss_scraper.requests.get")
    def test_returns_articles_within_hours(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<rss/>"
        mock_get.return_value = mock_resp

        entry = MagicMock()
        entry.get = lambda k, d=None: MOCK_ENTRY.get(k, d)
        entry.published_parsed = MOCK_ENTRY["published_parsed"]

        with patch("app.scrapers.rss_scraper.feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[entry])
            with patch.object(RSSScraper, "_fetch_article_content", return_value="content"):
                scraper = RSSScraper(sources=["OpenAI"])
                articles = scraper.get_articles(hours=200)

        assert len(articles) == 1
        assert articles[0].source == "OpenAI"
        assert articles[0].title == "Test Article"

    @patch("app.scrapers.rss_scraper.requests.get")
    def test_skips_entries_outside_window(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<rss/>"
        mock_get.return_value = mock_resp

        old_entry = MagicMock()
        old_entry.get = lambda k, d=None: MOCK_ENTRY.get(k, d)
        old_entry.published_parsed = (2020, 1, 1, 0, 0, 0, 0, 1, 0)

        with patch("app.scrapers.rss_scraper.feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[old_entry])
            scraper = RSSScraper(sources=["OpenAI"])
            articles = scraper.get_articles(hours=24)

        assert articles == []

    @patch("app.scrapers.rss_scraper.requests.get", side_effect=Exception("network error"))
    def test_failed_feed_returns_empty(self, mock_get):
        scraper = RSSScraper(sources=["OpenAI"])
        articles = scraper.get_articles(hours=24)
        assert articles == []

    @patch("app.scrapers.rss_scraper.requests.get")
    def test_articles_sorted_newest_first(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<rss/>"
        mock_get.return_value = mock_resp

        entry1 = MagicMock()
        entry1.get = lambda k, d=None: MOCK_ENTRY.get(k, d)
        entry1.published_parsed = (2026, 5, 28, 0, 0, 0, 0, 1, 0)

        entry2 = MagicMock()
        entry2.get = lambda k, d=None: MOCK_ENTRY.get(k, d)
        entry2.published_parsed = (2026, 5, 29, 0, 0, 0, 0, 1, 0)

        with patch("app.scrapers.rss_scraper.feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[entry1, entry2])
            with patch.object(RSSScraper, "_fetch_article_content", return_value=None):
                scraper = RSSScraper(sources=["OpenAI"])
                articles = scraper.get_articles(hours=200)

        assert articles[0].published_at > articles[1].published_at


class TestArticleModel:
    def test_article_fields(self):
        article = Article(
            title="Test",
            description="Desc",
            url="https://example.com",
            guid="https://example.com",
            published_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
            source="OpenAI",
        )
        assert article.category is None
        assert article.content is None
