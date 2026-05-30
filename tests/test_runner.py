from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.scrapers.models import Article
from app.runner import run_scrapers


def make_article(source: str, title: str = "Test") -> Article:
    return Article(
        title=title,
        description="desc",
        url="https://example.com",
        guid="https://example.com",
        published_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
        source=source,
    )


MOCK_ARTICLES = [
    make_article("OpenAI", "GPT-5 released"),
    make_article("OpenAI", "New safety research"),
    make_article("Anthropic Research", "Claude update"),
]


@patch("app.runner.RSSScraper")
class TestRunScrapers:
    def test_returns_articles_grouped_by_source(self, mock_scraper_cls):
        mock_scraper_cls.return_value.get_articles.return_value = MOCK_ARTICLES

        results = run_scrapers(hours=24)

        assert "OpenAI" in results
        assert "Anthropic Research" in results
        assert len(results["OpenAI"]) == 2
        assert len(results["Anthropic Research"]) == 1

    def test_passes_hours_to_scraper(self, mock_scraper_cls):
        mock_scraper_cls.return_value.get_articles.return_value = MOCK_ARTICLES

        run_scrapers(hours=48)

        mock_scraper_cls.return_value.get_articles.assert_called_once_with(hours=48)

    def test_returns_empty_dict_when_no_articles(self, mock_scraper_cls):
        mock_scraper_cls.return_value.get_articles.return_value = []

        results = run_scrapers(hours=24)

        assert results == {}

    def test_articles_in_results_are_article_instances(self, mock_scraper_cls):
        mock_scraper_cls.return_value.get_articles.return_value = MOCK_ARTICLES

        results = run_scrapers(hours=24)

        for articles in results.values():
            for article in articles:
                assert isinstance(article, Article)


@patch("app.runner.RSSScraper")
class TestMain:
    def test_main_returns_results(self, mock_scraper_cls):
        from main import main
        mock_scraper_cls.return_value.get_articles.return_value = MOCK_ARTICLES

        results = main(hours=24)

        assert isinstance(results, dict)
        assert "OpenAI" in results

    def test_main_default_hours(self, mock_scraper_cls):
        from main import main
        mock_scraper_cls.return_value.get_articles.return_value = []

        main()

        mock_scraper_cls.return_value.get_articles.assert_called_once_with(hours=24)
