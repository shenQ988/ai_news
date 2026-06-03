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


class TestMain:
    def test_main_scrapes_with_default_hours(self):
        from main import main
        with patch("main.setup_db"), \
             patch("main.run_scrape") as mock_scrape, \
             patch("sys.argv", ["main.py"]):
            main()
        mock_scrape.assert_called_once_with(24)

    def test_main_scrapes_with_custom_hours(self):
        from main import main
        with patch("main.setup_db"), \
             patch("main.run_scrape") as mock_scrape, \
             patch("sys.argv", ["main.py", "200"]):
            main()
        mock_scrape.assert_called_once_with(200)

    def test_main_digest_flag_calls_run_digest(self):
        from main import main
        with patch("main.setup_db"), \
             patch("main.run_scrape"), \
             patch("main.run_digest") as mock_digest, \
             patch("sys.argv", ["main.py", "--digest"]):
            main()
        mock_digest.assert_called_once_with(send_email=False)

    def test_main_digest_only_skips_scraping(self):
        from main import main
        with patch("main.setup_db"), \
             patch("main.run_scrape") as mock_scrape, \
             patch("main.run_digest"), \
             patch("sys.argv", ["main.py", "--digest-only"]):
            main()
        mock_scrape.assert_not_called()

    def test_main_send_flag_passed_to_run_digest(self):
        from main import main
        with patch("main.setup_db"), \
             patch("main.run_scrape"), \
             patch("main.run_digest") as mock_digest, \
             patch("sys.argv", ["main.py", "--digest", "--send"]):
            main()
        mock_digest.assert_called_once_with(send_email=True)
