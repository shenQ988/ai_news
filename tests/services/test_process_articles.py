"""
Tests for process_missing_content — retry logic, counting, failure handling.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.services.process_articles import process_missing_content


def make_mock_repo(articles_without_content=None):
    repo = MagicMock()
    repo.get_articles_without_content.return_value = articles_without_content or []
    return repo


def make_mock_scraper(content="Fetched content."):
    scraper = MagicMock()
    scraper._fetch_article_content.return_value = content
    return scraper


@patch("app.services.process_articles.RSSScraper")
@patch("app.services.process_articles.get_session")
class TestProcessMissingContent:
    def test_returns_correct_structure(self, mock_session, mock_scraper_cls, make_record):
        mock_session.return_value.__enter__.return_value = MagicMock()
        mock_scraper_cls.return_value = make_mock_scraper()

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            mock_repo_cls.return_value = make_mock_repo([])
            result = process_missing_content()

        assert "total" in result
        assert "processed" in result
        assert "failed" in result

    def test_returns_zeros_when_no_missing_articles(self, mock_session, mock_scraper_cls):
        mock_session.return_value.__enter__.return_value = MagicMock()
        mock_scraper_cls.return_value = make_mock_scraper()

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            mock_repo_cls.return_value = make_mock_repo([])
            result = process_missing_content()

        assert result == {"total": 0, "processed": 0, "failed": 0}

    def test_counts_successfully_processed_articles(self, mock_session, mock_scraper_cls, make_record):
        articles = [make_record(f"Article {i}", content=None) for i in range(3)]
        mock_session.return_value.__enter__.return_value = MagicMock()
        mock_scraper_cls.return_value = make_mock_scraper("Some content")

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            mock_repo_cls.return_value = make_mock_repo(articles)
            result = process_missing_content()

        assert result["total"] == 3
        assert result["processed"] == 3
        assert result["failed"] == 0

    def test_counts_failed_when_content_fetch_returns_none(self, mock_session, mock_scraper_cls, make_record):
        articles = [make_record("Article", content=None)]
        mock_session.return_value.__enter__.return_value = MagicMock()
        mock_scraper_cls.return_value = make_mock_scraper(content=None)

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            mock_repo_cls.return_value = make_mock_repo(articles)
            result = process_missing_content()

        assert result["failed"] == 1
        assert result["processed"] == 0

    def test_counts_failed_when_fetch_raises_exception(self, mock_session, mock_scraper_cls, make_record):
        articles = [make_record("Article", content=None)]
        mock_session.return_value.__enter__.return_value = MagicMock()
        scraper = MagicMock()
        scraper._fetch_article_content.side_effect = Exception("Network error")
        mock_scraper_cls.return_value = scraper

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            mock_repo_cls.return_value = make_mock_repo(articles)
            result = process_missing_content()

        assert result["failed"] == 1
        assert result["processed"] == 0

    def test_updates_content_for_each_processed_article(self, mock_session, mock_scraper_cls, make_record):
        articles = [make_record("A", content=None), make_record("B", content=None)]
        mock_session.return_value.__enter__.return_value = MagicMock()
        mock_scraper_cls.return_value = make_mock_scraper("content")

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            repo = make_mock_repo(articles)
            mock_repo_cls.return_value = repo
            process_missing_content()

        assert repo.update_article_content.call_count == 2

    def test_respects_limit_parameter(self, mock_session, mock_scraper_cls, make_record):
        mock_session.return_value.__enter__.return_value = MagicMock()
        mock_scraper_cls.return_value = make_mock_scraper()

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            repo = make_mock_repo([])
            mock_repo_cls.return_value = repo
            process_missing_content(limit=5)

        repo.get_articles_without_content.assert_called_once_with(limit=5)

    def test_continues_after_single_failure(self, mock_session, mock_scraper_cls, make_record):
        articles = [make_record("Good"), make_record("Bad"), make_record("Good2")]
        mock_session.return_value.__enter__.return_value = MagicMock()

        call_count = 0

        def fetch_side_effect(url):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Fetch failed")
            return "content"

        scraper = MagicMock()
        scraper._fetch_article_content.side_effect = fetch_side_effect
        mock_scraper_cls.return_value = scraper

        with patch("app.services.process_articles.ArticleRepository") as mock_repo_cls:
            mock_repo_cls.return_value = make_mock_repo(articles)
            result = process_missing_content()

        assert result["processed"] == 2
        assert result["failed"] == 1
        assert result["total"] == 3
