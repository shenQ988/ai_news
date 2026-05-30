"""
Integration tests — these make real network requests.
Run with: uv run pytest tests/scrapers/test_rss_scraper_integration.py -v
"""
import pytest

from app.scrapers.rss_scraper import RSSScraper
from app.scrapers.models import Article


@pytest.fixture(scope="module")
def anthropic_articles():
    scraper = RSSScraper(sources=["Anthropic Research"])
    return scraper.get_articles(hours=24 * 90)


class TestAnthropicResearchFeed:
    def test_returns_articles(self, anthropic_articles):
        assert len(anthropic_articles) > 0

    def test_articles_are_article_instances(self, anthropic_articles):
        for article in anthropic_articles:
            assert isinstance(article, Article)

    def test_required_fields_populated(self, anthropic_articles):
        for article in anthropic_articles:
            assert article.title
            assert article.url.startswith("https://")
            assert article.guid
            assert article.source == "Anthropic Research"
            assert article.published_at is not None

    def test_articles_sorted_newest_first(self, anthropic_articles):
        dates = [a.published_at for a in anthropic_articles]
        assert dates == sorted(dates, reverse=True)

    def test_content_fetched(self, anthropic_articles):
        articles_with_content = [a for a in anthropic_articles if a.content]
        assert len(articles_with_content) > 0

    def test_content_is_non_empty_string(self, anthropic_articles):
        for article in anthropic_articles:
            if article.content:
                assert len(article.content) > 100


class TestAddCustomFeed:
    def test_add_and_scrape_custom_feed(self):
        scraper = RSSScraper(sources=[])
        scraper.add_feed(
            "Anthropic Research",
            "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml",
        )
        articles = scraper.get_articles(hours=24 * 90)
        assert len(articles) > 0
        assert all(a.source == "Anthropic Research" for a in articles)


class TestMultipleSources:
    def test_scrape_multiple_sources(self):
        scraper = RSSScraper(sources=["Anthropic News", "Anthropic Research"])
        articles = scraper.get_articles(hours=24 * 90)
        sources = {a.source for a in articles}
        assert "Anthropic Research" in sources

    def test_articles_from_all_sources_present(self):
        scraper = RSSScraper(sources=["Anthropic News", "Anthropic Research"])
        articles = scraper.get_articles(hours=24 * 90)
        sources = {a.source for a in articles}
        assert sources == {"Anthropic News", "Anthropic Research"}
