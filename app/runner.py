import logging
from typing import Dict, List

from app.database.connection import get_session
from app.database.models import ContentType
from app.database.repositories import ArticleRepository
from app.scrapers import RSSScraper, Article

logger = logging.getLogger(__name__)


def run_scrapers(hours: int = 24) -> Dict[str, List[Article]]:
    """Scrape all feeds, embed and persist articles, return results grouped by source."""
    scraper = RSSScraper()
    articles = scraper.get_articles(hours=hours)

    with get_session() as session:
        repo = ArticleRepository(session)
        inserted = repo.save(articles, content_type=ContentType.ARTICLE)
        logger.info("Persisted %d new articles to database", inserted)

    # Group by source for the return value
    results: Dict[str, List[Article]] = {}
    for article in articles:
        results.setdefault(article.source, []).append(article)

    return results