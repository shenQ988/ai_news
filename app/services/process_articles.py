import logging
from typing import Optional

from app.database.connection import get_session
from app.database.repositories import ArticleRepository
from app.scrapers.rss_scraper import RSSScraper

logger = logging.getLogger(__name__)


def process_missing_content(limit: Optional[int] = None) -> dict:
    """Retry fetching content for articles that failed during initial scraping."""
    scraper = RSSScraper()

    with get_session() as session:
        repo = ArticleRepository(session)
        articles = repo.get_articles_without_content(limit=limit)

        processed = 0
        failed = 0

        for article in articles:
            try:
                content = scraper._fetch_article_content(article.url)
                if content:
                    repo.update_article_content(article.guid, content)
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Failed to process '%s': %s", article.title, e)
                failed += 1

    result = {
        "total": len(articles),
        "processed": processed,
        "failed": failed,
    }
    logger.info("Processing results: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    process_missing_content()
