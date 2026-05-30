import logging
from typing import Dict, List

from app.scrapers import RSSScraper, Article

logger = logging.getLogger(__name__)


def run_scrapers(hours: int = 24) -> Dict[str, List[Article]]:
    """Run all scrapers and return results grouped by source."""
    scraper = RSSScraper()  # uses all feeds from feeds.py
    articles = scraper.get_articles(hours=hours)

    # Group articles by source
    results: Dict[str, List[Article]] = {}
    for article in articles:
        if article.source not in results:
            results[article.source] = []
        results[article.source].append(article)

    return results