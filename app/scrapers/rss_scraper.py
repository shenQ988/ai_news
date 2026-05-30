from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import time

import feedparser
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from app.scrapers.models import Article
from app.scrapers.feeds import FEEDS

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class RSSScraper:
    """
    Generic RSS scraper that works with any feed source.

    Usage:
        # Scrape all configured feeds
        scraper = RSSScraper()
        articles = scraper.get_articles(hours=24)

        # Scrape specific sources only
        scraper = RSSScraper(sources=["OpenAI", "Anthropic News"])
        articles = scraper.get_articles(hours=24)

        # Add a custom feed on the fly
        scraper = RSSScraper()
        scraper.add_feed("My Blog", "https://myblog.com/rss.xml")
        articles = scraper.get_articles(hours=24)
    """

    def __init__(self, sources: Optional[List[str]] = None):
        """
        Initialize the scraper.

        Args:
            sources: List of source names to scrape (from feeds.py).
                     If None, scrapes all configured feeds.
        """
        if sources is None:
            self.feeds = dict(FEEDS)
        else:
            unknown = set(sources) - set(FEEDS.keys())
            if unknown:
                logger.warning(f"Unknown sources (skipped): {unknown}")
            self.feeds = {k: v for k, v in FEEDS.items() if k in sources}

        logger.info(f"RSSScraper initialized with {len(self.feeds)} feeds")

    def add_feed(self, name: str, url: str) -> None:
        """Add a custom feed at runtime."""
        self.feeds[name] = url
        logger.info(f"Added feed: {name} → {url}")

    def remove_feed(self, name: str) -> None:
        """Remove a feed by name."""
        if name in self.feeds:
            del self.feeds[name]
            logger.info(f"Removed feed: {name}")

    def _fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch article URL and convert to clean markdown."""
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove junk elements
            for tag in soup.find_all(
                ["nav", "footer", "header", "script", "style", "noscript"]
            ):
                tag.decompose()

            # Try to find the main article content
            article_el = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", {"role": "main"})
                or soup.find("div", class_=lambda c: c and "content" in c.lower())
                or soup.body
            )

            content = md(str(article_el), strip=["img"])

            # Clean up excessive whitespace
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _scrape_feed(self, source: str, rss_url: str, hours: int) -> List[Article]:
        """Scrape a single RSS feed and return articles."""
        logger.info(f"[{source}] Fetching RSS: {rss_url}")
        start = time.time()

        try:
            response = requests.get(rss_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
        except Exception as e:
            logger.error(f"[{source}] Failed to fetch RSS: {e}")
            return []

        logger.info(
            f"[{source}] RSS fetched in {time.time() - start:.1f}s "
            f"— {len(feed.entries)} entries"
        )

        if not feed.entries:
            return []

        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=hours)
        articles = []

        for entry in feed.entries:
            published_parsed = getattr(entry, "published_parsed", None)
            if not published_parsed:
                continue

            published_time = datetime(*published_parsed[:6], tzinfo=timezone.utc)
            if published_time < cutoff_time:
                continue

            article = Article(
                title=entry.get("title", ""),
                description=entry.get("description", ""),
                url=entry.get("link", ""),
                guid=entry.get("id", entry.get("link", "")),
                published_at=published_time,
                source=source,
                category=(
                    entry.get("tags", [{}])[0].get("term")
                    if entry.get("tags")
                    else None
                ),
            )

            logger.info(f"[{source}] Converting: {article.title}")
            start = time.time()
            article.content = self._fetch_article_content(article.url)
            chars = len(article.content) if article.content else 0
            logger.info(f"[{source}]   Done in {time.time() - start:.1f}s ({chars} chars)")

            articles.append(article)

        return articles

    def get_articles(self, hours: int = 24) -> List[Article]:
        """
        Scrape all configured feeds and return articles from the last N hours.

        Args:
            hours: Only return articles published within this many hours.

        Returns:
            List of Article objects sorted by published_at (newest first).
        """
        all_articles: List[Article] = []

        for source, rss_url in self.feeds.items():
            articles = self._scrape_feed(source, rss_url, hours)
            all_articles.extend(articles)
            logger.info(f"[{source}] {len(articles)} articles found")

        # Sort by newest first
        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        logger.info(
            f"Total: {len(all_articles)} articles from {len(self.feeds)} feeds"
        )
        return all_articles


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Example 1: Scrape all feeds
    # scraper = RSSScraper()

    # Example 2: Scrape only specific sources
    scraper = RSSScraper(sources=["OpenAI", "Anthropic News"])

    articles = scraper.get_articles(hours=200)

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(articles)} articles")
    print(f"{'='*70}")

    for a in articles:
        print(f"\n{a.published_at:%Y-%m-%d} | {a.source} | {a.category or 'uncategorized'}")
        print(f"  {a.title}")
        print(f"  Content: {'yes' if a.content else 'no'} ({len(a.content) if a.content else 0} chars)")