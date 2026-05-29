from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import time

import feedparser
import requests
from markdownify import markdownify as md
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenAIArticle(BaseModel):
    title: str
    description: str
    url: str
    guid: str
    published_at: datetime
    category: Optional[str] = None
    content: Optional[str] = None


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


class OpenAIScraper:
    def __init__(self):
        self.rss_url = "https://openai.com/news/rss.xml"

    def _fetch_article_content(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove junk elements
            for tag in soup.find_all(["nav", "footer", "header", "script", "style", "noscript"]):
                tag.decompose()

            # Try to find the main article content
            article = (
                soup.find("article") or
                soup.find("main") or
                soup.find("div", {"role": "main"}) or
                soup.find("div", class_=lambda c: c and "content" in c.lower()) or
                soup.body
            )

            content = md(str(article), strip=["img"])

            # Clean up whitespace
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_articles(self, hours: int = 24) -> List[OpenAIArticle]:
        logger.info(f"Fetching RSS feed from {self.rss_url}")
        start = time.time()

        # Fetch with browser user-agent to avoid being blocked
        try:
            response = requests.get(self.rss_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed: {e}")
            return []

        logger.info(f"RSS fetched in {time.time() - start:.1f}s — {len(feed.entries)} entries")

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
            if published_time >= cutoff_time:
                article = OpenAIArticle(
                    title=entry.get("title", ""),
                    description=entry.get("description", ""),
                    url=entry.get("link", ""),
                    guid=entry.get("id", entry.get("link", "")),
                    published_at=published_time,
                    category=entry.get("tags", [{}])[0].get("term") if entry.get("tags") else None,
                )

                logger.info(f"Converting: {article.title}")
                start = time.time()
                article.content = self._fetch_article_content(article.url)
                chars = len(article.content) if article.content else 0
                logger.info(f"  Done in {time.time() - start:.1f}s ({chars} chars)")

                articles.append(article)

        logger.info(f"Done! {len(articles)} articles processed")
        return articles


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    scraper = OpenAIScraper()
    articles = scraper.get_articles(hours=200)
    for a in articles:
        print(f"\n{'='*60}")
        print(f"{a.published_at:%Y-%m-%d} | {a.category or 'uncategorized'} | {a.title}")
        print(f"Content: {'yes' if a.content else 'no'} ({len(a.content) if a.content else 0} chars)")
        if a.content:
            print(f"Preview: {a.content[:200]}...")