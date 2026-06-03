import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.config import config
from app.database.models import ArticleRecord, UserProfile
from app.database.repositories import ArticleRepository

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "with", "on", "at", "by", "from", "as", "it", "its", "this", "that",
    "how", "what", "new", "ai", "has", "have", "will", "can", "been",
    "using", "use", "used", "into", "about", "more", "their",
}


class SmartRetrieval:
    def __init__(self, repo: ArticleRepository, user_profile: Optional[UserProfile] = None):
        self.repo = repo
        self._profile = user_profile

    @property
    def _interests(self) -> List[str]:
        if self._profile and self._profile.interests:
            return self._profile.interests
        return config.user_interests

    @property
    def _digest_size(self) -> int:
        return config.digest_size

    def retrieve(self) -> dict:
        seen: Dict[str, Tuple[ArticleRecord, int]] = {}

        for interest in self._interests:
            results = self.repo.search(interest, top_k=self._digest_size)
            for rank, article in enumerate(results):
                if article.guid not in seen or rank < seen[article.guid][1]:
                    seen[article.guid] = (article, rank)

        articles_with_rank = list(seen.values())
        scored = self._score(articles_with_rank)
        ranked_articles = [a for a, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

        clusters = self._cluster(ranked_articles)
        cross_source = [c for c in clusters if len({a.source for a in c}) >= 2]

        return {
            "ranked_articles": ranked_articles,
            "topic_clusters": clusters,
            "cross_source_topics": cross_source,
            "stats": {
                "total_retrieved": len(ranked_articles),
                "unique_sources": len({a.source for a in ranked_articles}),
                "queries_run": len(self._interests),
            },
        }

    def _score(self, articles_with_rank: List[Tuple[ArticleRecord, int]]) -> List[Tuple[ArticleRecord, float]]:
        now = datetime.now(timezone.utc)
        total = max(len(articles_with_rank), 1)

        source_counts: Dict[str, int] = defaultdict(int)
        for a, _ in articles_with_rank:
            source_counts[a.source] += 1
        max_count = max(source_counts.values(), default=1)

        scored = []
        for article, rank in articles_with_rank:
            sim = 1.0 - rank / total
            age_h = (now - article.published_at).total_seconds() / 3600
            recency = math.exp(-0.693 * age_h / 48)
            diversity = 1.0 - (source_counts[article.source] - 1) / max(max_count, 1)
            scored.append((article, 0.5 * sim + 0.3 * recency + 0.2 * diversity))

        return scored

    def _keywords(self, title: str) -> set:
        return {
            w.lower().strip("'\".,!?") for w in title.split()
            if len(w) > 3 and w.lower() not in STOP_WORDS
        }

    def _cluster(self, articles: List[ArticleRecord]) -> List[List[ArticleRecord]]:
        clusters: List[List[ArticleRecord]] = []
        assigned = set()

        for i, article in enumerate(articles):
            if i in assigned:
                continue
            cluster = [article]
            kw_i = self._keywords(article.title)
            for j, other in enumerate(articles[i + 1:], i + 1):
                if j in assigned:
                    continue
                if kw_i & self._keywords(other.title):
                    cluster.append(other)
                    assigned.add(j)
            clusters.append(cluster)
            assigned.add(i)

        return clusters
