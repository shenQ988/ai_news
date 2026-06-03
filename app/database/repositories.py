import logging
import os
from typing import List, Optional

from huggingface_hub import InferenceClient
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.database.models import ArticleRecord, ContentType, UserProfile
from app.scrapers.models import Article

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ArticleRepository:
    def __init__(self, session: Session):
        self.session = session
        self.client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))

    def _embed(self, text: str) -> List[float]:
        result = self.client.feature_extraction(text[:8000], model=EMBEDDING_MODEL)
        return result.tolist() if hasattr(result, "tolist") else list(result)

    def _article_to_text(self, article: Article) -> str:
        parts = [article.title]
        if article.description:
            parts.append(article.description)
        if article.content:
            parts.append(article.content[:2000])
        return "\n\n".join(parts)

    def save(self, articles: List[Article], content_type: ContentType = ContentType.ARTICLE) -> int:
        """Embed and upsert articles. Returns number of new rows inserted."""
        if not articles:
            return 0

        rows = []
        for article in articles:
            try:
                embedding = self._embed(self._article_to_text(article))
            except Exception as e:
                logger.warning("Failed to embed '%s': %s", article.title, e)
                embedding = None

            rows.append({
                "guid": article.guid,
                "title": article.title,
                "description": article.description,
                "url": article.url,
                "source": article.source,
                "category": article.category,
                "content": article.content,
                "published_at": article.published_at,
                "content_type": content_type,
                "embedding": embedding,
            })

        stmt = (
            insert(ArticleRecord)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["guid"])
        )
        result = self.session.execute(stmt)
        inserted = result.rowcount
        logger.info("Saved %d new articles (%d duplicates skipped)", inserted, len(rows) - inserted)
        return inserted

    def search(self, query: str, top_k: int = 10, source: Optional[str] = None, content_type: Optional[ContentType] = None) -> List[ArticleRecord]:
        """Return top_k articles most similar to the query string."""
        query_embedding = self._embed(query)

        stmt = (
            select(ArticleRecord)
            .order_by(ArticleRecord.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        if source:
            stmt = stmt.where(ArticleRecord.source == source)
        if content_type:
            stmt = stmt.where(ArticleRecord.content_type == content_type)

        return list(self.session.scalars(stmt).all())

    def get_articles_without_content(self, limit: Optional[int] = None) -> List[ArticleRecord]:
        stmt = (
            select(ArticleRecord)
            .where(ArticleRecord.content.is_(None))
            .order_by(ArticleRecord.published_at.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt).all())

    def update_article_content(self, guid: str, content: str) -> None:
        record = self.session.scalars(select(ArticleRecord).where(ArticleRecord.guid == guid)).first()
        if record:
            record.content = content
            record.embedding = self._embed(f"{record.title}\n\n{content[:2000]}")

    def get_all(self, source: Optional[str] = None, content_type: Optional[ContentType] = None) -> List[ArticleRecord]:
        stmt = select(ArticleRecord).order_by(ArticleRecord.published_at.desc())
        if source:
            stmt = stmt.where(ArticleRecord.source == source)
        if content_type:
            stmt = stmt.where(ArticleRecord.content_type == content_type)
        return list(self.session.scalars(stmt).all())

    def get_recent(self, hours: int = 24) -> List[ArticleRecord]:
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(ArticleRecord)
            .where(ArticleRecord.published_at >= cutoff)
            .order_by(ArticleRecord.published_at.desc())
        )
        return list(self.session.scalars(stmt).all())


class UserProfileRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_profile(self, email: str, name: str = "", role: str = "",
                     interests: List[str] = None, digest_frequency: str = "daily") -> UserProfile:
        """Create or update a user profile."""
        existing = self.get_profile(email)
        if existing:
            existing.name = name
            existing.role = role
            existing.interests = interests or []
            existing.digest_frequency = digest_frequency
            self.session.flush()
            return existing

        profile = UserProfile(
            email=email,
            name=name,
            role=role,
            interests=interests or [],
            digest_frequency=digest_frequency,
        )
        self.session.add(profile)
        self.session.flush()
        return profile

    def get_profile(self, email: str) -> Optional[UserProfile]:
        return self.session.scalars(
            select(UserProfile).where(UserProfile.email == email)
        ).first()

    def get_active_profiles(self) -> List[UserProfile]:
        return list(self.session.scalars(
            select(UserProfile).where(UserProfile.is_active == True)
        ).all())

    def update_interests(self, email: str, interests: List[str]) -> Optional[UserProfile]:
        profile = self.get_profile(email)
        if profile:
            profile.interests = interests
            self.session.flush()
        return profile

    def set_active(self, email: str, is_active: bool) -> Optional[UserProfile]:
        profile = self.get_profile(email)
        if profile:
            profile.is_active = is_active
            self.session.flush()
        return profile
