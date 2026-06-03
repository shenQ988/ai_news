from datetime import datetime
from enum import Enum
from typing import List

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Enum as SAEnum, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ContentType(str, Enum):
    ARTICLE = "article"
    VIDEO = "video"


class Base(DeclarativeBase):
    pass


class ArticleRecord(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    guid: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_type: Mapped[ContentType] = mapped_column(SAEnum(ContentType), nullable=False, default=ContentType.ARTICLE)
    embedding: Mapped[list | None] = mapped_column(Vector(384), nullable=True)

    __table_args__ = (UniqueConstraint("guid", name="uq_articles_guid"),)

    def __repr__(self) -> str:
        return f"<ArticleRecord {self.source} | {self.title[:50]}>"


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    role: Mapped[str] = mapped_column(String(256), default="")
    interests: Mapped[List[str]] = mapped_column(JSON, default=list)
    digest_frequency: Mapped[str] = mapped_column(String(16), default="daily")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<UserProfile {self.email}>"
