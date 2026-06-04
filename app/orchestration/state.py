from typing import TypedDict, Optional, List, Dict
from app.scrapers.models import Article


class PipelineState(TypedDict):
    """State that flows through the LangGraph nodes."""
    # Inputs
    hours: int
    user_profile: dict

    # Scraping outputs
    scraped_articles: List[Article]
    scrape_stats: dict

    # Retrieval outputs
    retrieved_articles: List[dict]
    topic_clusters: dict
    cross_source_topics: list

    # Digest outputs
    digest: Optional[dict]

    # Fact check outputs
    checked_digest: Optional[dict]
    hallucination_rate: float
    revision_count: int

    # Email outputs
    email_sent: bool

    # Control
    should_send_email: bool
    error: Optional[str]
