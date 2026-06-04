"""
LangGraph node functions. Each wraps an existing service class unchanged.
Nodes return updated state — no logic lives here, only orchestration.
"""
import logging
from sqlalchemy.orm import Session

from app.config import config
from app.scrapers.rss_scraper import RSSScraper
from app.database.repositories import ArticleRepository
from app.agent.smart_retrieval import SmartRetrieval
from app.agent.digest_agent import DigestAgent
from app.agent.fact_checker import FactChecker
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


def scrape_node(state: dict) -> dict:
    """Scrape articles from RSS feeds and store in DB."""
    logger.info("=== Node: scrape ===")
    try:
        scraper = RSSScraper()
        articles = scraper.get_articles(hours=state["hours"])

        from sqlalchemy import create_engine
        engine = create_engine(config.database_url)
        with Session(engine) as session:
            repo = ArticleRepository(session)
            inserted = repo.save(articles)
            session.commit()

        return {
            **state,
            "scraped_articles": articles,
            "scrape_stats": {
                "total": len(articles),
                "inserted": inserted,
            },
        }
    except Exception as e:
        logger.error("Scrape failed: %s", e)
        return {**state, "error": f"scrape: {e}"}


def retrieve_node(state: dict) -> dict:
    """Retrieve and rank articles using SmartRetrieval."""
    logger.info("=== Node: retrieve ===")
    try:
        from sqlalchemy import create_engine
        engine = create_engine(config.database_url)
        with Session(engine) as session:
            repo = ArticleRepository(session)
            retrieval = SmartRetrieval(repo)
            result = retrieval.retrieve()

        return {
            **state,
            "retrieved_articles": result["ranked_articles"],
            "topic_clusters": result["topic_clusters"],
            "cross_source_topics": result["cross_source_topics"],
        }
    except Exception as e:
        logger.error("Retrieve failed: %s", e)
        return {**state, "error": f"retrieve: {e}"}


def digest_node(state: dict) -> dict:
    """Generate digest from retrieved articles."""
    logger.info("=== Node: digest ===")
    try:
        agent = DigestAgent()
        articles = state["retrieved_articles"]
        retrieval_result = {
            "ranked_articles": articles,
            "topic_clusters": state["topic_clusters"],
            "cross_source_topics": state["cross_source_topics"],
            "stats": {
                "total_retrieved": len(articles),
                "unique_sources": len({a.source for a in articles}),
            },
        }
        digest = agent.generate_digest(retrieval_result)

        return {
            **state,
            "digest": digest,
        }
    except Exception as e:
        logger.error("Digest failed: %s", e)
        return {**state, "error": f"digest: {e}"}


def fact_check_node(state: dict) -> dict:
    """Run fact checker on the digest."""
    logger.info("=== Node: fact_check ===")

    digest = state.get("digest")
    if not digest:
        logger.warning("fact_check skipped — no digest in state (prior node may have failed)")
        return {**state, "checked_digest": None, "hallucination_rate": 0.0}

    try:
        checker = FactChecker()
        # retrieved_articles contains ArticleRecord objects from SmartRetrieval
        source_articles = state["retrieved_articles"]

        checked = checker.check_and_revise(digest, source_articles)

        report = checked.get("check_report", {})
        hallucination_rate = report.get("post_check_hallucination_rate", 0.0)

        return {
            **state,
            "checked_digest": checked,
            "hallucination_rate": hallucination_rate,
            "revision_count": state.get("revision_count", 0) + 1,
        }
    except Exception as e:
        logger.error("Fact check failed: %s", e)
        # Use original digest so save_preview / email can still run
        return {
            **state,
            "checked_digest": digest,
            "hallucination_rate": 0.0,
            "error": f"fact_check: {e}",
        }


def email_node(state: dict) -> dict:
    """Send the email."""
    logger.info("=== Node: email ===")

    if not state.get("should_send_email"):
        logger.info("Skipping email (should_send_email=False)")
        return {**state, "email_sent": False}

    try:
        email_service = EmailService()
        success = email_service.send_digest(state["checked_digest"])
        return {**state, "email_sent": success}
    except Exception as e:
        logger.error("Email failed: %s", e)
        return {**state, "email_sent": False, "error": f"email: {e}"}


def save_preview_node(state: dict) -> dict:
    """Save digest preview to HTML file."""
    logger.info("=== Node: save_preview ===")
    digest = state.get("checked_digest")
    if not digest:
        logger.warning("save_preview skipped — no digest available")
        return state
    with open("digest_preview.html", "w") as f:
        f.write(digest.get("html", ""))
    logger.info("Preview saved to digest_preview.html")
    return state
