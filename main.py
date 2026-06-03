# main.py
"""
AI News Aggregator — Full Pipeline

Usage:
  python main.py                    # scrape + store (last 24h)
  python main.py 200                # scrape + store (last 200h)
  python main.py --digest           # scrape + store + generate digest (config user)
  python main.py --digest --send    # scrape + store + digest + email (config user)
  python main.py --digest-only      # skip scraping, generate digest from DB (config user)
  python main.py --digest-all       # generate + send personalized digest to all active users
  python main.py --digest --trace   # print fact-checker reasoning trace
  python main.py --digest --no-check # skip fact-checking (faster, for testing)
"""

import argparse
import logging
import os

from dotenv import load_dotenv


def setup_db():
    """Initialize database and tables."""
    from sqlalchemy import create_engine, text
    from app.database.models import Base

    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    return engine


def run_scrape(hours: int):
    """Scrape and store articles."""
    from app.runner import run_scrapers

    results = run_scrapers(hours=hours)

    print(f"\n=== Scraping Results (last {hours} hours) ===")
    total = 0
    for source, articles in results.items():
        print(f"  {source}: {len(articles)} articles")
        total += len(articles)
    print(f"  Total: {total} articles")

    return results


def run_digest(send_email: bool = False, user_profile=None, fact_check: bool = True, show_trace: bool = False, run_eval: bool = False, eval_metric: str = None):
    """Generate and optionally send the digest for one user."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.database.repositories import ArticleRepository
    from app.agent.smart_retrieval import SmartRetrieval
    from app.agent.digest_agent import DigestAgent
    from app.agent.email_agent import EmailAgent
    from app.agent.fact_checker import FactChecker

    engine = create_engine(os.getenv("DATABASE_URL"))

    with Session(engine) as session:
        repo = ArticleRepository(session)

        print("\n=== Smart Retrieval ===")
        retrieval = SmartRetrieval(repo, user_profile=user_profile)
        result = retrieval.retrieve()

        print(f"  Articles retrieved: {len(result['ranked_articles'])}")
        print(f"  Topic clusters: {len(result['topic_clusters'])}")
        print(f"  Cross-source topics: {len(result['cross_source_topics'])}")

        if not result["ranked_articles"]:
            print("  No articles to digest. Run scraping first.")
            return

        print("\n=== Generating Digest ===")
        agent = DigestAgent(user_profile=user_profile)
        digest = agent.generate_digest(result)
        print(f"  Subject: {digest['subject']}")

        if fact_check:
            print("\n=== Fact Checking ===")
            try:
                checker = FactChecker()
                digest = checker.check_and_revise(digest, result["ranked_articles"])
                if show_trace:
                    checker.trace.print_trace()
            except Exception as e:
                logging.warning("Fact checker failed, sending unchecked digest: %s", e)
        else:
            print("  (Skipped — use without --no-check to enable)")

        output_path = "digest_preview.html"
        with open(output_path, "w") as f:
            f.write(digest["html"])
        print(f"\n  Preview saved to: {output_path}")

        if run_eval:
            print("\n=== RAGAS Evaluation ===")
            try:
                from app.eval.ragas_evaluator import RAGASEvaluator
                evaluator = RAGASEvaluator()
                scores = evaluator.evaluate(
                    digest,
                    source_articles=result["ranked_articles"],
                    user_interests=user_profile.interests if user_profile else None,
                    metric=eval_metric,
                )
                evaluator.print_report(scores)
                evaluator.save_scores(scores, "eval_results.json")
            except Exception as e:
                logging.warning("RAGAS evaluation failed: %s", e)

        if send_email:
            print("\n=== Sending Email ===")
            email_agent = EmailAgent()
            success = email_agent.send(digest)
            print(f"  Email sent: {'yes' if success else 'FAILED'}")
        else:
            print("  (Use --send to email the digest)")


def run_digest_all():
    """Generate and send a personalized digest to every active user profile."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.database.repositories import ArticleRepository, UserProfileRepository
    from app.agent.smart_retrieval import SmartRetrieval
    from app.agent.digest_agent import DigestAgent
    from app.agent.email_agent import EmailAgent

    engine = create_engine(os.getenv("DATABASE_URL"))

    with Session(engine) as session:
        profiles = UserProfileRepository(session).get_active_profiles()

    if not profiles:
        print("No active user profiles found. Create one via the web UI.")
        return

    print(f"\n=== Digest All — {len(profiles)} active users ===")

    for profile in profiles:
        print(f"\n--- {profile.email} ({profile.name}) ---")
        with Session(engine) as session:
            repo = ArticleRepository(session)
            retrieval = SmartRetrieval(repo, user_profile=profile)
            result = retrieval.retrieve()

            if not result["ranked_articles"]:
                print("  No articles — skipping")
                continue

            agent = DigestAgent(user_profile=profile)
            digest = agent.generate_digest(result)

            # Override recipient to profile email
            import os as _os
            _os.environ["EMAIL_TO"] = profile.email

            email_agent = EmailAgent()
            success = email_agent.send(digest)
            print(f"  Sent: {'✓' if success else '✗ FAILED'}")


def main():
    parser = argparse.ArgumentParser(description="AI News Aggregator")
    parser.add_argument("hours", nargs="?", type=int, default=24,
                        help="Hours to look back for articles (default: 24)")
    parser.add_argument("--digest", action="store_true",
                        help="Generate a personalized digest after scraping")
    parser.add_argument("--digest-only", action="store_true",
                        help="Skip scraping, generate digest from existing DB")
    parser.add_argument("--digest-all", action="store_true",
                        help="Generate and send digest to all active users")
    parser.add_argument("--send", action="store_true",
                        help="Send the digest via email")
    parser.add_argument("--trace", action="store_true",
                        help="Print fact-checker reasoning trace after digest")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip fact-checking (faster, for testing)")
    parser.add_argument("--eval", action="store_true",
                        help="Run RAGAS evaluation after digest generation")
    parser.add_argument("--metric", choices=["faithfulness", "answer_relevancy", "context_precision"],
                        help="Run a single RAGAS metric (avoids rate limits)")

    args = parser.parse_args()

    setup_db()

    if args.digest_all:
        if not args.digest_only:
            run_scrape(args.hours)
        run_digest_all()
        return

    if not args.digest_only:
        run_scrape(args.hours)

    if args.digest or args.digest_only:
        run_digest(
            send_email=args.send,
            fact_check=not args.no_check,
            show_trace=args.trace,
            run_eval=args.eval,
            eval_metric=args.metric,
        )


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
