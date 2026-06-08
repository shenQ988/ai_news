"""
Seed a default subscriber using EMAIL_TO env var.
Safe to run multiple times — skips if the user already exists.
Run with: uv run python scripts/seed_test_user.py
"""
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMAIL_TO = os.getenv("EMAIL_TO")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set — skipping seed")
    sys.exit(0)

if not EMAIL_TO:
    print("ERROR: EMAIL_TO not set — skipping seed")
    sys.exit(0)

from app.database.models import Base, UserProfile

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

with Session(engine) as session:
    existing = session.query(UserProfile).filter_by(email=EMAIL_TO).first()
    if existing:
        print(f"User {existing.email} already exists — skipping")
    else:
        user = UserProfile(
            email=EMAIL_TO,
            name="Subscriber",
            role="Software engineer interested in building AI-powered products",
            interests=[
                "AI agents and agentic workflows",
                "startup trends and funding",
                "LLM applications in production",
                "developer tools and coding assistants",
                "open source AI models",
            ],
            digest_frequency="weekly",
            is_active=True,
        )
        session.add(user)
        session.commit()
        print(f"Created subscriber: {user.email}")
