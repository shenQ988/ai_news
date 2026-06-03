"""
Run once to initialize the database schema.
Usage: uv run python app/database/create_tables.py
"""
import logging

from sqlalchemy import text

from app.database.connection import engine
from app.database.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def create_tables() -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector extension ready")

    Base.metadata.create_all(engine)
    logger.info("Tables created: %s", list(Base.metadata.tables.keys()))


if __name__ == "__main__":
    create_tables()
    logger.info("Done")
