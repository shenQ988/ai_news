from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import config
from app.database.models import Base
from app.database.repositories import ArticleRepository, UserProfileRepository

# ── Schemas ───────────────────────────────────────────────────────────────────


class SubscribeIn(BaseModel):
    email: EmailStr
    name: str = ""
    interests: List[str] = []
    frequency: str = "daily"


# ── App setup ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(config.database_url)
    Base.metadata.create_all(engine)
    app.state.engine = engine
    yield


app = FastAPI(title="AI News Aggregator", lifespan=lifespan)
templates = Jinja2Templates(directory="app/api/templates")


def get_session() -> Session:
    return Session(app.state.engine)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/subscribe")
def subscribe(body: SubscribeIn):
    if len(body.interests) < 2:
        raise HTTPException(400, "Please select at least 2 interests")
    if len(body.interests) > 10:
        raise HTTPException(400, "Maximum 10 interests allowed")

    with get_session() as session:
        repo = UserProfileRepository(session)
        profile = repo.save_profile(
            email=body.email,
            name=body.name,
            interests=body.interests,
            digest_frequency=body.frequency,
        )
        session.commit()

    return {"status": "subscribed", "interests_count": len(body.interests)}


@app.get("/api/digest/preview", response_class=HTMLResponse)
def preview_digest(email: str):
    from app.agent.smart_retrieval import SmartRetrieval
    from app.agent.digest_agent import DigestAgent

    with get_session() as session:
        profile_repo = UserProfileRepository(session)
        profile = profile_repo.get_profile(email)
        if not profile:
            raise HTTPException(404, "Profile not found")

        article_repo = ArticleRepository(session)
        result = SmartRetrieval(article_repo, user_profile=profile).retrieve()

        if not result["ranked_articles"]:
            return HTMLResponse("<p style='font-family:sans-serif;padding:2rem'>No articles yet — run the scraper first.</p>")

        digest = DigestAgent(user_profile=profile).generate_digest(result)
        return HTMLResponse(digest["html"])
