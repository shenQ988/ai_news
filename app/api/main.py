from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
    role: str = ""
    interests: List[str] = []
    digest_frequency: str = "daily"


class InterestsIn(BaseModel):
    interests: List[str]


def _profile_dict(p) -> dict:
    return {
        "id": p.id,
        "email": p.email,
        "name": p.name,
        "role": p.role,
        "interests": p.interests or [],
        "digest_frequency": p.digest_frequency,
        "is_active": p.is_active,
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


# ── App setup ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(config.database_url)
    Base.metadata.create_all(engine)
    app.state.engine = engine
    yield


app = FastAPI(title="AI News Aggregator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session() -> Session:
    return Session(app.state.engine)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("app/api/static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Profile ───────────────────────────────────────────────────────────────────


@app.post("/api/profile")
def save_profile(body: SubscribeIn):
    if body.interests and len(body.interests) > 10:
        raise HTTPException(400, "Maximum 10 interests allowed")
    with get_session() as session:
        repo = UserProfileRepository(session)
        profile = repo.save_profile(
            email=body.email,
            name=body.name,
            role=body.role,
            interests=body.interests,
            digest_frequency=body.digest_frequency,
        )
        session.commit()
        session.refresh(profile)
        return _profile_dict(profile)


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
            digest_frequency=body.digest_frequency,
        )
        session.commit()
        session.refresh(profile)
        return {"status": "subscribed", "interests_count": len(body.interests)}


@app.get("/api/profile/{email}")
def get_profile(email: str):
    with get_session() as session:
        profile = UserProfileRepository(session).get_profile(email)
        if not profile:
            raise HTTPException(404, "Profile not found")
        return _profile_dict(profile)


@app.patch("/api/profile/{email}/interests")
def update_interests(email: str, body: InterestsIn):
    if len(body.interests) < 2:
        raise HTTPException(400, "Please select at least 2 interests")
    if len(body.interests) > 10:
        raise HTTPException(400, "Maximum 10 interests allowed")
    with get_session() as session:
        repo = UserProfileRepository(session)
        profile = repo.update_interests(email, body.interests)
        if not profile:
            raise HTTPException(404, "Profile not found")
        session.commit()
        session.refresh(profile)
        return _profile_dict(profile)


@app.patch("/api/profile/{email}/active")
def set_active(email: str, is_active: bool):
    with get_session() as session:
        repo = UserProfileRepository(session)
        profile = repo.set_active(email, is_active)
        if not profile:
            raise HTTPException(404, "Profile not found")
        session.commit()
        session.refresh(profile)
        return _profile_dict(profile)


# ── Interests ─────────────────────────────────────────────────────────────────

_INTEREST_SUGGESTIONS = [
    {
        "category": "AI Research",
        "interests": [
            "Large language models",
            "AI agents and agentic workflows",
            "Multimodal AI",
            "AI safety and alignment",
            "Open source AI models",
            "Reinforcement learning",
        ],
    },
    {
        "category": "Engineering & Tools",
        "interests": [
            "LLM applications in production",
            "Developer tools and coding assistants",
            "MLOps and model deployment",
            "Vector databases and RAG",
            "AI infrastructure and hardware",
        ],
    },
    {
        "category": "Industry & Business",
        "interests": [
            "Startup trends and funding",
            "AI policy and regulation",
            "Enterprise AI adoption",
            "AI in healthcare",
            "AI in finance",
            "AI ethics and bias",
        ],
    },
    {
        "category": "Products & Platforms",
        "interests": [
            "OpenAI and ChatGPT",
            "Anthropic and Claude",
            "Google DeepMind",
            "Meta AI",
            "Robotics and embodied AI",
            "Autonomous vehicles",
        ],
    },
]


@app.get("/api/interests/suggestions")
def interest_suggestions():
    return _INTEREST_SUGGESTIONS


# ── Articles ──────────────────────────────────────────────────────────────────


def _article_dict(a) -> dict:
    return {
        "title": a.title,
        "description": a.description,
        "url": a.url,
        "source": a.source,
        "published_at": a.published_at.isoformat(),
        "category": a.category,
    }


@app.get("/api/articles/recent")
def recent_articles(hours: int = 24):
    with get_session() as session:
        articles = ArticleRepository(session).get_recent(hours=hours)
        return [_article_dict(a) for a in articles]


@app.get("/api/articles/search")
def search_articles(q: str, top_k: int = 20):
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")
    with get_session() as session:
        articles = ArticleRepository(session).search(q, top_k=top_k)
        return [_article_dict(a) for a in articles]


# ── Digest ────────────────────────────────────────────────────────────────────


@app.get("/api/digest/preview/{email}", response_class=HTMLResponse)
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
            return HTMLResponse(
                "<p style='font-family:sans-serif;padding:2rem'>No articles yet — run the scraper first.</p>"
            )

        digest = DigestAgent(user_profile=profile).generate_digest(result)
        return HTMLResponse(digest["html"])


# keep old query-param route for backward compat with CLI
@app.get("/api/digest/preview", response_class=HTMLResponse)
def preview_digest_query(email: str):
    return preview_digest(email)


@app.post("/api/digest/send/{email}")
def send_digest(email: str):
    from app.agent.smart_retrieval import SmartRetrieval
    from app.agent.digest_agent import DigestAgent
    from app.agent.fact_checker import FactChecker
    from app.services.email_service import EmailService

    with get_session() as session:
        profile = UserProfileRepository(session).get_profile(email)
        if not profile:
            raise HTTPException(404, "Profile not found")

        result = SmartRetrieval(ArticleRepository(session), user_profile=profile).retrieve()
        if not result["ranked_articles"]:
            raise HTTPException(422, "No articles available — run the scraper first")

        digest = DigestAgent(user_profile=profile).generate_digest(result)

    try:
        digest = FactChecker().check_and_revise(digest, result["ranked_articles"])
    except Exception:
        pass  # send unchecked rather than fail

    import os
    os.environ["EMAIL_TO"] = email
    sent = EmailService().send_digest(digest)
    return {"sent": sent, "email": email}
