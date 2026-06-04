# AI News Digest

A self-hosted weekly newsletter that scrapes AI news, ranks it by your interests, fact-checks the summary with an LLM, and emails it to subscribers every Monday — fully automated via GitHub Actions.

---

## How it works

```
GitHub Actions (every Monday)
  → Scrape 16 AI RSS feeds
  → Embed articles into Supabase (pgvector)
  → SmartRetrieval: rank by interest similarity + recency
  → DigestAgent: generate personalized digest via LLM
  → FactChecker: verify claims against source articles
  → EmailService: send to all active subscribers
```

## Subscribe

> **Live at:** `https://your-railway-or-render-url.com`

Visit the link above, enter your email, pick your interests, and you'll receive a digest every Monday.

---

## Stack

| Layer | Service |
|---|---|
| Automation | GitHub Actions (free) |
| Database | Supabase PostgreSQL + pgvector (free) |
| Backend | FastAPI on Railway / Render (free tier) |
| LLM | Qwen 2.5 72B via HuggingFace Inference API |
| Email | Gmail SMTP |

---

## Local setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), a Supabase project, a HuggingFace API key, Gmail App Password.

```bash
git clone https://github.com/YOUR_USERNAME/ai_news.git
cd ai_news
uv sync
cp .env.example .env   # fill in your credentials
```

Create tables:
```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from sqlalchemy import create_engine, text
from app.database.models import Base
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()
Base.metadata.create_all(engine)
"
```

Run the pipeline manually:
```bash
uv run python main.py --digest --send   # scrape + digest + email
uv run python main.py --use-graph       # same via LangGraph orchestration
```

Start the web UI locally:
```bash
uv run uvicorn app.api.main:app --reload
# open http://localhost:8000
```

---

## Deploy the web UI (Railway or Render)

So subscribers can sign up via the landing page at a public URL.

**Railway:**
1. [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
2. Add environment variables (same as GitHub secrets)
3. Railway auto-detects the `Procfile` and deploys

**Render:**
1. [render.com](https://render.com) → New Web Service → connect repo
2. Render reads `render.yaml` automatically
3. Add environment variables in the Render dashboard

Update the **Subscribe** link at the top of this README with your deployed URL.

---

## GitHub Actions secrets

Add these in: **Repo → Settings → Secrets and variables → Actions**

| Secret | Description |
|---|---|
| `DATABASE_URL` | Supabase connection string (Transaction pooler, port 6543) |
| `HUGGINGFACE_API_KEY` | From huggingface.co/settings/tokens |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASSWORD` | Gmail App Password (16 chars) |
| `EMAIL_FROM` | Your Gmail address |

Trigger a test run anytime: **Actions → Weekly AI News Digest → Run workflow**

---

## Project structure

```
app/
├── scrapers/        # RSS feed scraping (20+ AI sources)
├── agent/           # SmartRetrieval, DigestAgent, FactChecker
├── services/        # EmailService
├── database/        # SQLAlchemy models + pgvector repositories
├── orchestration/   # LangGraph state machine
└── api/             # FastAPI backend + landing page UI
.github/
└── workflows/
    └── weekly_digest.yml   # Monday 07:00 UTC cron
```
