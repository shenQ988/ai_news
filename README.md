# AI News Digest

A fully automated, personalized AI newsletter — built end-to-end with a RAG pipeline, agentic fact-checking, and LangGraph orchestration. Scrapes 16 AI news sources every week, ranks articles by each subscriber's interests, generates a digest with an LLM, and delivers it by email. No manual steps.

**Live:** coming soon &nbsp;·&nbsp; **Stack:** Python · FastAPI · PostgreSQL · pgvector · LangGraph · GitHub Actions

---

## What it does

<img width="500" alt="Landing page" src="https://github.com/user-attachments/assets/bc200dae-5f62-4400-ac45-262b55c3457f" />

Subscribers pick their interests on the landing page. Every Monday, the pipeline runs automatically:

```
Scrape 16 RSS feeds (OpenAI, Anthropic, DeepMind, Meta AI, Mistral…)
  → Embed articles with sentence-transformers → store in Supabase pgvector
  → SmartRetrieval: semantic search + recency + source diversity scoring
  → DigestAgent: generate personalized digest via Qwen 2.5 72B (HuggingFace)
  → FactChecker: ReAct agent verifies claims against source articles
  → Email delivered to all active subscribers via Gmail SMTP
```

---

## What's in each digest

**Highlights** — the week's most important AI developments, ranked and summarized based on your specific interests.
<img width="500" height=auto alt="image" src="https://github.com/user-attachments/assets/5a8cfc8a-2370-42da-9dd9-56c14edce1f9" />

**Cross-source analysis** — stories that appeared across multiple independent sources are surfaced and synthesized, separating signal from noise.
<img width="500" height=auto alt="image" src="https://github.com/user-attachments/assets/59d76825-b71b-4fe7-b9f2-4c240590747e" />


**Action items** — concrete takeaways: tools to try, papers to read, trends to watch. Turns news into next steps.
<img width="500" height=auto alt="image" src="https://github.com/user-attachments/assets/e0174059-f09b-499a-af56-54632143a732" />

---

## Technical highlights

**RAG pipeline** — articles are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in Supabase with pgvector. Retrieval uses cosine similarity weighted with recency and source diversity to avoid echo chambers.

**Agentic fact-checking** — a ReAct agent extracts claims from the generated digest, verifies each against the source articles, and rewrites unsupported sections. Reduces hallucination rate before delivery.

**LangGraph orchestration** — the pipeline is a compiled state machine with conditional retry loops: if the fact-checker flags too many unsupported claims, the digest is regenerated before sending.

**Fully serverless** — GitHub Actions runs the weekly cron. Supabase hosts the database. No always-on server required for the core pipeline.

---

## Stack

| | |
|---|---|
| **Language** | Python 3.11 |
| **Backend** | FastAPI |
| **Database** | Supabase (PostgreSQL + pgvector) |
| **Orchestration** | LangGraph |
| **LLM** | Qwen 2.5 72B via HuggingFace Inference API |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Automation** | GitHub Actions (weekly cron) |
| **Email** | Gmail SMTP |

---

## Project structure

```
app/
├── scrapers/        # RSS feed scraping + content extraction
├── agent/           # SmartRetrieval, DigestAgent, FactChecker (ReAct)
├── services/        # EmailService
├── database/        # SQLAlchemy ORM + pgvector repository
├── orchestration/   # LangGraph state machine + nodes
└── api/             # FastAPI backend + subscription landing page
.github/
└── workflows/
    └── weekly_digest.yml   # Monday 07:00 UTC
```

---

## Local setup

```bash
git clone https://github.com/shenQ988/ai_news.git
cd ai_news
uv sync
cp .env.example .env   # add your API keys
uv run uvicorn app.api.main:app --reload   # http://localhost:8000
```

See `.env.example` for required environment variables.
