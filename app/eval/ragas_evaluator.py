"""
RAGAS evaluation for the AI News Aggregator RAG pipeline.

Metrics:
  faithfulness      — digest claims supported by source articles (primary)
  answer_relevancy  — digest addresses user interests
  context_precision — retrieved articles are relevant to user interests

Run individual metrics to avoid rate limits:
  python main.py --digest-only --eval --metric faithfulness
  python main.py --digest-only --eval --metric answer_relevancy
  python main.py --digest-only --eval --metric context_precision
"""

# ── Compatibility patch ───────────────────────────────────────────────────────
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _fake = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI:
        pass
    _fake.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake

# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from app.config import config
from app.database.models import ArticleRecord

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AVAILABLE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
}


def safe_score(value) -> Optional[float]:
    """Return a rounded float, or None if the value is NaN/None/invalid."""
    if value is None:
        return None
    if isinstance(value, list):
        values = [v for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return round(sum(values) / len(values), 3) if values else None
    try:
        f = float(value)
        return round(f, 3) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _build_ragas_llm() -> LangchainLLMWrapper:
    groq_llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_retries=5,
        timeout=60,
    )
    return LangchainLLMWrapper(groq_llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(hf_embeddings)


class RAGASEvaluator:
    def __init__(self):
        self._embeddings = _build_ragas_embeddings()
        try:
            self._llm = _build_ragas_llm()
            self._llm_available = True
        except Exception as e:
            logger.warning("LLM unavailable (%s) — embedding-only metrics only", e)
            self._llm = None
            self._llm_available = False

    # ── Dataset construction ─────────────────────────────────────────────────

    def _build_dataset(
        self,
        digest: Dict,
        source_articles: List[ArticleRecord],
        user_interests: List[str],
    ) -> Dataset:
        sections = digest.get("sections", {})
        answer = "\n\n".join(
            sections.get(k, "")
            for k in ("highlights", "cross_source", "action_items")
            if sections.get(k)
        )

        # Limit context to 5 articles to reduce tokens per call
        context_pool = [
            f"{a.title}\n{a.description or ''}\n{(a.content or '')[:500]}"
            for a in source_articles[:5]
            if a.title
        ]

        rows: Dict[str, List] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        # Cap at 3 rows — RAGAS makes ~3 LLM calls per row per metric
        for interest in user_interests[:3]:
            rows["question"].append(interest)
            rows["answer"].append(answer)
            rows["contexts"].append(context_pool)
            rows["ground_truth"].append(interest)

        return Dataset.from_dict(rows)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        digest: Dict,
        source_articles: List[ArticleRecord],
        user_interests: Optional[List[str]] = None,
        metric: Optional[str] = None,
    ) -> Dict:
        """
        Run RAGAS evaluation.
        Pass metric="faithfulness"|"answer_relevancy"|"context_precision"
        to run a single metric and avoid rate limits.
        """
        if user_interests is None:
            user_interests = config.user_interests

        if not source_articles:
            logger.warning("No source articles — skipping RAGAS evaluation")
            return {}

        dataset = self._build_dataset(digest, source_articles, user_interests)

        # Decide which metrics to run
        if metric:
            if metric not in AVAILABLE_METRICS:
                return {"error": f"Unknown metric '{metric}'. Choose from: {list(AVAILABLE_METRICS)}"}
            metrics_to_run = [AVAILABLE_METRICS[metric]]
            logger.info("Running single RAGAS metric: %s", metric)
        else:
            metrics_to_run = list(AVAILABLE_METRICS.values())
            logger.info("Running all RAGAS metrics (%d rows × %d metrics)", len(dataset), len(metrics_to_run))

        scores: Dict = {}

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,
                run_config=RunConfig(max_workers=1, max_wait=60),
            )
            for key in AVAILABLE_METRICS:
                if key in result:
                    scores[key] = safe_score(result[key])
            scores["mode"] = "single" if metric else "full"
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            return {"error": str(e)}

        scores["metric_requested"] = metric or "all"
        scores["sample_count"] = len(dataset)
        scores["evaluated_at"] = datetime.now().isoformat()

        if "check_report" in digest:
            scores["fact_checker"] = digest["check_report"]

        return scores

    # ── Output ───────────────────────────────────────────────────────────────

    def print_report(self, scores: Dict) -> None:
        if not scores:
            print("\n=== RAGAS Evaluation — no results ===")
            return
        if "error" in scores:
            print(f"\n=== RAGAS Evaluation — FAILED ===\n  {scores['error']}")
            return

        print("\n" + "═" * 52)
        print("  RAGAS EVALUATION RESULTS")
        if scores.get("mode") == "single":
            print(f"  (single metric: {scores.get('metric_requested')})")
        print("═" * 52)

        def bar(score: float) -> str:
            return "█" * int(score * 20) + "░" * (20 - int(score * 20))

        metric_defs = [
            ("Faithfulness",      "faithfulness",      "No hallucinations"),
            ("Answer Relevancy",  "answer_relevancy",  "Digest addresses user interests"),
            ("Context Precision", "context_precision", "Retrieved articles are relevant"),
        ]

        for label, key, description in metric_defs:
            score = scores.get(key)
            if score is None:
                print(f"\n  —  {label:<20} N/A")
                continue
            grade = "✅" if score >= 0.8 else ("⚠️" if score >= 0.6 else "❌")
            print(f"\n  {grade} {label:<20} {score:.3f}  {bar(score)}")
            print(f"     {description}")

        if "fact_checker" in scores:
            fc = scores["fact_checker"]
            print(f"\n  Fact Checker:  {fc.get('verified','?')}/{fc.get('total_claims','?')} claims verified"
                  f"  |  {fc.get('revisions_made', 0)} revision(s)")
            pre = fc.get("pre_check_hallucination_rate", 0)
            post = fc.get("post_check_hallucination_rate", 0)
            print(f"  Hallucination: {pre:.1%} → {post:.1%}")

        print(f"\n  Samples: {scores.get('sample_count','?')} | {scores.get('evaluated_at','')[:19]}")
        print("═" * 52)

    def save_scores(self, scores: Dict, path: str = "eval_results.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        existing: List = []
        if Path(path).exists():
            try:
                with open(path) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append(scores)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info("Scores appended to %s", path)
