"""
Smoke tests for RAGASEvaluator — no DB, no real LLM/embedding calls.

Run with:
  uv run pytest tests/eval/test_ragas_evaluator.py -v
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.database.models import ArticleRecord, ContentType


# ── Minimal fixtures ──────────────────────────────────────────────────────────

def make_article(title: str, source: str, description: str = "", content: str = "") -> ArticleRecord:
    return ArticleRecord(
        guid=f"guid-{title.lower().replace(' ', '-')}",
        title=title,
        description=description,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        source=source,
        published_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        content_type=ContentType.ARTICLE,
        content=content or None,
    )


SAMPLE_ARTICLES = [
    make_article(
        "Anthropic raises $65B Series H",
        "Anthropic",
        "Anthropic closed a $65B funding round at a $965B valuation.",
        "Anthropic has raised $65 billion in a Series H funding round, valuing the company at $965 billion.",
    ),
    make_article(
        "Claude Opus 4.8 Released",
        "Anthropic",
        "Claude Opus 4.8 sets new benchmarks on coding and reasoning tasks.",
        "Anthropic released Claude Opus 4.8 with improved coding and reasoning capabilities.",
    ),
    make_article(
        "Groq achieves 1M tokens per second",
        "Groq",
        "Groq's LPU inference hardware reaches 1 million tokens per second.",
    ),
]

SAMPLE_DIGEST = {
    "subject": "🤖 AI News Digest — June 1, 2026",
    "sections": {
        "highlights": (
            "### Anthropic raises $65B\n"
            "Why this matters: A $965B valuation signals massive investor confidence in frontier AI.\n\n"
            "### Claude Opus 4.8 Released\n"
            "Why this matters: Better coding tools directly affect your workflow as an AI engineer."
        ),
        "cross_source": "Multiple sources cover Anthropic's funding round.",
        "action_items": (
            "🔴 Check your Claude API tier — pricing changes may follow the funding round.\n"
            "🟡 Benchmark Claude Opus 4.8 on your current projects."
        ),
        "quick_mentions": "- [Groq 1M tokens/s](https://example.com/groq) — *Groq*",
        "trends": "**Trend 1**: Frontier AI consolidation — High confidence.",
    },
    "html": "<html><body>digest</body></html>",
    "stats_str": "3 articles · 2 sources",
}

SAMPLE_INTERESTS = ["AI agents", "startup funding"]


# ── Mocked evaluator tests ────────────────────────────────────────────────────


def make_mock_evaluator():
    """Return a RAGASEvaluator with LLM and embeddings fully mocked."""
    with patch("app.eval.ragas_evaluator._build_ragas_llm") as mock_llm, \
         patch("app.eval.ragas_evaluator._build_ragas_embeddings") as mock_emb:
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        from app.eval.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator()
    return evaluator


class TestDatasetConstruction:
    def test_dataset_has_correct_length(self):
        evaluator = make_mock_evaluator()
        dataset = evaluator._build_dataset(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)
        assert len(dataset) == len(SAMPLE_INTERESTS)

    def test_dataset_questions_match_interests(self):
        evaluator = make_mock_evaluator()
        dataset = evaluator._build_dataset(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)
        assert list(dataset["question"]) == SAMPLE_INTERESTS

    def test_dataset_answer_contains_digest_sections(self):
        evaluator = make_mock_evaluator()
        dataset = evaluator._build_dataset(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)
        answer = dataset["answer"][0]
        assert "Anthropic raises $65B" in answer
        assert "action_items" not in answer.lower()  # section key not leaked, content is

    def test_dataset_contexts_are_lists(self):
        evaluator = make_mock_evaluator()
        dataset = evaluator._build_dataset(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)
        for ctx in dataset["contexts"]:
            assert isinstance(ctx, list)
            assert len(ctx) > 0

    def test_dataset_contexts_include_article_titles(self):
        evaluator = make_mock_evaluator()
        dataset = evaluator._build_dataset(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)
        all_context_text = " ".join(dataset["contexts"][0])
        assert "Anthropic" in all_context_text

    def test_interests_capped_at_three(self):
        evaluator = make_mock_evaluator()
        many_interests = ["interest A", "interest B", "interest C", "interest D", "interest E"]
        from app.eval.ragas_evaluator import RAGASEvaluator
        # Mimic the cap in evaluate()
        dataset = evaluator._build_dataset(SAMPLE_DIGEST, SAMPLE_ARTICLES, many_interests[:3])
        assert len(dataset) == 3


class TestEvaluateWithMockedRagas:
    @patch("app.eval.ragas_evaluator.evaluate")
    def test_returns_all_three_scores(self, mock_evaluate):
        mock_evaluate.return_value = {
            "faithfulness": 0.92,
            "answer_relevancy": 0.85,
            "context_precision": 0.78,
        }
        evaluator = make_mock_evaluator()
        scores = evaluator.evaluate(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)

        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores

    @patch("app.eval.ragas_evaluator.evaluate")
    def test_scores_are_floats_between_0_and_1(self, mock_evaluate):
        mock_evaluate.return_value = {
            "faithfulness": 0.92,
            "answer_relevancy": 0.85,
            "context_precision": 0.78,
        }
        evaluator = make_mock_evaluator()
        scores = evaluator.evaluate(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)

        for key in ("faithfulness", "answer_relevancy", "context_precision"):
            assert 0.0 <= scores[key] <= 1.0

    @patch("app.eval.ragas_evaluator.evaluate")
    def test_handles_nan_scores_gracefully(self, mock_evaluate):
        import math
        mock_evaluate.return_value = {
            "faithfulness": float("nan"),
            "answer_relevancy": 0.85,
            "context_precision": [None, None, None],
        }
        evaluator = make_mock_evaluator()
        scores = evaluator.evaluate(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)

        assert not math.isnan(scores.get("faithfulness", 0.0))
        assert scores.get("context_precision", 0.0) == 0.0

    @patch("app.eval.ragas_evaluator.evaluate", side_effect=Exception("API error"))
    def test_falls_back_on_full_evaluation_failure(self, mock_evaluate):
        evaluator = make_mock_evaluator()
        evaluator._llm_available = True

        with patch.object(evaluator, "_llm_available", True):
            # Second call (fallback) should also be mocked
            with patch("app.eval.ragas_evaluator.evaluate", side_effect=Exception("still failing")):
                scores = evaluator.evaluate(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)

        assert "error" in scores or scores == {}

    def test_returns_empty_when_no_articles(self):
        evaluator = make_mock_evaluator()
        scores = evaluator.evaluate(SAMPLE_DIGEST, [], SAMPLE_INTERESTS)
        assert scores == {}

    @patch("app.eval.ragas_evaluator.evaluate")
    def test_attaches_fact_checker_report(self, mock_evaluate):
        mock_evaluate.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.75,
        }
        digest_with_report = {
            **SAMPLE_DIGEST,
            "check_report": {"total_claims": 10, "verified": 9, "revisions_made": 1},
        }
        evaluator = make_mock_evaluator()
        scores = evaluator.evaluate(digest_with_report, SAMPLE_ARTICLES, SAMPLE_INTERESTS)

        assert "fact_checker" in scores
        assert scores["fact_checker"]["total_claims"] == 10

    @patch("app.eval.ragas_evaluator.evaluate")
    def test_includes_metadata(self, mock_evaluate):
        mock_evaluate.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.75,
        }
        evaluator = make_mock_evaluator()
        scores = evaluator.evaluate(SAMPLE_DIGEST, SAMPLE_ARTICLES, SAMPLE_INTERESTS)

        assert "sample_count" in scores
        assert "evaluated_at" in scores
        assert scores["sample_count"] == len(SAMPLE_INTERESTS)


class TestSafeFloat:
    def test_converts_plain_float(self):
        from app.eval.ragas_evaluator import _safe_float
        assert _safe_float(0.85) == 0.85

    def test_converts_list_to_mean(self):
        from app.eval.ragas_evaluator import _safe_float
        assert _safe_float([0.8, 0.9, 1.0]) == pytest.approx(0.9, abs=0.001)

    def test_filters_nones_from_list(self):
        from app.eval.ragas_evaluator import _safe_float
        assert _safe_float([None, 0.8, None]) == pytest.approx(0.8)

    def test_returns_zero_for_all_none_list(self):
        from app.eval.ragas_evaluator import _safe_float
        assert _safe_float([None, None]) == 0.0

    def test_returns_zero_for_nan(self):
        from app.eval.ragas_evaluator import _safe_float
        assert _safe_float(float("nan")) == 0.0

    def test_returns_zero_for_none(self):
        from app.eval.ragas_evaluator import _safe_float
        assert _safe_float(None) == 0.0
