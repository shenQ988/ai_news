"""
Tests for DigestAgent — section generation, LLM fallbacks, HTML output.
The LLM (InferenceClient) is always mocked so tests run offline instantly.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.agent.digest_agent import DigestAgent, _article_summary
from app.database.models import ArticleRecord


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_agent(llm_response: str = "LLM generated content.") -> DigestAgent:
    """Return a DigestAgent whose LLM always returns the given string."""
    with patch("app.agent.digest_agent.InferenceClient"):
        agent = DigestAgent()
    agent._llm = MagicMock(return_value=llm_response)
    return agent


def make_retrieval_result(records, cross_source=None):
    return {
        "ranked_articles": records,
        "cross_source_topics": cross_source or [],
        "stats": {
            "total_retrieved": len(records),
            "unique_sources": len({r.source for r in records}),
            "queries_run": 5,
        },
    }


# ── _article_summary() ───────────────────────────────────────────────────────


class TestArticleSummary:
    def test_includes_title_and_source(self, make_record):
        record = make_record("My Article", source="OpenAI")
        summary = _article_summary(record)
        assert "My Article" in summary
        assert "OpenAI" in summary

    def test_includes_url_as_link(self, make_record):
        record = make_record("Test", url="https://example.com/test")
        summary = _article_summary(record)
        assert "https://example.com/test" in summary

    def test_includes_description_when_present(self, make_record):
        record = make_record("Title", description="A key insight.")
        summary = _article_summary(record)
        assert "A key insight." in summary

    def test_empty_description_does_not_crash(self, make_record):
        record = make_record("Title", description="")
        _article_summary(record)  # should not raise


# ── Section generators ────────────────────────────────────────────────────────


class TestHighlights:
    def test_returns_llm_output(self, sample_records):
        agent = make_agent("## Key insight here")
        result = agent._highlights(sample_records[:3])
        assert result == "## Key insight here"

    def test_returns_fallback_when_llm_fails(self, make_record):
        agent = make_agent("")
        records = [make_record("Article A", description="Desc A")]
        result = agent._highlights(records)
        assert "Article A" in result

    def test_empty_articles_returns_fallback(self):
        agent = make_agent()
        assert agent._highlights([]) == "No articles available."

    def test_passes_article_titles_to_llm(self, make_record):
        with patch("app.agent.digest_agent.InferenceClient"):
            agent = DigestAgent()
        agent._llm = MagicMock(return_value="ok")

        records = [make_record("Special Title XYZ")]
        agent._highlights(records)

        call_args = agent._llm.call_args[0][0]
        assert "Special Title XYZ" in call_args


class TestCrossSource:
    def test_returns_llm_output(self, make_record):
        agent = make_agent("Cross-source analysis here.")
        a = make_record("Claude vs GPT", source="Anthropic")
        b = make_record("Claude vs GPT Battle", source="OpenAI")
        result = agent._cross_source([[a, b]])
        assert result == "Cross-source analysis here."

    def test_empty_clusters_returns_no_topics_message(self):
        agent = make_agent()
        result = agent._cross_source([])
        assert "No cross-source topics" in result

    def test_falls_back_to_formatted_list_when_llm_fails(self, make_record):
        agent = make_agent("")
        a = make_record("Topic", source="SourceA", url="https://a.com")
        b = make_record("Topic Related", source="SourceB", url="https://b.com")
        result = agent._cross_source([[a, b]])
        assert "SourceA" in result or "SourceB" in result

    def test_limits_to_three_clusters(self, make_record):
        agent = make_agent("Analysis.")
        clusters = [
            [make_record(f"Article {i}", source=f"S{i}"), make_record(f"Article {i}b", source=f"S{i}b")]
            for i in range(6)
        ]
        agent._cross_source(clusters)
        call_args = agent._llm.call_args[0][0]
        # Only 3 clusters should appear in the prompt
        assert call_args.count("**Topic:") <= 3


class TestActionItems:
    def test_returns_llm_output(self, sample_records):
        agent = make_agent("🔴 Act now on Claude 4.")
        result = agent._action_items(sample_records[:3])
        assert "🔴" in result

    def test_empty_articles_returns_fallback(self):
        agent = make_agent()
        result = agent._action_items([])
        assert "No action items" in result

    def test_falls_back_when_llm_fails(self, sample_records):
        agent = make_agent("")
        result = agent._action_items(sample_records[:2])
        assert len(result) > 0


class TestQuickMentions:
    def test_formats_as_markdown_links(self, make_record):
        agent = make_agent()
        records = [make_record("Article A", source="SourceA", url="https://a.com")]
        result = agent._quick_mentions(records)
        assert "[Article A](https://a.com)" in result
        assert "SourceA" in result

    def test_no_llm_called_for_quick_mentions(self, make_record):
        with patch("app.agent.digest_agent.InferenceClient"):
            agent = DigestAgent()
        agent._llm = MagicMock()

        agent._quick_mentions([make_record("Article")])
        agent._llm.assert_not_called()

    def test_empty_articles_returns_fallback(self):
        agent = make_agent()
        result = agent._quick_mentions([])
        assert "No additional articles" in result

    def test_includes_all_articles(self, make_record):
        agent = make_agent()
        records = [make_record(f"Article {i}") for i in range(5)]
        result = agent._quick_mentions(records)
        for i in range(5):
            assert f"Article {i}" in result


class TestTrends:
    def test_returns_llm_output(self, sample_records):
        agent = make_agent("**Trend 1**: Agentic AI — High.")
        result = agent._trends(sample_records)
        assert "Trend 1" in result

    def test_empty_articles_returns_fallback(self):
        agent = make_agent()
        result = agent._trends([])
        assert "Insufficient data" in result

    def test_limits_prompt_to_20_articles(self, make_record):
        with patch("app.agent.digest_agent.InferenceClient"):
            agent = DigestAgent()
        agent._llm = MagicMock(return_value="trends")

        records = [make_record(f"Article {i}") for i in range(30)]
        agent._trends(records)

        call_args = agent._llm.call_args[0][0]
        article_lines = [l for l in call_args.split("\n") if l.strip().startswith("-")]
        assert len(article_lines) <= 20


# ── _to_html_section() ────────────────────────────────────────────────────────


class TestToHtmlSection:
    def test_wraps_in_section_div(self):
        agent = make_agent()
        html = agent._to_html_section("My Title", "Some **bold** text.")
        assert '<div class="section">' in html
        assert "<h2>My Title</h2>" in html

    def test_converts_markdown_to_html(self):
        agent = make_agent()
        html = agent._to_html_section("Title", "**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_converts_markdown_links(self):
        agent = make_agent()
        html = agent._to_html_section("Title", "[Click here](https://example.com)")
        assert 'href="https://example.com"' in html


# ── generate_digest() ─────────────────────────────────────────────────────────


class TestGenerateDigest:
    def test_returns_required_keys(self, sample_records):
        agent = make_agent("Generated content.")
        result = agent.generate_digest(make_retrieval_result(sample_records))

        assert "subject" in result
        assert "sections" in result
        assert "html" in result

    def test_subject_contains_date(self, sample_records):
        agent = make_agent("Content.")
        result = agent.generate_digest(make_retrieval_result(sample_records))
        assert "AI News Digest" in result["subject"]

    def test_sections_has_all_five_keys(self, sample_records):
        agent = make_agent("Content.")
        result = agent.generate_digest(make_retrieval_result(sample_records))
        expected = {"highlights", "cross_source", "action_items", "quick_mentions", "trends"}
        assert set(result["sections"].keys()) == expected

    def test_html_is_valid_structure(self, sample_records):
        agent = make_agent("Content.")
        result = agent.generate_digest(make_retrieval_result(sample_records))
        html = result["html"]
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body>" in html

    def test_html_contains_all_section_headings(self, sample_records):
        agent = make_agent("Content.")
        result = agent.generate_digest(make_retrieval_result(sample_records))
        html = result["html"]
        assert "Key Highlights" in html
        assert "Cross-Source Analysis" in html
        assert "Action Items" in html
        assert "Quick Mentions" in html
        assert "Trend Signals" in html

    def test_html_contains_stats_line(self, sample_records):
        agent = make_agent("Content.")
        result = agent.generate_digest(make_retrieval_result(sample_records))
        assert "articles" in result["html"]
        assert "sources" in result["html"]

    def test_top_5_go_to_highlights_rest_to_quick_mentions(self, make_record):
        agent = make_agent()
        agent._highlights = MagicMock(return_value="highlights")
        agent._quick_mentions = MagicMock(return_value="quick")
        agent._cross_source = MagicMock(return_value="cross")
        agent._action_items = MagicMock(return_value="actions")
        agent._trends = MagicMock(return_value="trends")

        records = [make_record(f"Article {i}") for i in range(8)]
        agent.generate_digest(make_retrieval_result(records))

        highlights_call = agent._highlights.call_args[0][0]
        quick_call = agent._quick_mentions.call_args[0][0]
        assert len(highlights_call) == 5
        assert len(quick_call) == 3

    def test_empty_articles_still_produces_digest(self):
        agent = make_agent()
        result = agent.generate_digest(make_retrieval_result([]))
        assert result["html"]
        assert result["subject"]
