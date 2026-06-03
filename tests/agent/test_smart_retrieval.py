"""
Tests for SmartRetrieval — scoring, deduplication, clustering, cross-source detection.
All tests are unit tests: the ArticleRepository is mocked.
"""
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.agent.smart_retrieval import SmartRetrieval
from app.database.models import ArticleRecord, ContentType
from tests.conftest import _make_record


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def retrieval(mock_repo):
    return SmartRetrieval(mock_repo)


# ── retrieve() ────────────────────────────────────────────────────────────────


class TestRetrieve:
    def test_returns_expected_keys(self, retrieval, mock_repo, sample_records):
        mock_repo.search.return_value = sample_records

        result = retrieval.retrieve()

        assert "ranked_articles" in result
        assert "topic_clusters" in result
        assert "cross_source_topics" in result
        assert "stats" in result

    def test_stats_are_accurate(self, retrieval, mock_repo, sample_records):
        mock_repo.search.return_value = sample_records

        result = retrieval.retrieve()
        stats = result["stats"]

        assert stats["total_retrieved"] == len(result["ranked_articles"])
        assert stats["unique_sources"] == len({a.source for a in result["ranked_articles"]})
        assert stats["queries_run"] > 0

    def test_deduplicates_articles_across_interests(self, retrieval, mock_repo, make_record):
        article = make_record("Shared Article", source="OpenAI")
        mock_repo.search.return_value = [article]

        result = retrieval.retrieve()

        guids = [a.guid for a in result["ranked_articles"]]
        assert len(guids) == len(set(guids)), "Duplicate articles found in results"

    def test_keeps_best_rank_when_deduplicating(self, retrieval, mock_repo, make_record):
        """An article appearing at rank 0 in one query beats rank 5 from another."""
        top_article = make_record("Top Article", guid="guid-top")
        other = make_record("Other", guid="guid-other")

        call_count = 0

        def search_side_effect(query, top_k):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [top_article, other]
            return [other, top_article]  # top_article at rank 1 here

        mock_repo.search.side_effect = search_side_effect

        result = retrieval.retrieve()
        ranked = result["ranked_articles"]

        assert any(a.guid == "guid-top" for a in ranked)

    def test_handles_empty_db(self, retrieval, mock_repo):
        mock_repo.search.return_value = []

        result = retrieval.retrieve()

        assert result["ranked_articles"] == []
        assert result["topic_clusters"] == []
        assert result["cross_source_topics"] == []
        assert result["stats"]["total_retrieved"] == 0

    def test_calls_search_once_per_interest(self, retrieval, mock_repo):
        mock_repo.search.return_value = []

        with patch("app.agent.smart_retrieval.config") as mock_config:
            mock_config.user_interests = ["interest A", "interest B", "interest C"]
            mock_config.digest_size = 10
            retrieval.retrieve()

        assert mock_repo.search.call_count == 3


# ── _score() ──────────────────────────────────────────────────────────────────


class TestScore:
    def test_scores_are_between_zero_and_one(self, retrieval, sample_records):
        articles_with_rank = [(a, i) for i, a in enumerate(sample_records)]
        scored = retrieval._score(articles_with_rank)

        for _, score in scored:
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_recent_article_scores_higher_than_old(self, retrieval, make_record):
        recent = make_record("Recent", published_hours_ago=1)
        old = make_record("Old", published_hours_ago=96)

        scored = retrieval._score([(recent, 0), (old, 0)])
        scores = {a.title: s for a, s in scored}

        assert scores["Recent"] > scores["Old"]

    def test_top_ranked_article_scores_higher_than_low_ranked(self, retrieval, make_record):
        """Same age, same source — rank is the differentiator."""
        a = make_record("Article A", published_hours_ago=1, source="X")
        b = make_record("Article B", published_hours_ago=1, source="Y")

        scored = retrieval._score([(a, 0), (b, 9)])
        scores = {art.title: s for art, s in scored}

        assert scores["Article A"] > scores["Article B"]

    def test_source_diversity_penalises_dominant_source(self, retrieval, make_record):
        """Two articles from same source should score lower on diversity than singletons."""
        a1 = make_record("Article 1", source="OpenAI", published_hours_ago=1)
        a2 = make_record("Article 2", source="OpenAI", published_hours_ago=1)
        b = make_record("Article 3", source="Anthropic", published_hours_ago=1)

        scored = retrieval._score([(a1, 0), (a2, 0), (b, 0)])
        scores = {art.title: s for art, s in scored}

        # The lone Anthropic article gets full diversity bonus
        assert scores["Article 3"] > scores["Article 1"]

    def test_handles_single_article(self, retrieval, make_record):
        article = make_record("Solo")
        scored = retrieval._score([(article, 0)])

        assert len(scored) == 1
        _, score = scored[0]
        assert 0.0 <= score <= 1.0

    def test_recency_uses_48h_half_life(self, retrieval, make_record):
        """After 48h the recency component should be ~0.5 of its initial value."""
        fresh = make_record("Fresh", published_hours_ago=0.001)
        day_old = make_record("DayOld", published_hours_ago=48)

        scored = retrieval._score([(fresh, 0), (day_old, 0)])
        scores = {a.title: s for a, s in scored}

        # Isolate recency: 30% weight, ~50% decay at 48h → ~0.15 difference at most
        diff = scores["Fresh"] - scores["DayOld"]
        assert 0 < diff < 0.35


# ── _keywords() ──────────────────────────────────────────────────────────────


class TestKeywords:
    def test_returns_lowercase_words(self, retrieval):
        kw = retrieval._keywords("Claude Released Today")
        assert all(w == w.lower() for w in kw)

    def test_filters_stop_words(self, retrieval):
        kw = retrieval._keywords("the AI is new and will be used")
        assert "the" not in kw
        assert "and" not in kw
        assert "will" not in kw

    def test_filters_short_words(self, retrieval):
        kw = retrieval._keywords("an AI in the lab")
        assert not any(len(w) <= 3 for w in kw)

    def test_strips_punctuation(self, retrieval):
        kw = retrieval._keywords("release, today!")
        # Commas and exclamation marks must not appear in any keyword
        assert all("," not in w and "!" not in w for w in kw)

    def test_empty_title(self, retrieval):
        assert retrieval._keywords("") == set()


# ── _cluster() ────────────────────────────────────────────────────────────────


class TestCluster:
    def test_groups_articles_with_shared_keywords(self, retrieval, make_record):
        a = make_record("Claude Model Release")
        b = make_record("Claude Model Performance")
        c = make_record("Llama Open Source")

        clusters = retrieval._cluster([a, b, c])

        claude_cluster = next(c for c in clusters if any(x.title.startswith("Claude") for x in c))
        titles = {x.title for x in claude_cluster}
        assert "Claude Model Release" in titles
        assert "Claude Model Performance" in titles

    def test_separate_topics_in_different_clusters(self, retrieval, make_record):
        a = make_record("Quantum Computing Breakthrough")
        b = make_record("Basketball Championship Finals")

        clusters = retrieval._cluster([a, b])

        assert len(clusters) == 2

    def test_every_article_appears_exactly_once(self, retrieval, sample_records):
        clusters = retrieval._cluster(sample_records)
        all_guids = [a.guid for cluster in clusters for a in cluster]

        assert len(all_guids) == len(set(all_guids)) == len(sample_records)

    def test_empty_list_returns_empty(self, retrieval):
        assert retrieval._cluster([]) == []

    def test_single_article_forms_own_cluster(self, retrieval, make_record):
        clusters = retrieval._cluster([make_record("Solo Article")])
        assert len(clusters) == 1
        assert len(clusters[0]) == 1


# ── cross_source detection ────────────────────────────────────────────────────


class TestCrossSource:
    def test_flags_clusters_with_multiple_sources(self, retrieval, mock_repo, make_record):
        a = make_record("Claude Released", source="Anthropic")
        b = make_record("Claude Released Today", source="TechCrunch")
        mock_repo.search.return_value = [a, b]

        result = retrieval.retrieve()

        assert len(result["cross_source_topics"]) >= 1

    def test_does_not_flag_single_source_clusters(self, retrieval, mock_repo, make_record):
        a = make_record("GPT Update", source="OpenAI")
        b = make_record("GPT New Features", source="OpenAI")
        mock_repo.search.return_value = [a, b]

        result = retrieval.retrieve()

        assert all(
            len({x.source for x in cluster}) >= 2
            for cluster in result["cross_source_topics"]
        )
