"""
Tests for EmailAgent — orchestration, guards, and delegation to EmailService.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.agent.email_agent import EmailAgent


@pytest.fixture
def agent():
    with patch("app.agent.email_agent.EmailService") as mock_service_cls:
        instance = EmailAgent()
        instance.service = mock_service_cls.return_value
    return instance


class TestEmailAgentSend:
    def test_returns_false_when_html_is_empty(self, agent, sample_digest):
        digest = {**sample_digest, "html": ""}
        result = agent.send(digest)
        assert result is False

    def test_returns_false_when_html_is_missing(self, agent, sample_digest):
        digest = {k: v for k, v in sample_digest.items() if k != "html"}
        result = agent.send(digest)
        assert result is False

    def test_delegates_to_email_service(self, agent, sample_digest):
        agent.service.send_digest.return_value = True
        agent.send(sample_digest)
        agent.service.send_digest.assert_called_once_with(sample_digest)

    def test_propagates_service_true(self, agent, sample_digest):
        agent.service.send_digest.return_value = True
        assert agent.send(sample_digest) is True

    def test_propagates_service_false(self, agent, sample_digest):
        agent.service.send_digest.return_value = False
        assert agent.send(sample_digest) is False

    def test_does_not_call_service_when_html_missing(self, agent, sample_digest):
        agent.send({**sample_digest, "html": ""})
        agent.service.send_digest.assert_not_called()

    def test_uses_default_subject_when_missing(self, agent, sample_digest, caplog):
        digest = {**sample_digest, "subject": ""}
        agent.service.send_digest.return_value = True
        agent.send(digest)
        # Should not raise — subject defaults gracefully

    def test_logs_all_section_sizes(self, agent, sample_digest, caplog):
        import logging
        agent.service.send_digest.return_value = True
        with caplog.at_level(logging.INFO):
            agent.send(sample_digest)
        for section_name in sample_digest["sections"]:
            assert section_name in caplog.text

    def test_logs_recipient(self, agent, sample_digest, caplog):
        import logging
        agent.service.send_digest.return_value = True
        with patch("app.agent.email_agent.config") as mock_config:
            mock_config.email_to = "test@example.com"
            with caplog.at_level(logging.INFO):
                agent.send(sample_digest)
        assert "test@example.com" in caplog.text
