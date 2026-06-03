"""
Tests for EmailService — SMTP interaction, config validation, message structure.
"""
import smtplib
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.email_service import EmailService


@pytest.fixture
def service():
    return EmailService()


@pytest.fixture
def full_config():
    return {
        "smtp_user": "sender@gmail.com",
        "smtp_password": "app-password",
        "email_from": "sender@gmail.com",
        "email_to": "recipient@gmail.com",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
    }


def patch_config(overrides=None):
    defaults = {
        "smtp_user": "sender@gmail.com",
        "smtp_password": "secret",
        "email_from": "sender@gmail.com",
        "email_to": "recipient@gmail.com",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
    }
    if overrides:
        defaults.update(overrides)

    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestConfigValidation:
    @pytest.mark.parametrize("missing_field", [
        "smtp_user", "smtp_password", "email_from", "email_to"
    ])
    def test_raises_when_field_missing(self, service, sample_digest, missing_field):
        config = patch_config({missing_field: ""})
        with patch("app.services.email_service.config", config):
            with pytest.raises(RuntimeError, match=missing_field.upper()):
                service.send_digest(sample_digest)

    def test_error_message_lists_all_missing_fields(self, service, sample_digest):
        config = patch_config({"smtp_user": "", "email_to": ""})
        with patch("app.services.email_service.config", config):
            with pytest.raises(RuntimeError) as exc:
                service.send_digest(sample_digest)
        assert "SMTP_USER" in str(exc.value)
        assert "EMAIL_TO" in str(exc.value)


class TestSmtpInteraction:
    def test_connects_to_configured_host_and_port(self, service, sample_digest):
        config = patch_config()
        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)
        mock_smtp.assert_called_once_with("smtp.gmail.com", 587)

    def test_calls_starttls(self, service, sample_digest):
        config = patch_config()
        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)
        mock_server.starttls.assert_called_once()

    def test_logs_in_with_credentials(self, service, sample_digest):
        config = patch_config()
        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)
        mock_server.login.assert_called_once_with("sender@gmail.com", "secret")

    def test_sends_to_correct_recipient(self, service, sample_digest):
        config = patch_config()
        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)
        args = mock_server.sendmail.call_args[0]
        assert args[0] == "sender@gmail.com"
        assert args[1] == "recipient@gmail.com"

    def test_returns_true_on_success(self, service, sample_digest):
        config = patch_config()
        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                result = service.send_digest(sample_digest)
        assert result is True

    def test_returns_false_on_smtp_error(self, service, sample_digest):
        config = patch_config()
        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP", side_effect=smtplib.SMTPException("connection refused")):
                result = service.send_digest(sample_digest)
        assert result is False


class TestMessageStructure:
    def test_email_has_correct_subject(self, service, sample_digest):
        import email
        import email.header

        config = patch_config()
        sent_message = None

        def capture_sendmail(from_addr, to_addr, msg_string):
            nonlocal sent_message
            sent_message = msg_string

        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_server.sendmail.side_effect = capture_sendmail
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)

        # Decode the MIME-encoded subject (may be base64/utf-8 encoded)
        msg = email.message_from_string(sent_message)
        decoded_subject = email.header.decode_header(msg["Subject"])[0]
        subject_text = decoded_subject[0]
        if isinstance(subject_text, bytes):
            subject_text = subject_text.decode(decoded_subject[1] or "utf-8")
        assert "AI News Digest" in subject_text

    def test_email_contains_html_part(self, service, sample_digest):
        config = patch_config()
        sent_message = None

        def capture_sendmail(from_addr, to_addr, msg_string):
            nonlocal sent_message
            sent_message = msg_string

        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_server.sendmail.side_effect = capture_sendmail
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)

        assert "text/html" in sent_message

    def test_email_contains_plain_text_fallback(self, service, sample_digest):
        config = patch_config()
        sent_message = None

        def capture_sendmail(from_addr, to_addr, msg_string):
            nonlocal sent_message
            sent_message = msg_string

        with patch("app.services.email_service.config", config):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_server.sendmail.side_effect = capture_sendmail
                mock_smtp.return_value.__enter__.return_value = mock_server
                service.send_digest(sample_digest)

        assert "text/plain" in sent_message
