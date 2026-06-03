"""
Tests for Config — defaults, env var loading, user profile.
"""
import os
from unittest.mock import patch

import pytest

from app.config import Config


class TestConfigDefaults:
    def test_default_database_url(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert "postgresql" in cfg.database_url
        assert "ainews" in cfg.database_url

    def test_default_smtp_host_is_gmail(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.smtp_host == "smtp.gmail.com"

    def test_default_smtp_port(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.smtp_port == 587

    def test_default_digest_hours(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.digest_hours == 24

    def test_default_digest_size(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.digest_size == 10

    def test_default_llm_model_is_set(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.llm_model != ""

    def test_user_name_has_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.user_name != ""

    def test_user_role_has_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert cfg.user_role != ""

    def test_user_interests_is_non_empty_list(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert isinstance(cfg.user_interests, list)
        assert len(cfg.user_interests) > 0


class TestConfigEnvOverrides:
    def test_database_url_from_env(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://custom:5432/mydb"}):
            cfg = Config()
        assert cfg.database_url == "postgresql://custom:5432/mydb"

    def test_smtp_user_from_env(self):
        with patch.dict(os.environ, {"SMTP_USER": "me@gmail.com"}):
            cfg = Config()
        assert cfg.smtp_user == "me@gmail.com"

    def test_smtp_port_parsed_as_int(self):
        with patch.dict(os.environ, {"SMTP_PORT": "465"}):
            cfg = Config()
        assert cfg.smtp_port == 465
        assert isinstance(cfg.smtp_port, int)

    def test_digest_hours_parsed_as_int(self):
        with patch.dict(os.environ, {"DIGEST_HOURS": "48"}):
            cfg = Config()
        assert cfg.digest_hours == 48

    def test_digest_size_parsed_as_int(self):
        with patch.dict(os.environ, {"DIGEST_SIZE": "20"}):
            cfg = Config()
        assert cfg.digest_size == 20

    def test_llm_model_from_env(self):
        with patch.dict(os.environ, {"LLM_MODEL": "mistralai/Mistral-7B-Instruct-v0.3"}):
            cfg = Config()
        assert cfg.llm_model == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_email_to_from_env(self):
        with patch.dict(os.environ, {"EMAIL_TO": "test@example.com"}):
            cfg = Config()
        assert cfg.email_to == "test@example.com"

    def test_hf_api_key_from_env(self):
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf_testkey123"}):
            cfg = Config()
        assert cfg.huggingface_api_key == "hf_testkey123"


class TestConfigTypes:
    def test_user_interests_are_strings(self):
        cfg = Config()
        assert all(isinstance(i, str) for i in cfg.user_interests)

    def test_smtp_port_is_int(self):
        cfg = Config()
        assert isinstance(cfg.smtp_port, int)

    def test_digest_hours_is_int(self):
        cfg = Config()
        assert isinstance(cfg.digest_hours, int)
