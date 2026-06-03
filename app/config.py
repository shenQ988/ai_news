import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Database
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://ainews:ainews@localhost:5432/ainews"))

    # HuggingFace
    huggingface_api_key: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"))

    # Email
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    email_from: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))
    email_to: str = field(default_factory=lambda: os.getenv("EMAIL_TO", ""))

    # Digest
    digest_hours: int = field(default_factory=lambda: int(os.getenv("DIGEST_HOURS", "24")))
    digest_size: int = field(default_factory=lambda: int(os.getenv("DIGEST_SIZE", "10")))
    digest_sources: list[str] = field(default_factory=list)

    # User Profile (drives personalization)
    user_name: str = field(default_factory=lambda: os.getenv("USER_NAME", "Qing"))
    user_role: str = field(default_factory=lambda: os.getenv("USER_ROLE", "Software engineer interested in building AI-powered products"))
    user_interests: list[str] = field(default_factory=lambda: [
        "AI agents and agentic workflows",
        "startup trends and funding",
        "LLM applications in production",
        "developer tools and coding assistants",
        "open source AI models",
    ])


config = Config()