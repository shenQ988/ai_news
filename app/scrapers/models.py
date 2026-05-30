from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Article(BaseModel):
    title: str
    description: str
    url: str
    guid: str
    published_at: datetime
    source: str  # e.g. "OpenAI", "Anthropic", "Meta AI"
    category: Optional[str] = None
    content: Optional[str] = None