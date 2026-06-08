import logging
from typing import Optional

from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


class EmailAgent:
    def __init__(self):
        self.service = EmailService()

    def send(self, digest: dict, to: Optional[str] = None) -> bool:
        """Send the digest. Pass `to` to override the default EMAIL_TO recipient."""
        subject = digest.get("subject", "AI News Digest")
        sections = digest.get("sections", {})
        html = digest.get("html", "")

        if not html:
            logger.error("Digest has no HTML — aborting send")
            return False

        logger.info("Preparing email: '%s'", subject)
        for name, body in sections.items():
            logger.info("  Section '%s': %d chars", name, len(body))

        success = self.service.send_digest(digest, to=to)

        if success:
            logger.info("Email delivered successfully")
        else:
            logger.error("Email delivery failed — check SMTP credentials")

        return success
