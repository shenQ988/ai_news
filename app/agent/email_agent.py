import logging

from app.config import config
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


class EmailAgent:
    def __init__(self):
        self.service = EmailService()

    def send(self, digest: dict) -> bool:
        """
        Take the digest produced by DigestAgent and send it to the configured recipient.
        Logs a preview of the subject and section lengths before sending.
        """
        subject = digest.get("subject", "AI News Digest")
        sections = digest.get("sections", {})
        html = digest.get("html", "")

        if not html:
            logger.error("Digest has no HTML — aborting send")
            return False

        logger.info("Preparing email: '%s'", subject)
        logger.info("Recipient: %s", config.email_to)
        for name, body in sections.items():
            logger.info("  Section '%s': %d chars", name, len(body))

        success = self.service.send_digest(digest)

        if success:
            logger.info("Email delivered successfully to %s", config.email_to)
        else:
            logger.error("Email delivery failed — check SMTP credentials in .env")

        return success
