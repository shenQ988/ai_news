import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)


class EmailService:
    def send_digest(self, digest: dict, to: Optional[str] = None) -> bool:
        """Send the digest email. Returns True on success, False on any failure."""
        recipient = to or os.getenv("EMAIL_TO") or config.email_to
        if not recipient:
            logger.error(
                "No email recipient. Set EMAIL_TO env var or pass a profile email. "
                "Skipping email send."
            )
            return False

        sender = config.smtp_user or os.getenv("SMTP_USER")
        password = config.smtp_password or os.getenv("SMTP_PASSWORD")
        if not sender or not password:
            logger.error("SMTP credentials missing. Skipping email send.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = digest["subject"]
        msg["From"] = config.email_from or sender
        msg["To"] = recipient

        plain = "\n\n".join(
            f"=== {k.upper()} ===\n{v}"
            for k, v in digest.get("sections", {}).items()
        )
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(digest["html"], "html"))

        try:
            with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
            logger.info("Digest sent to %s", recipient)
            return True
        except Exception as e:
            logger.error("Failed to send email: %s", e)
            return False
