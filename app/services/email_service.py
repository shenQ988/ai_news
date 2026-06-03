import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import config

logger = logging.getLogger(__name__)


class EmailService:
    def send_digest(self, digest: dict) -> bool:
        """Send the digest email. Returns True on success."""
        missing = [k for k in ("smtp_user", "smtp_password", "email_from", "email_to")
                   if not getattr(config, k)]
        if missing:
            raise RuntimeError(f"Email config incomplete — set these in .env: {', '.join(missing).upper()}")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = digest["subject"]
        msg["From"] = config.email_from
        msg["To"] = config.email_to

        # Plain-text fallback
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
                server.login(config.smtp_user, config.smtp_password)
                server.sendmail(config.email_from, config.email_to, msg.as_string())
            logger.info("Digest sent to %s", config.email_to)
            return True
        except Exception as e:
            logger.error("Failed to send email: %s", e)
            return False
