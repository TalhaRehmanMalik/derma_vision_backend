"""
utils/email.py
Sends OTP emails via Gmail SMTP.
Set MAIL_USER and MAIL_PASS in .env (use a Gmail App Password, not your login password).
If credentials are missing the OTP is only printed to the log — useful during development.
"""
import os
import random
import string
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger("email")


def generate_otp(length: int = 6) -> str:
    """Return a random numeric OTP string of the given length."""
    return "".join(random.choices(string.digits, k=length))


def otp_expiry(minutes: int = 10) -> datetime:
    """Return a UTC datetime that is `minutes` from now."""
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def send_otp_email(recipient: str, otp: str) -> bool:
    """
    Send an HTML OTP email to the recipient.
    Returns True on success or when running without credentials (dev mode).
    Returns False if sending fails.
    """
    sender   = os.getenv("MAIL_USER")
    password = os.getenv("MAIL_PASS")

    # Development fallback — log OTP instead of sending
    if not sender or not password:
        logger.warning(f"Email credentials not configured — OTP for {recipient}: {otp}")
        return True

    body = f"""
    <div style="font-family: Arial; max-width: 500px; margin: auto;">
        <h2 style="color: #1A5276;">Derma Vision</h2>
        <p>Your verification code is:</p>
        <h1 style="letter-spacing: 8px; color: #2E86C1;">{otp}</h1>
        <p>This code expires in <strong>10 minutes</strong>.</p>
        <p style="color: gray; font-size: 12px;">
            If you did not request this, please ignore this email.
        </p>
    </div>
    """

    msg            = MIMEText(body, "html")
    msg["Subject"] = "Derma Vision — Email Verification Code"
    msg["From"]    = sender
    msg["To"]      = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())
        logger.info(f"OTP email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {recipient}: {e}")
        return False