"""
routes/auth.py
Authentication endpoints — register, OTP verify, login, password reset.

POST /api/auth/register        — Create account + send OTP email
POST /api/auth/verify-otp      — Verify OTP → returns JWT
POST /api/auth/login           — Login with email+password → returns JWT
POST /api/auth/forgot-password — Send new OTP to email
POST /api/auth/reset-password  — Reset password using OTP
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from bcrypt import hashpw, checkpw, gensalt

from database import get_db
from models import User
from utils.auth import create_token
from utils.email import generate_otp, otp_expiry, send_otp_email
from utils.logger import get_logger

logger = get_logger("routes.auth")
router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ── Request schemas ───────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email:    EmailStr
    password: str

class OTPRequest(BaseModel):
    email: EmailStr
    otp:   str

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class ForgotRequest(BaseModel):
    email: EmailStr

class ResetRequest(BaseModel):
    email:        EmailStr
    otp:          str
    new_password: str


# ── Register ──────────────────────────────────────────────────
@router.post("/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if db.query(User).filter_by(email=body.email).first():
        raise HTTPException(409, "Email already registered")
    if db.query(User).filter_by(username=body.username).first():
        raise HTTPException(409, "Username already taken")

    hashed = hashpw(body.password.encode(), gensalt()).decode()
    otp    = generate_otp()

    user = User(
        username      = body.username,
        email         = body.email,
        password_hash = hashed,
        otp           = otp,
        otp_expires   = otp_expiry(10),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    send_otp_email(body.email, otp)
    logger.info(f"New user registered: {body.username} ({body.email})")

    return {
        "message": "Registered successfully. Check your email for OTP.",
        "user_id": user.id
    }


# ── Verify OTP ────────────────────────────────────────────────
@router.post("/verify-otp")
def verify_otp(body: OTPRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.is_verified:
        return {"message": "Already verified"}
    if user.otp != body.otp:
        logger.warning(f"Invalid OTP attempt for {body.email}")
        raise HTTPException(400, "Invalid OTP")

    # Normalize timezone before comparing — MySQL stores datetimes without tz info
    if user.otp_expires:
        expires = user.otp_expires
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires:
            raise HTTPException(400, "OTP expired. Request a new one.")

    user.is_verified = True
    user.otp         = None
    user.otp_expires = None
    db.commit()

    token = create_token(user.id)
    logger.info(f"User verified: {body.email}")
    return {
        "message": "Email verified successfully",
        "token":   token,
        "user":    user.to_dict()
    }


# ── Login ─────────────────────────────────────────────────────
@router.post("/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email).first()

    # Check password — same error message for wrong email OR wrong password (security)
    if not user or not checkpw(body.password.encode(), user.password_hash.encode()):
        logger.warning(f"Failed login attempt for {body.email}")
        raise HTTPException(401, "Invalid email or password")
    if not user.is_verified:
        raise HTTPException(403, "Email not verified. Check your inbox for OTP.")

    token = create_token(user.id)
    logger.info(f"User logged in: {body.email}")
    return {"token": token, "user": user.to_dict()}


# ── Forgot Password ───────────────────────────────────────────
@router.post("/forgot-password")
def forgot_password(body: ForgotRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email).first()

    # Always return the same message — do not reveal whether the email exists
    if user:
        otp              = generate_otp()
        user.otp         = otp
        user.otp_expires = otp_expiry(10)
        db.commit()
        send_otp_email(body.email, otp)
        logger.info(f"Password reset OTP sent to {body.email}")

    return {"message": "If that email is registered, an OTP has been sent."}


# ── Reset Password ────────────────────────────────────────────
@router.post("/reset-password")
def reset_password(body: ResetRequest, db: Session = Depends(get_db)):
    if len(body.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    user = db.query(User).filter_by(email=body.email).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.otp != body.otp:
        raise HTTPException(400, "Invalid OTP")

    if user.otp_expires:
        expires = user.otp_expires
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires:
            raise HTTPException(400, "OTP expired")

    user.password_hash = hashpw(body.new_password.encode(), gensalt()).decode()
    user.otp           = None
    user.otp_expires   = None
    db.commit()

    logger.info(f"Password reset successful for {body.email}")
    return {"message": "Password reset successfully"}