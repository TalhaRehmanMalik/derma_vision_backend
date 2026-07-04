
"""
utils/auth.py
JWT creation, verification, and admin guard.
All routes import from here — no auth logic lives in route files.
"""
import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger("auth")

SECRET    = os.getenv("JWT_SECRET", "derma_vision_secret")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_H  = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# Extracts Bearer token from Authorization header
bearer = HTTPBearer()


def create_token(user_id: int) -> str:
    """Create a signed JWT that encodes the user's id."""
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=EXPIRE_H),
    }
    token = jwt.encode(payload, SECRET, algorithm=ALGORITHM)
    logger.info(f"Token created for user_id={user_id}")
    return token


def decode_token(token: str) -> int:
    """Decode and validate a JWT. Raises 401 if invalid or expired."""
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except JWTError:
        logger.warning("Invalid or expired token received")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


def get_current_user_id(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> int:
    """FastAPI dependency — injects authenticated user_id into route functions."""
    return decode_token(creds.credentials)


def require_admin(user_id: int, db: Session):
    """
    Guard for admin-only routes.
    Raises 403 if the user does not exist or is not an admin.
    Kept here (not in admin.py) so any future route can reuse it.

    BUG FIX: was duplicated inside admin.py — now single source of truth.
    """
    from models import User  # local import avoids circular dependency
    user = db.query(User).filter_by(id=user_id).first()
    if not user or not user.is_admin:
        logger.warning(f"Unauthorized admin access attempt by user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user