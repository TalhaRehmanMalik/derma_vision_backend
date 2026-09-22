"""
models.py
Database tables — Users aur Scans.
Yeh tables automatically create honge startup pe.
"""
import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(80),  unique=False, nullable=False)
    email         = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    is_verified   = Column(Boolean, default=False)
    otp           = Column(String(6),   nullable=True)
    otp_expires   = Column(DateTime,    nullable=True)
    is_admin      = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scans = relationship("Scan", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":          self.id,
            "username":    self.username,
            "email":       self.email,
            "is_verified": self.is_verified,
            "is_admin":    self.is_admin,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
        }


class Scan(Base):
    __tablename__ = "scans"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    user_id           = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_filename    = Column(String(256), nullable=False)
    predicted_class   = Column(String(100), nullable=False)
    confidence        = Column(Float, nullable=False)
    all_probabilities = Column(Text, nullable=True)   # JSON string
    timestamp         = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="scans")

    def to_dict(self):
        return {
            "id":               self.id,
            "user_id":          self.user_id,
            "image_filename":   self.image_filename,
            "predicted_class":  self.predicted_class,
            "confidence":       round(self.confidence, 4),
            "all_probabilities": json.loads(self.all_probabilities) if self.all_probabilities else {},
            "timestamp":        self.timestamp.isoformat() if self.timestamp else None,
        }
