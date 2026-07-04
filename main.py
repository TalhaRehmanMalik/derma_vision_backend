"""
main.py — Derma Vision FastAPI Backend
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import logging

# Suppress TensorFlow noise before any TF import — must be first
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # show ERROR only, hide INFO/WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN op messages
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger
logger = get_logger("main")

# ── Database ──────────────────────────────────────────────────
from database import create_database_if_not_exists, engine, Base, get_db
import models  # registers all ORM models with Base — do not remove

create_database_if_not_exists()
Base.metadata.create_all(bind=engine)
logger.info("All tables created/verified.")

# ── ML model — loaded once at startup, reused for every request
from utils.inference import load_model

MODEL_PATH   = os.path.join("ml_models", "derma_vision_mobilenetv2.h5")
MAPPING_PATH = os.path.join("ml_models", "class_mapping.json")
load_model(MODEL_PATH, MAPPING_PATH)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "Derma Vision API",
    description = "AI-powered skin cancer classification system",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow React dev servers ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Serve uploaded scan images as static files ────────────────
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# ── Register routers ─────────────────────────────────────────
from routes.auth    import router as auth_router
from routes.predict import router as predict_router
from routes.chat    import router as chat_router
from routes.admin   import router as admin_router

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(chat_router)
app.include_router(admin_router)

logger.info("All routes registered.")


# ── Health check ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":    "Derma Vision API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "http://localhost:8000/docs",
    }


# ── Promote a user to admin — requires caller to be admin ────
# BUG FIX: original endpoint had no auth — anyone could call it
from utils.auth import get_current_user_id, require_admin

@app.post("/make-admin/{target_user_id}", tags=["Admin"])
def make_admin(
    target_user_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Promote a user to admin. Only an existing admin can call this.

    First-time setup (no admin exists yet) — run this SQL once:
        UPDATE users SET is_admin=1 WHERE id=1;
    After that use this endpoint for all future promotions.
    """
    require_admin(user_id, db)  # 403 if caller is not admin

    target = db.query(models.User).filter_by(id=target_user_id).first()
    if not target:
        logger.warning(f"make-admin: user id={target_user_id} not found")
        return {"error": "User not found"}

    target.is_admin = True
    db.commit()
    logger.info(f"User '{target.username}' (id={target_user_id}) promoted to admin by user_id={user_id}")
    return {"message": f"User '{target.username}' is now admin"}