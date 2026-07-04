"""
routes/predict.py
Skin image upload, ML inference, and scan history.

POST   /api/predict       — Upload image → AI diagnosis (JWT required)
GET    /api/scans         — Get current user's scan history (JWT required)
DELETE /api/scans/{id}    — Delete own scan by id (JWT required)
"""
import json
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from database import get_db
from models import Scan
from utils.auth import get_current_user_id
from utils.inference import predict, save_image, allowed_file
from utils.logger import get_logger

logger = get_logger("routes.predict")
router = APIRouter(prefix="/api", tags=["Predict"])

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")


# ── POST /api/predict ─────────────────────────────────────────
@router.post("/predict")
async def predict_image(
    image: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    # Validate file type before reading content
    if not allowed_file(image.filename):
        raise HTTPException(400, "Only JPG and PNG images are accepted")

    contents = await image.read()

    # Reject files over 10 MB
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 10MB")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename   = save_image(contents, image.filename, UPLOAD_FOLDER)
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        result = predict(image_path)
    except RuntimeError as e:
        # Clean up saved file if inference fails
        os.remove(image_path)
        logger.error(f"Inference failed: {e}")
        raise HTTPException(503, str(e))

    # Pick disclaimer based on confidence level
    disclaimer = (
        "Confidence is below threshold. Result is inconclusive — "
        "please consult a dermatologist for proper diagnosis."
        if result["inconclusive"] else
        "This is an AI-assisted screening tool only. "
        "It does NOT replace a professional medical diagnosis. "
        "Please consult a dermatologist."
    )

    # Persist scan to database
    scan = Scan(
        user_id           = user_id,
        image_filename    = filename,
        predicted_class   = result["predicted_class"],
        confidence        = result["confidence"],
        all_probabilities = json.dumps(result["all_probabilities"]),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    logger.info(f"Scan saved: id={scan.id} | user={user_id} | class={result['predicted_class']} | conf={result['confidence']:.2%}")

    return {
        "scan_id":           scan.id,
        "predicted_class":   result["predicted_class"],
        "confidence":        result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "inconclusive":      result["inconclusive"],
        "disclaimer":        disclaimer,
    }


# ── GET /api/scans ────────────────────────────────────────────
@router.get("/scans")
def get_my_scans(
    page: int = 1,
    limit: int = 10,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit
    total  = db.query(Scan).filter_by(user_id=user_id).count()
    scans  = (
        db.query(Scan)
        .filter_by(user_id=user_id)
        .order_by(Scan.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "scans": [s.to_dict() for s in scans],
        "total": total,
        "page":  page,
        "pages": (total + limit - 1) // limit,
    }


# ── DELETE /api/scans/{scan_id} ───────────────────────────────
@router.delete("/scans/{scan_id}")
def delete_scan(
    scan_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    # Filter by both scan_id and user_id — users can only delete their own scans
    scan = db.query(Scan).filter_by(id=scan_id, user_id=user_id).first()
    if not scan:
        raise HTTPException(404, "Scan not found")

    img_path = os.path.join(UPLOAD_FOLDER, scan.image_filename)
    if os.path.exists(img_path):
        os.remove(img_path)

    db.delete(scan)
    db.commit()
    logger.info(f"Scan deleted: id={scan_id} by user_id={user_id}")
    return {"message": f"Scan {scan_id} deleted successfully"}