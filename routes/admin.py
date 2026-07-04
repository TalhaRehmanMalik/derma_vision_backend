"""
routes/admin.py
Admin-only endpoints. All routes require a valid admin JWT.

GET    /api/admin/stats          — System stats (users, scans, class distribution)
GET    /api/admin/users          — All users with scan counts
GET    /api/admin/scans          — All scans, paginated
DELETE /api/admin/scans/{id}     — Delete any scan + its image file
DELETE /api/admin/users/{id}     — Delete user + all their scans and images
"""
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from models import User, Scan
from utils.auth import get_current_user_id, require_admin  # BUG FIX: was duplicated here, now imported
from utils.logger import get_logger

logger = get_logger("routes.admin")
router = APIRouter(prefix="/api/admin", tags=["Admin"])

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")


# ── GET /api/admin/stats ──────────────────────────────────────
@router.get("/stats")
def get_stats(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    require_admin(user_id, db)

    total_users = db.query(User).count()
    total_scans = db.query(Scan).count()

    # Count scans per predicted class
    dist = (
        db.query(Scan.predicted_class, func.count(Scan.id))
        .group_by(Scan.predicted_class).all()
    )
    class_distribution = {cls: cnt for cls, cnt in dist}

    # Average confidence per class
    conf = (
        db.query(Scan.predicted_class, func.avg(Scan.confidence))
        .group_by(Scan.predicted_class).all()
    )
    avg_confidence = {cls: round(float(avg), 4) for cls, avg in conf}

    verified   = db.query(User).filter_by(is_verified=True).count()
    unverified = total_users - verified

    logger.info(f"Stats fetched by admin user_id={user_id}")
    return {
        "total_users":        total_users,
        "total_scans":        total_scans,
        "verified_users":     verified,
        "unverified_users":   unverified,
        "class_distribution": class_distribution,
        "avg_confidence":     avg_confidence,
    }


# ── GET /api/admin/users ──────────────────────────────────────
@router.get("/users")
def get_all_users(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    require_admin(user_id, db)

    users  = db.query(User).order_by(User.created_at.desc()).all()
    result = []
    for u in users:
        d = u.to_dict()
        d["scan_count"] = db.query(Scan).filter_by(user_id=u.id).count()
        result.append(d)

    logger.info(f"All users fetched by admin user_id={user_id} (total={len(result)})")
    return {"users": result, "total": len(result)}


# ── GET /api/admin/scans ──────────────────────────────────────
@router.get("/scans")
def get_all_scans(
    page: int = 1,
    limit: int = 20,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    require_admin(user_id, db)

    offset = (page - 1) * limit
    total  = db.query(Scan).count()
    scans  = (
        db.query(Scan)
        .order_by(Scan.timestamp.desc())
        .offset(offset).limit(limit).all()
    )

    return {
        "scans": [s.to_dict() for s in scans],
        "total": total,
        "page":  page,
        "pages": (total + limit - 1) // limit,
    }


# ── DELETE /api/admin/scans/{scan_id} ────────────────────────
@router.delete("/scans/{scan_id}")
def delete_scan(
    scan_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    require_admin(user_id, db)

    scan = db.query(Scan).filter_by(id=scan_id).first()
    if not scan:
        raise HTTPException(404, "Scan not found")

    # Delete image file from disk before removing DB record
    img_path = os.path.join(UPLOAD_FOLDER, scan.image_filename)
    if os.path.exists(img_path):
        os.remove(img_path)

    db.delete(scan)
    db.commit()
    logger.info(f"Scan id={scan_id} deleted by admin user_id={user_id}")
    return {"message": f"Scan {scan_id} deleted"}


# ── DELETE /api/admin/users/{target_user_id} ─────────────────
@router.delete("/users/{target_user_id}")
def delete_user(
    target_user_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    require_admin(user_id, db)

    target = db.query(User).filter_by(id=target_user_id).first()
    if not target:
        raise HTTPException(404, "User not found")

    # Remove all scan image files before deleting the user
    for scan in target.scans:
        img_path = os.path.join(UPLOAD_FOLDER, scan.image_filename)
        if os.path.exists(img_path):
            os.remove(img_path)

    # cascade="all, delete-orphan" on User.scans handles DB rows automatically
    db.delete(target)
    db.commit()
    logger.info(f"User id={target_user_id} and all data deleted by admin user_id={user_id}")
    return {"message": f"User {target_user_id} and all their data deleted"}