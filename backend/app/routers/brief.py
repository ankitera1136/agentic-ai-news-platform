"""CareerRadar weekly brief API (signed-in users only; each user sees only their own briefs).

GET  /api/career/brief             this week's brief (generated and stored on first request)
POST /api/career/brief/regenerate  replace this week's brief (at most 5 per hour per user)
GET  /api/career/brief/history     the user's past briefs: week, headline, source
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.auth import get_current_active_user
from backend.app.database import get_db
from backend.app.models import User
from backend.services.brief import service

router = APIRouter(prefix="/api/career/brief", tags=["careerradar-brief"])


@router.get("")
def get_brief(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """The current ISO week's brief. A user with no profile gets a market-only brief."""
    return service.brief_response(service.get_or_create_brief(db, current_user))


@router.post("/regenerate")
def regenerate(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    wait = service.regenerate_limiter.retry_after(current_user.id)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"You can regenerate your brief at most {service.REGENERATE_LIMIT} times per hour. "
                   f"Try again in about {max(1, wait // 60)} minute(s).",
            headers={"Retry-After": str(wait)},
        )
    service.regenerate_limiter.record(current_user.id)
    return service.brief_response(service.regenerate_brief(db, current_user))


@router.get("/history")
def get_history(
    limit: int = Query(8, ge=1, le=26, description="Number of past briefs"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    return {"briefs": service.brief_history(db, current_user, limit=limit)}
