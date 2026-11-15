"""CareerRadar API: skill demand and the signed-in student's skill gap.

GET /api/career/skill-demand   public; ranked skills from hiring/expansion/new-office signals
GET /api/career/my-gap         requires login; compares the user's profile to that demand
GET /api/career/company-board  public; per-company hiring/expansion vs layoff/freeze signal counts
GET /api/career/heatmap        public; signals aggregated by city with coordinates from a static table
GET /api/career/trend          public; weekly growth vs cuts signal counts

Responses are built from stored career signals only. article_bodies.body_text is never read.
"""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.auth import get_current_active_user
from backend.app.database import get_db
from backend.app.models import StudentProfile, User
from backend.services.career import market, matching

router = APIRouter(prefix="/api/career", tags=["careerradar"])


@router.get("/skill-demand")
def get_skill_demand(
    days: int = Query(30, ge=1, le=90, description="Look-back window in days"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of skills returned"),
    db: Session = Depends(get_db),
):
    """Public ranked list of skills mentioned in recent hiring, expansion and new-office news."""
    return matching.skill_demand(db, days=days, limit=limit)


@router.get("/my-gap")
def get_my_gap(
    days: int = Query(30, ge=1, le=90, description="Look-back window in days"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Matched and missing skills plus relevant signals for the current user's profile.

    A user with no profile gets 200 with empty_profile=true and the top demand list.
    """
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    return matching.gap_for_profile(db, profile, days=days)


COUNTRY_QUERY = Query(
    None, max_length=60,
    description="Country filter, for example India. Omit (or send Global) for all countries.",
)


@router.get("/company-board")
def get_company_board(
    days: int = Query(30, ge=1, le=90, description="Look-back window in days"),
    country: Optional[str] = COUNTRY_QUERY,
    limit: int = Query(30, ge=1, le=50, description="Maximum number of companies returned"),
    db: Session = Depends(get_db),
):
    """Public per-company counts of growth signals (hiring, expansion, new office) versus cuts."""
    return market.company_board(db, days=days, country=country, limit=limit)


@router.get("/heatmap")
def get_heatmap(
    days: int = Query(30, ge=1, le=90, description="Look-back window in days"),
    country: Optional[str] = COUNTRY_QUERY,
    group: Literal["growth", "cuts", "all"] = Query("all", description="growth, cuts or all"),
    db: Session = Depends(get_db),
):
    """Public city-level aggregation. Only places in the built-in coordinate table are plotted."""
    return market.heatmap_points(db, days=days, country=country, action_group=group)


@router.get("/trend")
def get_trend(
    weeks: int = Query(8, ge=1, le=26, description="Number of ISO weeks, ending with this week"),
    country: Optional[str] = COUNTRY_QUERY,
    db: Session = Depends(get_db),
):
    """Public weekly counts of growth versus cuts signals, with empty weeks filled with zero."""
    return market.weekly_trend(db, weeks=weeks, country=country)
