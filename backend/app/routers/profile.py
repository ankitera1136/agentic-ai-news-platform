"""Student career profile API.

GET    /api/profile   current user's profile, or an empty default (no row is created)
PUT    /api/profile   full upsert of the current user's profile
DELETE /api/profile   delete the profile row

Only the profile fields are ever returned. No email, password hash or other user data.
"""
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from backend.app.auth import get_current_active_user
from backend.app.database import get_db
from backend.app.models import StudentProfile, User
from backend.services.career.normalizer import normalize_company
from backend.services.profile.skills import canonicalize_skill
from backend.services.profile.text import clean_text

router = APIRouter(prefix="/api/profile", tags=["profile"])

# The app-wide limiter lives in webapp.py, which imports this module, so the router keeps
# its own in-memory limiter. The RateLimitExceeded handler registered on the app serves both.
limiter = Limiter(key_func=get_remote_address)

MAX_SKILLS, MAX_SKILL_LEN = 50, 40
MAX_COMPANIES, MAX_COMPANY_LEN = 20, 100
MAX_ROLES, MAX_ROLE_LEN = 10, 60
MAX_BRANCH_LEN, MAX_COLLEGE_LEN = 100, 150


def _clean_string_list(
    value: Any,
    *,
    label: str,
    max_items: int,
    max_len: int,
    canonicalize: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Trim, drop empties, canonicalize, dedupe case-insensitively, then enforce limits."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of strings")
    seen = set()
    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} must contain only strings")
        text = clean_text(item)
        if not text:
            continue
        if len(text) > max_len:
            raise ValueError(f"{label}: each item must be at most {max_len} characters")
        if canonicalize is not None:
            text = canonicalize(text)
            if not text:
                continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if len(cleaned) > max_items:
        raise ValueError(f"{label}: at most {max_items} items are allowed")
    return cleaned


def _canonical_company(text: str) -> str:
    return normalize_company(text).canonical


def _clean_optional_text(value: Any, label: str, max_len: int) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = clean_text(value)
    if not text:
        return None
    if len(text) > max_len:
        raise ValueError(f"{label} must be at most {max_len} characters")
    return text


class ProfileUpdate(BaseModel):
    """Request body for PUT /api/profile. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    branch: Optional[str] = None
    college: Optional[str] = None
    year_of_study: Optional[StrictInt] = Field(default=None, ge=1, le=6)
    graduation_year: Optional[StrictInt] = Field(default=None, ge=2020, le=2040)
    skills: List[str] = Field(default_factory=list)
    target_companies: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)

    @field_validator("branch", mode="before")
    @classmethod
    def _branch(cls, value):
        return _clean_optional_text(value, "branch", MAX_BRANCH_LEN)

    @field_validator("college", mode="before")
    @classmethod
    def _college(cls, value):
        return _clean_optional_text(value, "college", MAX_COLLEGE_LEN)

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, value):
        return _clean_string_list(
            value, label="skills", max_items=MAX_SKILLS, max_len=MAX_SKILL_LEN,
            canonicalize=canonicalize_skill,
        )

    @field_validator("target_companies", mode="before")
    @classmethod
    def _companies(cls, value):
        return _clean_string_list(
            value, label="target_companies", max_items=MAX_COMPANIES, max_len=MAX_COMPANY_LEN,
            canonicalize=_canonical_company,
        )

    @field_validator("target_roles", mode="before")
    @classmethod
    def _roles(cls, value):
        return _clean_string_list(
            value, label="target_roles", max_items=MAX_ROLES, max_len=MAX_ROLE_LEN,
        )


class ProfileResponse(BaseModel):
    """Profile fields only."""
    model_config = ConfigDict(from_attributes=True)

    branch: Optional[str] = None
    college: Optional[str] = None
    year_of_study: Optional[int] = None
    graduation_year: Optional[int] = None
    skills: List[str] = Field(default_factory=list)
    target_companies: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    updated_at: Optional[datetime] = None


def _get_row(db: Session, user_id: str) -> Optional[StudentProfile]:
    return db.query(StudentProfile).filter(StudentProfile.user_id == user_id).first()


@router.get("", response_model=ProfileResponse)
def get_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Return the current user's profile, or an empty default. Never creates a row."""
    row = _get_row(db, current_user.id)
    if row is None:
        return ProfileResponse()
    return ProfileResponse.model_validate(row)


@router.put("", response_model=ProfileResponse)
@limiter.limit("30/minute")
def put_profile(
    request: Request,
    payload: ProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Replace the current user's whole profile (creates the row on first save)."""
    row = _get_row(db, current_user.id)
    if row is None:
        row = StudentProfile(user_id=current_user.id)
        db.add(row)
    row.branch = payload.branch
    row.college = payload.college
    row.year_of_study = payload.year_of_study
    row.graduation_year = payload.graduation_year
    row.skills = list(payload.skills)
    row.target_companies = list(payload.target_companies)
    row.target_roles = list(payload.target_roles)
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return ProfileResponse.model_validate(row)


@router.delete("", status_code=204)
def delete_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete the current user's profile row. Idempotent: 204 even if none exists."""
    row = _get_row(db, current_user.id)
    if row is not None:
        db.delete(row)
        db.commit()
    return Response(status_code=204)
