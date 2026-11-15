"""Get, create, regenerate and list a user's weekly briefs.

One row per user per ISO week (weekly_briefs, unique on user_id + week_start). A GET reuses the
stored row, so the writer runs at most once per user per week unless the user regenerates.
"""
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import StudentProfile, User, WeeklyBrief
from backend.services.brief import facts as facts_module
from backend.services.brief import generator as generator_module

REGENERATE_LIMIT = 5
REGENERATE_WINDOW_SECONDS = 3600


class RegenerateLimiter:
    """At most `limit` regenerations per user per sliding window. In-memory, so per process."""

    def __init__(self, limit: int = REGENERATE_LIMIT, window_seconds: int = REGENERATE_WINDOW_SECONDS,
                 clock: Callable[[], float] = time.time):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._calls: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, user_id: str) -> int:
        """Seconds until another regenerate is allowed (0 when allowed now). Does not record a call."""
        with self._lock:
            now = self.clock()
            calls = self._calls[user_id]
            while calls and now - calls[0] >= self.window:
                calls.popleft()
            if len(calls) < self.limit:
                return 0
            return max(1, int(self.window - (now - calls[0])) + 1)

    def record(self, user_id: str) -> None:
        with self._lock:
            self._calls[user_id].append(self.clock())

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


regenerate_limiter = RegenerateLimiter()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _find(db: Session, user_id: str, week_start: date) -> Optional[WeeklyBrief]:
    return (
        db.query(WeeklyBrief)
        .filter(WeeklyBrief.user_id == user_id, WeeklyBrief.week_start == week_start)
        .first()
    )


def _build(db: Session, user: User, now: datetime):
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == user.id).first()
    facts = facts_module.build_facts(db, profile, now=now)
    # Looked up on the module at call time so tests can replace the writer.
    return generator_module.generate_brief(facts)


def _store(row: WeeklyBrief, result, now: datetime) -> None:
    row.facts = result.llm_facts
    row.content = result.content
    row.source = result.source
    row.model_used = result.model_used
    row.generated_at = now


def get_or_create_brief(db: Session, user: User, now: Optional[datetime] = None) -> WeeklyBrief:
    """This ISO week's brief for the user: the stored one, or a newly generated and stored one."""
    now = now or _now()
    week_start = facts_module.week_start_for(now)
    existing = _find(db, user.id, week_start)
    if existing is not None:
        return existing

    result = _build(db, user, now)
    row = WeeklyBrief(user_id=user.id, week_start=week_start)
    _store(row, result, now)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # A concurrent request created this week's row first; use theirs.
        db.rollback()
        winner = _find(db, user.id, week_start)
        if winner is None:
            raise
        return winner
    db.refresh(row)
    return row


def regenerate_brief(db: Session, user: User, now: Optional[datetime] = None) -> WeeklyBrief:
    """Replace this week's brief. The caller is responsible for rate limiting."""
    now = now or _now()
    week_start = facts_module.week_start_for(now)
    result = _build(db, user, now)
    row = _find(db, user.id, week_start)
    if row is None:
        row = WeeklyBrief(user_id=user.id, week_start=week_start)
        db.add(row)
    _store(row, result, now)
    db.commit()
    db.refresh(row)
    return row


def brief_history(db: Session, user: User, limit: int = 8) -> List[Dict[str, Any]]:
    rows = (
        db.query(WeeklyBrief)
        .filter(WeeklyBrief.user_id == user.id)
        .order_by(WeeklyBrief.week_start.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "week_start": r.week_start.isoformat(),
            "headline": (r.content or {}).get("headline"),
            "source": r.source,
        }
        for r in rows
    ]


def _links_by_skill(content: Dict[str, Any]) -> Dict[str, Dict[str, list]]:
    return {
        item["skill"]: {"learning": item.get("learning_links", []), "sources": item.get("sources", [])}
        for item in content.get("focus_skills", [])
    }


def brief_response(row: WeeklyBrief) -> Dict[str, Any]:
    """The API shape. Only the finished brief, its links and the basis string; never the raw facts,
    article bodies or evidence quotes."""
    content = row.content or {}
    return {
        "week_start": row.week_start.isoformat(),
        "generated_at": row.generated_at.isoformat() if row.generated_at else None,
        "source": row.source,
        "content": content,
        "links": _links_by_skill(content),
        "basis": (row.facts or {}).get("basis"),
    }
