"""Skill demand and skill-gap matching built only from stored CareerRadar signals.

All aggregation happens in Python (SQLite and PostgreSQL differ on JSON queries). Nothing here
calls an LLM, reads article bodies, or invents data: every number is a count of stored signals.

Pipeline:
1. Load HIRING / EXPANSION / NEW_OFFICE signals with timing current or planned.
2. Keep those whose article date falls inside the requested window.
3. Collapse signals with the same company + action + ISO week into one (across articles).
4. Count skills and match them against a student profile.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.app.models import Article, CareerSignal
from backend.services.career.normalizer import normalize_company
from backend.services.profile.skills import canonicalize_skill
from backend.services.profile.text import clean_text

DEMAND_ACTIONS = ("HIRING", "EXPANSION", "NEW_OFFICE")
DEMAND_TIMINGS = ("current", "planned")
MAX_COMPANIES_PER_SKILL = 5
MAX_QUOTE_CHARS = 200
MAX_RELEVANT_SIGNALS = 50

NOTE = (
    "Skill gaps are inferred from skills mentioned in news articles about hiring, expansion and "
    "new offices, not from job postings. Treat them as hints, not a survey of the job market."
)


@dataclass
class DedupedSignal:
    """One signal after the company + action + ISO-week collapse."""
    id: int
    company: str
    action: str
    timing: str
    count: Optional[int]
    location_city: Optional[str]
    location_country: Optional[str]
    skills: List[str]
    confidence: float
    evidence_quote: str
    event_date: datetime
    signal_ids: List[int] = field(default_factory=list)
    articles: List[Dict[str, Optional[str]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "company": self.company,
            "action": self.action,
            "timing": self.timing,
            "count": self.count,
            "location": {"city": self.location_city, "country": self.location_country},
            "skills": list(self.skills),
            "confidence": self.confidence,
            "evidence_quote": _truncate(self.evidence_quote, MAX_QUOTE_CHARS),
            "date": self.event_date.date().isoformat(),
            "articles": [_public_article_ref(a) for a in self.articles],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: Optional[str], limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "\u2026"


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _event_date(article: Article, signal: CareerSignal) -> datetime:
    """Best date for when the news happened: published header, else scrape time, else extraction time."""
    if article.published:
        try:
            parsed = _as_utc(parsedate_to_datetime(article.published))
            if parsed is not None:
                return parsed
        except (TypeError, ValueError, IndexError):
            pass
    return (
        _as_utc(article.created_at)
        or _as_utc(signal.extracted_at)
        or datetime.now(timezone.utc)
    )


def clean_skill_list(skills: Optional[Iterable[Any]]) -> List[str]:
    """Clean, alias-normalize and case-insensitively dedupe skills, keeping first-seen order."""
    seen = set()
    out: List[str] = []
    for item in skills or []:
        if not isinstance(item, str):
            continue
        text = clean_text(item)
        if not text:
            continue
        canonical = canonicalize_skill(text)
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _company_key(name: str) -> str:
    canonical = normalize_company(name).canonical or (name or "")
    return canonical.casefold()


def _article_ref(article: Article) -> Dict[str, Optional[str]]:
    return {"id": article.id, "title": article.title, "link": article.link, "source": article.source}


def _public_article_ref(ref: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {"title": ref.get("title"), "link": ref.get("link"), "source": ref.get("source")}


# ---------------------------------------------------------------------------
# Loading and dedupe
# ---------------------------------------------------------------------------

def load_demand_signals(
    db: Session,
    days: int = 30,
    now: Optional[datetime] = None,
) -> List[DedupedSignal]:
    """Eligible signals inside the window, deduped by company + action + ISO week."""
    now_utc = _as_utc(now) or datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=days)

    rows = (
        db.query(CareerSignal, Article)
        .join(Article, Article.id == CareerSignal.source_article_id)
        .filter(CareerSignal.action.in_(DEMAND_ACTIONS), CareerSignal.timing.in_(DEMAND_TIMINGS))
        .all()
    )

    groups: Dict[Tuple[str, str, int, int], List[Tuple[CareerSignal, Article, datetime]]] = {}
    for signal, article in rows:
        event = _event_date(article, signal)
        if event < cutoff or event > now_utc + timedelta(days=1):
            continue
        iso_year, iso_week, _ = event.isocalendar()
        key = (_company_key(signal.company), signal.action, iso_year, iso_week)
        groups.setdefault(key, []).append((signal, article, event))

    deduped: List[DedupedSignal] = []
    for members in groups.values():
        # Highest confidence wins; ties go to the newest event, then the lowest id (deterministic).
        members.sort(key=lambda m: (-(m[0].confidence or 0.0), -m[2].timestamp(), m[0].id))
        best, best_article, best_event = members[0]

        merged_skills = clean_skill_list(
            [skill for member, _, _ in members for skill in (member.skills_mentioned or [])]
        )
        articles: List[Dict[str, Optional[str]]] = []
        seen_articles = set()
        for _, article, _ in members:
            if article.id in seen_articles:
                continue
            seen_articles.add(article.id)
            articles.append(_article_ref(article))

        company = normalize_company(best.company).canonical or best.company
        deduped.append(
            DedupedSignal(
                id=best.id,
                company=company,
                action=best.action,
                timing=best.timing,
                count=best.count,
                location_city=best.location_city,
                location_country=best.location_country,
                skills=merged_skills,
                confidence=float(best.confidence or 0.0),
                evidence_quote=best.evidence_quote or "",
                event_date=best_event,
                signal_ids=[member.id for member, _, _ in members],
                articles=articles,
            )
        )

    deduped.sort(key=lambda s: (-s.event_date.timestamp(), -s.confidence, s.id))
    return deduped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class _SkillStat:
    skill: str
    signals: set = field(default_factory=set)
    companies: Dict[str, str] = field(default_factory=dict)  # casefolded key -> display name

    def as_dict(self, include_companies: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "skill": self.skill,
            "company_count": len(self.companies),
            "signal_count": len(self.signals),
        }
        if include_companies:
            names = sorted(self.companies.values(), key=str.casefold)
            out["companies"] = names[:MAX_COMPANIES_PER_SKILL]
        return out


def _rank_skills(signals: Sequence[DedupedSignal]) -> List[_SkillStat]:
    stats: Dict[str, _SkillStat] = {}
    for signal in signals:
        for skill in signal.skills:
            stat = stats.setdefault(skill.casefold(), _SkillStat(skill=skill))
            stat.signals.add(signal.id)
            stat.companies.setdefault(signal.company.casefold(), signal.company)
    return sorted(
        stats.values(),
        key=lambda s: (-len(s.companies), -len(s.signals), s.skill.casefold()),
    )


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def build_basis(signals: Sequence[DedupedSignal], days: int) -> str:
    """Plain-language description of exactly what the numbers are based on."""
    window = f"the last {days} day" if days == 1 else f"the last {days} days"
    if not signals:
        return f"No hiring, expansion or new-office signals were found in {window}."
    article_ids = {ref["id"] for s in signals for ref in s.articles}
    text = f"Based on {_plural(len(signals), 'signal')} from {_plural(len(article_ids), 'article')} in {window}."
    with_skills = sum(1 for s in signals if s.skills)
    if with_skills < len(signals):
        verb = "names" if with_skills == 1 else "name"
        text += f" {with_skills} of them {verb} specific skills."
    return text


def _signal_counts(signals: Sequence[DedupedSignal]) -> Dict[str, int]:
    return {
        "signals": len(signals),
        "articles": len({ref["id"] for s in signals for ref in s.articles}),
        "signals_with_skills": sum(1 for s in signals if s.skills),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def skill_demand(
    db: Session,
    days: int = 30,
    limit: int = 20,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Ranked skills mentioned by hiring / expansion / new-office signals."""
    signals = load_demand_signals(db, days=days, now=now)
    ranked = _rank_skills(signals)
    return {
        "days": days,
        "skills": [stat.as_dict() for stat in ranked[:limit]],
        **_signal_counts(signals),
        "basis": build_basis(signals, days),
        "note": NOTE,
    }


def gap_for_profile(
    db: Session,
    profile: Any,
    days: int = 30,
    missing_limit: int = 10,
    demand_limit: int = 20,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Compare a student profile (anything with .skills and .target_companies) to current demand."""
    signals = load_demand_signals(db, days=days, now=now)
    ranked = _rank_skills(signals)
    base = {
        "days": days,
        **_signal_counts(signals),
        "basis": build_basis(signals, days),
        "note": NOTE,
    }

    user_skills = clean_skill_list(getattr(profile, "skills", None) or [])
    target_keys = {
        _company_key(name)
        for name in (getattr(profile, "target_companies", None) or [])
        if isinstance(name, str) and name.strip()
    }

    if profile is None or not (user_skills or target_keys):
        return {
            **base,
            "empty_profile": True,
            "matched_skills": [],
            "missing_skills": [],
            "relevant_signals": [],
            "top_demand": [stat.as_dict() for stat in ranked[:demand_limit]],
        }

    user_keys = {skill.casefold() for skill in user_skills}

    matched = [stat.as_dict(include_companies=False) for stat in ranked if stat.skill.casefold() in user_keys]
    missing = [stat.as_dict() for stat in ranked if stat.skill.casefold() not in user_keys][:missing_limit]

    relevant: List[Tuple[Tuple, Dict[str, Any]]] = []
    for signal in signals:
        company_match = signal.company.casefold() in target_keys
        matched_skills = [s for s in signal.skills if s.casefold() in user_keys]
        if not (company_match or matched_skills):
            continue
        missing_skills = [s for s in signal.skills if s.casefold() not in user_keys]
        item = signal.to_dict()
        item["matches_target_company"] = company_match
        item["matched_skills"] = matched_skills
        item["missing_skills"] = missing_skills
        sort_key = (
            0 if company_match else 1,
            -len(matched_skills),
            -signal.confidence,
            -signal.event_date.timestamp(),
            signal.id,
        )
        relevant.append((sort_key, item))
    relevant.sort(key=lambda pair: pair[0])

    return {
        **base,
        "empty_profile": False,
        "matched_skills": matched,
        "missing_skills": missing,
        "relevant_signals": [item for _, item in relevant[:MAX_RELEVANT_SIGNALS]],
    }
