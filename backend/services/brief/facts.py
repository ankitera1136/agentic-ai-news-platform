"""Deterministic facts for a user's weekly brief. No LLM, no network.

Every value comes from stored CareerRadar signals via matching.py and market.py. The result has
two views:
- the full facts (includes source article links and learning links), used by code to attach
  links to the finished brief;
- llm_view(facts), the same facts with every URL and article title removed. That is the only
  thing the writer ever sees, so it cannot copy a URL and cannot be steered by article text.
"""
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.services.brief.resources import links_for
from backend.services.career import market, matching

DEFAULT_WINDOW_DAYS = 7
FALLBACK_WINDOW_DAYS = 30
MIN_SIGNALS_FOR_SHORT_WINDOW = 5
MAX_MISSING_SKILLS = 3
MAX_MATCHED_SKILLS = 5
MAX_RELEVANT_SIGNALS = 3
MAX_COMPANIES_PER_SKILL = 3
MAX_SOURCES_PER_SKILL = 2
MAX_MARKET_COMPANIES = 3

# Keys removed from the writer's view: they hold URLs or untrusted article titles.
_LLM_HIDDEN_KEYS = {"sources", "source", "learning_links", "url", "link", "title"}


def week_start_for(moment: datetime) -> date:
    """Monday of the ISO week containing `moment` (UTC)."""
    moment = moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return (moment - timedelta(days=moment.weekday())).date()


def llm_view(value: Any) -> Any:
    """Copy of the facts without URLs, article titles or source references."""
    if isinstance(value, dict):
        return {k: llm_view(v) for k, v in value.items() if k not in _LLM_HIDDEN_KEYS}
    if isinstance(value, list):
        return [llm_view(v) for v in value]
    return value


def _place(location: Dict[str, Optional[str]]) -> Optional[str]:
    parts = [p for p in ((location or {}).get("city"), (location or {}).get("country")) if p]
    return ", ".join(parts) or None


def _skill_item(
    entry: Dict[str, Any],
    demand_signals: List[matching.DedupedSignal],
) -> Dict[str, Any]:
    """One demanded skill with its counts, up to 3 companies, up to 2 source articles, and links."""
    skill = entry["skill"]
    sources: List[Dict[str, Optional[str]]] = []
    seen = set()
    for signal in demand_signals:                       # newest first
        if skill.casefold() not in {s.casefold() for s in signal.skills}:
            continue
        for ref in signal.articles:
            if ref["id"] in seen:
                continue
            seen.add(ref["id"])
            sources.append({"title": ref.get("title"), "link": ref.get("link")})
            if len(sources) >= MAX_SOURCES_PER_SKILL:
                break
        if len(sources) >= MAX_SOURCES_PER_SKILL:
            break
    return {
        "skill": skill,
        "demand_signals": entry["signal_count"],
        "company_count": entry["company_count"],
        "companies": list(entry.get("companies", []))[:MAX_COMPANIES_PER_SKILL],
        "sources": sources,
        "learning_links": links_for(skill),
    }


def _market_snapshot(board: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    companies = board["companies"]
    growing = sorted(
        (c for c in companies if c["net_signal"] == "growing"),
        key=lambda c: (-c["growth"], c["company"].casefold()),
    )[:MAX_MARKET_COMPANIES]
    cutting = sorted(
        (c for c in companies if c["net_signal"] == "cutting"),
        key=lambda c: (-c["cuts"], c["company"].casefold()),
    )[:MAX_MARKET_COMPANIES]

    def headcount(stat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Only reported when at least one signal stated a number, and always with "from X of Y".
        if stat.get("total") is None:
            return None
        return {
            "total": stat["total"],
            "signals_with_count": stat["signals_with_count"],
            "signals": stat["signals"],
        }

    return {
        "growing": [
            {
                "company": c["company"],
                "growth_signals": c["growth"],
                "cut_signals": c["cuts"],
                "reported_hires": headcount(c["reported_hires"]),
                "sources": [{"title": a["title"], "link": a["link"]} for a in c["articles"][:1]],
            }
            for c in growing
        ],
        "cutting": [
            {
                "company": c["company"],
                "growth_signals": c["growth"],
                "cut_signals": c["cuts"],
                "reported_layoffs": headcount(c["reported_layoffs"]),
                "sources": [{"title": a["title"], "link": a["link"]} for a in c["articles"][:1]],
            }
            for c in cutting
        ],
    }


def choose_window(db: Session, now: datetime) -> Dict[str, Any]:
    """7 days when it has enough signals, otherwise 30 days (and say so)."""
    short = market._load_market_signals(db, now - timedelta(days=DEFAULT_WINDOW_DAYS), now)
    if len(short) >= MIN_SIGNALS_FOR_SHORT_WINDOW:
        return {"days": DEFAULT_WINDOW_DAYS, "fallback": False, "short_window_signals": len(short)}
    return {"days": FALLBACK_WINDOW_DAYS, "fallback": True, "short_window_signals": len(short)}


def build_facts(
    db: Session,
    profile: Any,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Structured facts for one user. `profile` is a StudentProfile (or anything with .skills and
    .target_companies), or None."""
    now_utc = now.astimezone(timezone.utc) if now and now.tzinfo else (now.replace(tzinfo=timezone.utc) if now else datetime.now(timezone.utc))
    window = choose_window(db, now_utc)
    days = window["days"]

    gap = matching.gap_for_profile(db, profile, days=days, missing_limit=MAX_MISSING_SKILLS, now=now_utc)
    demand_signals = matching.load_demand_signals(db, days=days, now=now_utc)
    board = market.company_board(db, days=days, limit=50, now=now_utc)

    basis = gap["basis"]
    if window["fallback"]:
        basis += (
            f" The last {DEFAULT_WINDOW_DAYS} days had only {window['short_window_signals']} "
            f"hiring or layoff signals, so this brief uses the last {FALLBACK_WINDOW_DAYS} days."
        )

    facts: Dict[str, Any] = {
        "week_start": week_start_for(now_utc).isoformat(),
        "window_days": days,
        "window_fallback": window["fallback"],
        "basis": basis,
        "note": gap["note"],
        "empty_profile": bool(gap["empty_profile"]),
        "totals": {"signals": gap["signals"], "articles": gap["articles"]},
        "market": _market_snapshot(board),
    }

    if gap["empty_profile"]:
        facts["demand_skills"] = [
            _skill_item(entry, demand_signals) for entry in gap["top_demand"][:MAX_MISSING_SKILLS]
        ]
        return facts

    facts["profile"] = {
        "skills": matching.clean_skill_list(getattr(profile, "skills", None) or []),
        "target_companies": [
            c for c in (getattr(profile, "target_companies", None) or []) if isinstance(c, str) and c.strip()
        ],
    }
    facts["missing_skills"] = [
        _skill_item(entry, demand_signals) for entry in gap["missing_skills"][:MAX_MISSING_SKILLS]
    ]
    facts["matched_skills"] = [
        {"skill": m["skill"], "demand_signals": m["signal_count"]}
        for m in gap["matched_skills"][:MAX_MATCHED_SKILLS]
    ]
    facts["relevant_signals"] = [
        {
            "company": s["company"],
            "action": s["action"],
            "timing": s["timing"],
            "location": _place(s.get("location")),
            "count": s.get("count"),
            "matches_target_company": s["matches_target_company"],
            "matched_skills": s["matched_skills"],
            "missing_skills": s["missing_skills"],
            "source": (
                {"title": s["articles"][0]["title"], "link": s["articles"][0]["link"]}
                if s.get("articles") else None
            ),
        }
        for s in gap["relevant_signals"][:MAX_RELEVANT_SIGNALS]
    ]
    return facts
