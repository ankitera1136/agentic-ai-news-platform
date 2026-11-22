"""Hiring-vs-layoff market views built only from stored CareerRadar signals.

Provides a company board, a city heatmap and a weekly trend. Everything is derived from
stored signals: no LLM, no network, no article bodies. Numbers describe news coverage, not
official hiring data, and headcount sums are never presented as complete totals.

Events come from matching.merge_events (same canonical company + action within 14 days, equal or
missing counts, across articles and ISO weeks, collapse to one signal; the highest-confidence
signal wins). The same function is used by matching.py, and a test checks both stay in agreement.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.app.models import Article, CareerSignal
from backend.services.career import geo
from backend.services.career.matching import (
    DedupedSignal,
    _article_ref,
    _as_utc,
    _company_key,
    _event_date,
    _plural,
    _public_article_ref,
    merge_events,
    merged_count,
    MERGE_NOTE,
    clean_skill_list,
)
from backend.services.career.normalizer import normalize_company

GROWTH_ACTIONS = ("HIRING", "EXPANSION", "NEW_OFFICE")
CUT_ACTIONS = ("LAYOFF", "HIRING_FREEZE")
MARKET_ACTIONS = GROWTH_ACTIONS + CUT_ACTIONS
MARKET_TIMINGS = ("current", "planned")
MAX_ARTICLES_PER_ITEM = 3
MAX_COMPANIES_PER_POINT = 5
GLOBAL_WORDS = {"", "global", "all", "worldwide", "world"}

NOTE = (
    "This reflects news coverage, not official hiring data. A company with more signals was "
    "written about more often, not necessarily hiring or cutting more."
)
HEADCOUNT_NOTE = (
    "Headcount sums only include signals that stated a number of people. They are partial and "
    "are not totals."
)
WEEK_BASIS = (
    "Weeks use the date the article was published; if that is missing, the time it was collected; "
    "if that is missing, the time the signal was extracted."
)


@dataclass
class _Item:
    signal: DedupedSignal
    group: str                 # "growth" | "cuts"
    location: geo.Location


# ---------------------------------------------------------------------------
# Loading, dedupe and location resolution
# ---------------------------------------------------------------------------

def country_filter(country: Optional[str]) -> Optional[str]:
    """Canonical country for a filter value; None means global."""
    if country is None or country.strip().casefold() in GLOBAL_WORDS:
        return None
    return geo.normalize_country(country)


def _load_market_signals(db: Session, cutoff: datetime, now_utc: datetime) -> List[DedupedSignal]:
    """HIRING / EXPANSION / NEW_OFFICE / LAYOFF / HIRING_FREEZE signals (current or planned),
    merged into events with matching.merge_events (same company + action within 14 days)."""
    rows = (
        db.query(CareerSignal, Article)
        .join(Article, Article.id == CareerSignal.source_article_id)
        .filter(CareerSignal.action.in_(MARKET_ACTIONS), CareerSignal.timing.in_(MARKET_TIMINGS))
        .all()
    )
    eligible = []
    for signal, article in rows:
        event = _event_date(article, signal)
        if event < cutoff or event > now_utc + timedelta(days=1):
            continue
        eligible.append((signal, article, event))
    groups = merge_events(eligible)

    deduped: List[DedupedSignal] = []
    for members in groups:
        members.sort(key=lambda m: (-(m[0].confidence or 0.0), -m[2].timestamp(), m[0].id))
        best, _, best_event = members[0]
        articles: List[Dict[str, Optional[str]]] = []
        seen_articles = set()
        for _, article, _ in members:
            if article.id in seen_articles:
                continue
            seen_articles.add(article.id)
            articles.append(_article_ref(article))
        deduped.append(
            DedupedSignal(
                id=best.id,
                company=normalize_company(best.company).canonical or best.company,
                action=best.action,
                timing=best.timing,
                count=merged_count(members),
                location_city=best.location_city,
                location_country=best.location_country,
                skills=clean_skill_list(
                    [s for member, _, _ in members for s in (member.skills_mentioned or [])]
                ),
                confidence=float(best.confidence or 0.0),
                evidence_quote="",   # market views never return quotes
                event_date=best_event,
                signal_ids=[member.id for member, _, _ in members],
                articles=articles,
            )
        )
    deduped.sort(key=lambda s: (-s.event_date.timestamp(), -s.confidence, s.id))
    return deduped


def _items(signals: Sequence[DedupedSignal]) -> List[_Item]:
    return [
        _Item(
            signal=s,
            group="growth" if s.action in GROWTH_ACTIONS else "cuts",
            location=geo.resolve_location(s.location_city, s.location_country),
        )
        for s in signals
    ]


def _now(now: Optional[datetime]) -> datetime:
    return _as_utc(now) or datetime.now(timezone.utc)


def _unique_articles(items: Sequence[_Item], limit: int = MAX_ARTICLES_PER_ITEM) -> List[Dict[str, Optional[str]]]:
    seen = set()
    out: List[Dict[str, Optional[str]]] = []
    for item in sorted(items, key=lambda i: -i.signal.event_date.timestamp()):
        for ref in item.signal.articles:
            if ref["id"] in seen:
                continue
            seen.add(ref["id"])
            out.append(_public_article_ref(ref))
            if len(out) >= limit:
                return out
    return out


def _headcount(items: Sequence[_Item]) -> Dict[str, Any]:
    """Sum of stated counts. total is None when no signal stated one (never a misleading 0)."""
    with_count = [i.signal.count for i in items if i.signal.count is not None]
    return {
        "total": sum(with_count) if with_count else None,
        "signals_with_count": len(with_count),
        "signals": len(items),
    }


def _basis(items: Sequence[_Item], period: str, country: Optional[str], extra: str = "") -> str:
    where = f" in {country}" if country else ""
    if not items:
        return f"No hiring or layoff signals were found in {period}{where}.{extra}"
    articles = {ref["id"] for i in items for ref in i.signal.articles}
    return (
        f"Based on {_plural(len(items), 'signal')} from {_plural(len(articles), 'article')} "
        f"in {period}{where}.{extra}"
    )


def _period(days: int) -> str:
    return "the last day" if days == 1 else f"the last {days} days"


# ---------------------------------------------------------------------------
# Company board
# ---------------------------------------------------------------------------

def company_board(
    db: Session,
    days: int = 30,
    country: Optional[str] = None,
    limit: int = 30,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_utc = _now(now)
    wanted = country_filter(country)
    items = _items(_load_market_signals(db, now_utc - timedelta(days=days), now_utc))
    if wanted:
        items = [i for i in items if i.location.country == wanted]

    by_company: Dict[str, List[_Item]] = {}
    for item in items:
        by_company.setdefault(item.signal.company.casefold(), []).append(item)

    board: List[Dict[str, Any]] = []
    for members in by_company.values():
        growth = [m for m in members if m.group == "growth"]
        cuts = [m for m in members if m.group == "cuts"]
        layoffs = [m for m in members if m.signal.action == "LAYOFF"]
        if len(growth) > len(cuts):
            net = "growing"
        elif len(cuts) > len(growth):
            net = "cutting"
        else:
            net = "mixed"
        board.append({
            "company": members[0].signal.company,
            "growth": len(growth),
            "cuts": len(cuts),
            "net_signal": net,
            "reported_hires": _headcount(growth),
            "reported_layoffs": _headcount(layoffs),
            "reported_by_articles": len({ref["id"] for m in members for ref in m.signal.articles}),
            "articles": _unique_articles(members),
        })
    board.sort(key=lambda c: (-(c["growth"] + c["cuts"]), c["company"].casefold()))

    return {
        "days": days,
        "country": wanted,
        "companies": board[:limit],
        "signals": len(items),
        "articles": len({ref["id"] for i in items for ref in i.signal.articles}),
        "basis": _basis(items, _period(days), wanted),
        "note": NOTE,
        "headcount_note": HEADCOUNT_NOTE,
        "merge_note": MERGE_NOTE,
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _point_kind(growth: int, cuts: int) -> str:
    if growth and cuts:
        return "mixed"
    return "growth" if growth else "cuts"


def heatmap_points(
    db: Session,
    days: int = 30,
    country: Optional[str] = None,
    action_group: str = "all",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    if action_group not in ("growth", "cuts", "all"):
        raise ValueError("action_group must be growth, cuts or all")
    now_utc = _now(now)
    wanted = country_filter(country)
    items = _items(_load_market_signals(db, now_utc - timedelta(days=days), now_utc))
    if action_group != "all":
        items = [i for i in items if i.group == action_group]

    not_stated = [i for i in items if not i.location.stated]
    stated = [i for i in items if i.location.stated]
    if wanted:
        in_scope = [i for i in stated if i.location.country == wanted]
    else:
        in_scope = stated
    excluded_by_country = len(stated) - len(in_scope)

    buckets: Dict[Tuple[str, str, str], List[_Item]] = {}
    unmapped: Dict[Tuple[str, str], int] = {}
    for item in in_scope:
        loc = item.location
        if loc.plottable:
            key = (loc.precision, loc.city or "", loc.country or "")
            buckets.setdefault(key, []).append(item)
        else:
            ukey = (loc.city or "", loc.country or "")
            unmapped[ukey] = unmapped.get(ukey, 0) + 1

    points: List[Dict[str, Any]] = []
    for (precision, city, place_country), members in buckets.items():
        growth = [m for m in members if m.group == "growth"]
        cuts = [m for m in members if m.group == "cuts"]
        layoffs = [m for m in members if m.signal.action == "LAYOFF"]
        companies = sorted({m.signal.company for m in members}, key=str.casefold)
        loc = members[0].location
        points.append({
            "id": f"{precision}:{city or place_country}|{place_country}",
            "city": city or None,
            "country": place_country or None,
            "lat": loc.lat,
            "lng": loc.lng,
            "precision": precision,
            "signal_count": len(members),
            "growth_count": len(growth),
            "cuts_count": len(cuts),
            "kind": _point_kind(len(growth), len(cuts)),
            "reported_hires": _headcount(growth),
            "reported_layoffs": _headcount(layoffs),
            "company_count": len(companies),
            "companies": companies[:MAX_COMPANIES_PER_POINT],
            "articles": _unique_articles(members),
        })
    points.sort(key=lambda p: (-p["signal_count"], (p["city"] or p["country"] or "").casefold()))

    unmapped_list = [
        {"city": city or None, "country": place_country or None, "signal_count": count}
        for (city, place_country), count in sorted(unmapped.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    inferred = sum(1 for i in in_scope if i.location.country_inferred)

    extra = ""
    if not_stated:
        if len(not_stated) == 1:
            extra = " 1 more signal in this period states no location and is not plotted."
        else:
            extra = f" {len(not_stated)} more signals in this period state no location and are not plotted."
    return {
        "days": days,
        "country": wanted,
        "group": action_group,
        "points": points,
        "signals": len(in_scope),
        "articles": len({ref["id"] for i in in_scope for ref in i.signal.articles}),
        "location_not_stated": len(not_stated),
        "unmapped_locations": unmapped_list,
        "excluded_by_country_filter": excluded_by_country,
        "country_inferred_from_city": inferred,
        "basis": _basis(in_scope, _period(days), wanted, extra),
        "note": NOTE + " Locations are the places the articles state. Places not in the built-in "
                       "coordinate table are listed, not plotted.",
        "headcount_note": HEADCOUNT_NOTE,
        "merge_note": MERGE_NOTE,
    }


# ---------------------------------------------------------------------------
# Weekly trend
# ---------------------------------------------------------------------------

def weekly_trend(
    db: Session,
    weeks: int = 8,
    country: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_utc = _now(now)
    wanted = country_filter(country)
    this_monday = (now_utc - timedelta(days=now_utc.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    first_monday = this_monday - timedelta(weeks=weeks - 1)

    items = _items(_load_market_signals(db, first_monday, now_utc))
    if wanted:
        items = [i for i in items if i.location.country == wanted]

    buckets: Dict[Tuple[int, int], Dict[str, int]] = {}
    for item in items:
        iso_year, iso_week, _ = item.signal.event_date.isocalendar()
        bucket = buckets.setdefault((iso_year, iso_week), {"growth": 0, "cuts": 0})
        bucket[item.group] += 1

    series: List[Dict[str, Any]] = []
    for offset in range(weeks):
        monday = first_monday + timedelta(weeks=offset)
        iso_year, iso_week, _ = monday.isocalendar()
        counts = buckets.get((iso_year, iso_week), {"growth": 0, "cuts": 0})
        series.append({
            "week": f"{iso_year}-W{iso_week:02d}",
            "start": monday.date().isoformat(),
            "growth": counts["growth"],
            "cuts": counts["cuts"],
        })

    return {
        "weeks": weeks,
        "country": wanted,
        "series": series,
        "totals": {
            "growth": sum(p["growth"] for p in series),
            "cuts": sum(p["cuts"] for p in series),
        },
        "signals": len(items),
        "articles": len({ref["id"] for i in items for ref in i.signal.articles}),
        "week_basis": WEEK_BASIS,
        "basis": _basis(items, f"the last {weeks} week{'s' if weeks != 1 else ''}", wanted),
        "note": NOTE,
    }
