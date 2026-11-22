#!/usr/bin/env python3
"""CareerRadar funnel report: per ISO week and per source (read-only).

Columns:
  articles       non-copy articles created in that week
  prefilter      articles that pass career_relevance() on title + RSS summary
  extracted      articles with a career_extractions ledger row (includes skipped rows)
  full_body      ledger rows extracted from a full article body
  headline_only  ledger rows that only had the stored summary (no LLM call)
  signals        career_signals produced

Week and pre-filter are computed in Python, so the same code runs on SQLite and
PostgreSQL. Only SELECTs are issued.

Examples:
  python scripts/career_funnel_report.py
  python scripts/career_funnel_report.py --weeks 4
  python scripts/career_funnel_report.py --database-url sqlite:///./copy.db
"""
import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, func  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from backend.app.models import Article, CareerExtraction, CareerSignal  # noqa: E402
from backend.services.career.filters import career_relevance  # noqa: E402

COLUMNS = ("articles", "prefilter", "extracted", "full_body", "headline_only", "signals")


def iso_week_label(moment) -> str:
    year, week, _ = moment.isocalendar()
    return f"{year}-W{week:02d}"


def build_report(db, weeks: Optional[int] = None) -> List[dict]:
    """Return one dict per (week, source), newest week first."""
    signal_counts = dict(
        db.query(CareerSignal.source_article_id, func.count(CareerSignal.id))
        .group_by(CareerSignal.source_article_id)
        .all()
    )
    rows = (
        db.query(Article, CareerExtraction)
        .outerjoin(CareerExtraction, CareerExtraction.article_id == Article.id)
        .filter(Article.is_copy_of.is_(None))
        .all()
    )

    table: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {c: 0 for c in COLUMNS})
    for article, extraction in rows:
        if article.created_at is None:
            continue
        cell = table[(iso_week_label(article.created_at), article.source or "Unknown")]
        cell["articles"] += 1
        if career_relevance(article.title, article.summary):
            cell["prefilter"] += 1
        if extraction is not None:
            cell["extracted"] += 1
            if extraction.text_source == "full_body":
                cell["full_body"] += 1
            elif extraction.text_source == "stored_summary":
                cell["headline_only"] += 1
        cell["signals"] += signal_counts.get(article.id, 0)

    result = [dict(week=w, source=s, **vals) for (w, s), vals in table.items()]
    result.sort(key=lambda r: (r["week"], r["articles"]), reverse=True)
    if weeks:
        keep = sorted({r["week"] for r in result}, reverse=True)[:weeks]
        result = [r for r in result if r["week"] in keep]
    return result


def format_report(rows: List[dict]) -> str:
    header = f"{'week':9} {'source':24}" + "".join(f"{c:>14}" for c in COLUMNS)
    lines = [header, "-" * len(header)]
    totals = {c: 0 for c in COLUMNS}
    for r in rows:
        lines.append(
            f"{r['week']:9} {r['source'][:24]:24}" + "".join(f"{r[c]:>14}" for c in COLUMNS)
        )
        for c in COLUMNS:
            totals[c] += r[c]
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':9} {'':24}" + "".join(f"{totals[c]:>14}" for c in COLUMNS))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, default=0, help="Only the newest N weeks (default: all)")
    parser.add_argument("--database-url", default=None,
                        help="Override DATABASE_URL (for example a copy of the DB)")
    args = parser.parse_args()

    if args.database_url:
        engine = create_engine(args.database_url)
        session_factory = sessionmaker(bind=engine)
    else:
        from backend.app.database import SessionLocal as session_factory  # noqa: E402

    db = session_factory()
    try:
        print(format_report(build_report(db, args.weeks or None)))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
