#!/usr/bin/env python3
"""List (and optionally fix) stored career_signals whose count is an estimate.

A count is an estimate when the sentence of its evidence_quote that states the number contains
estimate / estimated / reportedly / around / approximately / nearly / about / could / may /
"not (yet) been confirmed" (see grounding.ESTIMATE_CUES).

Default: LIST ONLY. Nothing is written unless --apply is given. --apply sets count = NULL for
exactly the rows listed (inside one transaction) and changes nothing else.

Examples:
  python scripts/estimate_guard_report.py
  python scripts/estimate_guard_report.py --database-url sqlite:///./copy.db
  python scripts/estimate_guard_report.py --apply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from backend.app.models import Article, CareerSignal  # noqa: E402
from backend.services.career.grounding import estimate_cue_in_count_sentence  # noqa: E402


def find_rows(db):
    rows = []
    query = db.query(CareerSignal, Article).join(Article, Article.id == CareerSignal.source_article_id)
    for signal, article in query.filter(CareerSignal.count.isnot(None)).order_by(CareerSignal.id).all():
        cue = estimate_cue_in_count_sentence(signal.count, signal.evidence_quote)
        if cue:
            rows.append((signal, article, cue))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=None, help="Override DATABASE_URL")
    parser.add_argument("--apply", action="store_true", help="Set count = NULL for the listed rows")
    args = parser.parse_args()

    if args.database_url:
        session_factory = sessionmaker(bind=create_engine(args.database_url))
    else:
        from backend.app.database import SessionLocal as session_factory  # noqa: E402
    db = session_factory()
    try:
        total_with_count = db.query(CareerSignal).filter(CareerSignal.count.isnot(None)).count()
        rows = find_rows(db)
        print(f"signals with a count: {total_with_count} | estimated (would change): {len(rows)}\n")
        for signal, article, cue in rows:
            print(f"id={signal.id} | {signal.company} | {signal.action} | count={signal.count} | cue='{cue}'")
            print(f"    article: {article.title[:90]}")
            print(f"    quote:   {(signal.evidence_quote or '')[:230]}\n")
        if not rows:
            return 0
        if not args.apply:
            print("LIST ONLY: nothing was changed. Re-run with --apply after reviewing the list.")
            return 0
        for signal, _, _ in rows:
            signal.count = None
        db.commit()
        print(f"APPLIED: count set to NULL on {len(rows)} signal(s).")
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
