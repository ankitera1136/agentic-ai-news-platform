#!/usr/bin/env python3
"""Backfill CareerRadar signals for stored articles.

Runs regardless of CAREER_RADAR_ENABLED; that flag only controls the automatic
pipeline hook in the scraper.

Examples:
  python scripts/backfill_career_signals.py --limit 20
  python scripts/backfill_career_signals.py --limit 20 --max-per-source 4
  python scripts/backfill_career_signals.py --limit 20 --source YourStory --source Inc42
  python scripts/backfill_career_signals.py --limit 20 --keyword-filter
  python scripts/backfill_career_signals.py --article-id abc123 --article-id def456
  python scripts/backfill_career_signals.py --limit 20 --dry-run
  python scripts/backfill_career_signals.py --limit 20 --keyword-filter --force
  python scripts/backfill_career_signals.py --reextract-all --no-fetch
  python scripts/backfill_career_signals.py --limit 40 --unprocessed --title-keyword layoff --title-keyword hiring
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

from sqlalchemy import func, or_

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import SessionLocal, init_database  # noqa: E402
from backend.app.models import (  # noqa: E402
    Article,
    CareerExtraction,
    CareerSignal,
)
from backend.services.career.extractor import (  # noqa: E402
    BatchStats,
    CareerExtractor,
    default_batch_limit,
)
from backend.services.career.repository import pending_articles  # noqa: E402

DEFAULT_KEYWORDS = (
    "hiring",
    "layoff",
    "laid off",
    "jobs",
    "funding",
    "raises",
    "expands",
    "office",
    "headcount",
    "recruit",
)


def _short(value, width: int) -> str:
    text = "" if value is None else str(value).replace("\n", " ").replace("|", "/")
    if len(text) > width:
        text = text[: max(0, width - 1)] + "…"
    return text.ljust(width)


def _print_summary(stats: BatchStats) -> None:
    print("\n" + "=" * 72)
    print("CAREERRADAR BACKFILL SUMMARY")
    print("=" * 72)
    print(f"processed (LLM attempted): {stats.processed}")
    print(f"succeeded:                 {stats.succeeded}")
    print(f"failed:                    {stats.failed}")
    print(f"zero-signal:               {stats.zero_signal}")
    print(f"skipped (unchanged/limit): {stats.skipped}")
    print(f"skipped (roundup title):   {stats.skipped_roundup}")
    print(f"headline-only (no signals): {stats.headline_only}")
    print(f"no cached body (skipped):  {stats.no_body}")
    print(f"not found:                 {stats.not_found}")

    counters = stats.dropped.as_dict()
    print(f"verified signals:          {counters.pop('verified', 0)}")
    drops = {k: v for k, v in counters.items() if v}
    if drops:
        print("dropped by check:")
        for key, value in drops.items():
            print(f"  {key}: {value}")
    else:
        print("dropped by check:          none")

    if stats.unknown_companies:
        print(f"unknown companies ({len(stats.unknown_companies)}):")
        for company in stats.unknown_companies:
            print(f"  - {company}")
    else:
        print("unknown companies:         none")


def _parse_keywords(value: str) -> List[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _matching_candidates(
    db,
    window: int,
    sources: List[str],
    keywords: List[str],
) -> List[Article]:
    query = db.query(Article).filter(Article.is_copy_of.is_(None))

    conditions = []
    clean_sources = [source.strip().lower() for source in sources if source.strip()]
    if clean_sources:
        conditions.append(func.lower(Article.source).in_(clean_sources))
    for keyword in keywords:
        pattern = f"%{keyword.lower()}%"
        conditions.append(func.lower(Article.title).like(pattern))
        conditions.append(func.lower(Article.summary).like(pattern))

    if conditions:
        # Union: articles from the requested sources OR matching any keyword.
        query = query.filter(or_(*conditions))

    return query.order_by(Article.created_at.desc()).limit(window).all()


def _select_articles(
    db,
    limit: int,
    article_ids: List[str],
    max_per_source: int,
    sources: List[str],
    keywords: List[str],
) -> List[Article]:
    if article_ids:
        return (
            db.query(Article)
            .filter(Article.id.in_(article_ids))
            .order_by(Article.created_at.desc())
            .all()
        )

    filtered = bool(sources or keywords)
    if not filtered and max_per_source <= 0:
        return pending_articles(db, limit)

    window = max(limit * 10, 100) if (filtered or max_per_source > 0) else limit
    candidates = _matching_candidates(db, window, sources, keywords)
    if max_per_source <= 0:
        return candidates[:limit]

    per_source: Dict[str, int] = defaultdict(int)
    selected: List[Article] = []
    for article in candidates:
        source = article.source or "Unknown"
        if per_source[source] >= max_per_source:
            continue
        selected.append(article)
        per_source[source] += 1
        if len(selected) >= limit:
            break
    return selected


def _unprocessed_filter(query):
    """Restrict a query to articles with no CareerRadar ledger row."""
    processed = db_exists_subquery()
    return query.filter(~processed)


def db_exists_subquery():
    from sqlalchemy import exists

    return exists().where(CareerExtraction.article_id == Article.id)


def _title_candidates(db, limit: int, keywords: List[str], sources: List[str], unprocessed: bool):
    """Newest non-copy articles whose TITLE matches any keyword and/or source, optionally unprocessed."""
    query = db.query(Article).filter(Article.is_copy_of.is_(None))
    if unprocessed:
        query = _unprocessed_filter(query)
    title_conditions = [
        func.lower(Article.title).like(f"%{kw.strip().lower()}%")
        for kw in keywords if kw.strip()
    ]
    if title_conditions:
        query = query.filter(or_(*title_conditions))
    clean_sources = [source.strip().lower() for source in sources if source.strip()]
    if clean_sources:
        query = query.filter(func.lower(Article.source).in_(clean_sources))
    return query.order_by(Article.created_at.desc()).limit(limit).all()


def _all_previously_extracted_ids(db) -> List[str]:
    ledger = [row[0] for row in db.query(CareerExtraction.article_id).all()]
    signals = [row[0] for row in db.query(CareerSignal.source_article_id).distinct().all()]
    return sorted(set(ledger) | set(signals))


def _reset_all(db) -> tuple:
    """Delete EVERY career_signals and career_extractions row. article_bodies are kept."""
    signals = db.query(CareerSignal).delete(synchronize_session=False)
    extractions = db.query(CareerExtraction).delete(synchronize_session=False)
    db.commit()
    return signals, extractions


def _reset_extractions(db, article_ids: List[str]) -> tuple:
    """Delete signals and extraction ledger rows so the articles are re-extracted.

    Cached article bodies are kept, so no page is fetched again.
    """
    if not article_ids:
        return 0, 0
    signals = (
        db.query(CareerSignal)
        .filter(CareerSignal.source_article_id.in_(article_ids))
        .delete(synchronize_session=False)
    )
    extractions = (
        db.query(CareerExtraction)
        .filter(CareerExtraction.article_id.in_(article_ids))
        .delete(synchronize_session=False)
    )
    db.commit()
    return signals, extractions


def _print_signal_table(db, article_ids: List[str]) -> None:
    if not article_ids:
        print("\nNo articles were processed.")
        return

    rows = (
        db.query(CareerSignal, Article, CareerExtraction)
        .join(Article, Article.id == CareerSignal.source_article_id)
        .outerjoin(CareerExtraction, CareerExtraction.article_id == Article.id)
        .filter(CareerSignal.source_article_id.in_(article_ids))
        .order_by(Article.created_at.desc(), CareerSignal.id.asc())
        .all()
    )

    print("\n" + "=" * 72)
    print("EXTRACTED SIGNALS")
    print("=" * 72)
    if not rows:
        print("No signals were extracted from the processed articles.")
        return

    header = (
        f"{_short('article title', 40)} | {_short('company', 22)} | "
        f"{_short('action', 14)} | {'count':>7} | {_short('location', 20)} | "
        f"{_short('skills', 25)} | {'conf':>5} | {_short('timing', 12)} | "
        f"{_short('text_source', 14)} | evidence_quote"
    )
    print(header)
    print("-" * len(header))

    for signal, article, extraction in rows:
        location = ", ".join(
            part for part in (signal.location_city, signal.location_country) if part
        ) or "-"
        skills = ", ".join(signal.skills_mentioned or []) or "-"
        count = str(signal.count) if signal.count is not None else "-"
        text_source = extraction.text_source if extraction else "-"
        print(
            f"{_short(article.title, 40)} | {_short(signal.company, 22)} | "
            f"{_short(signal.action, 14)} | {count:>7} | {_short(location, 20)} | "
            f"{_short(skills, 25)} | {signal.confidence:5.2f} | "
            f"{_short(signal.timing or '-', 12)} | {_short(text_source, 14)} | "
            f"{_short(signal.evidence_quote, 180)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill CareerRadar signals")
    parser.add_argument("--limit", type=int, default=default_batch_limit(),
                        help="Maximum number of articles to inspect (default: runtime limit)")
    parser.add_argument("--article-id", action="append", dest="article_ids", default=[],
                        help="Process a specific article id (repeatable)")
    parser.add_argument("--max-per-source", type=int, default=0,
                        help="Cap articles per source to get a mix (0 = no cap)")
    parser.add_argument("--source", action="append", dest="sources", default=[],
                        help="Only select articles from this source (repeatable)")
    parser.add_argument("--keyword-filter", nargs="?", default=None,
                        const=",".join(DEFAULT_KEYWORDS),
                        help="Select articles whose title or summary matches any comma-separated "
                             "keyword. Omit the value to use the built-in hiring/layoff list.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select and print candidates without fetching or calling the LLM")
    parser.add_argument("--force", action="store_true",
                        help="Delete the selected articles' career_signals and career_extractions "
                             "rows first, then re-extract them (cached bodies are kept)")
    parser.add_argument("--reextract-all", action="store_true",
                        help="Delete ALL career_signals and career_extractions rows, then "
                             "re-extract every article that had one (cached bodies are kept)")
    parser.add_argument("--unprocessed", action="store_true",
                        help="Only select articles that have no career_extractions row")
    parser.add_argument("--title-keyword", action="append", dest="title_keywords", default=[],
                        help="Only select articles whose TITLE contains this keyword (repeatable)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Only use already-cached bodies; never fetch")
    parser.add_argument("--no-table", action="store_true",
                        help="Print only the summary, not the signal table")
    parser.add_argument("--model", default=None,
                        help="Optional Groq model override for this run")
    args = parser.parse_args()

    init_database()
    db = SessionLocal()
    try:
        reextract_ids: List[str] = []
        if args.reextract_all:
            reextract_ids = _all_previously_extracted_ids(db)
            print(f"--reextract-all: {len(reextract_ids)} previously processed article(s) found.")
        if args.reextract_all:
            articles = (
                db.query(Article).filter(Article.id.in_(reextract_ids))
                .order_by(Article.created_at.desc()).all()
                if reextract_ids else []
            )
        elif args.title_keywords or args.unprocessed:
            articles = _title_candidates(
                db, max(1, args.limit), args.title_keywords, args.sources, args.unprocessed
            )
        else:
            articles = _select_articles(
                db,
                max(1, args.limit),
                args.article_ids,
                max(0, args.max_per_source),
                args.sources,
                _parse_keywords(args.keyword_filter),
            )
        if not articles:
            print("No matching articles found.")
            return 0

        print(f"Selected {len(articles)} article(s), newest first:")
        for article in articles:
            print(f"  - [{article.source or 'Unknown'}] {article.title}")

        if args.dry_run:
            print("\nDry run: no bodies fetched, no LLM calls, no writes.")
            return 0

        extractor = CareerExtractor(model=args.model)
        if not extractor.api_key:
            print(
                "\nERROR: GROQ_API_KEY (or GROK_API_KEY) is not set. "
                "Set it in the environment or .env before running the backfill.",
                file=sys.stderr,
            )
            return 1

        if args.reextract_all:
            deleted_signals, deleted_extractions = _reset_all(db)
            print(
                f"\n--reextract-all: deleted {deleted_signals} signal(s) and "
                f"{deleted_extractions} extraction record(s) (all rows); re-extracting."
            )
        elif args.force:
            deleted_signals, deleted_extractions = _reset_extractions(
                db, [article.id for article in articles]
            )
            print(
                f"\n--force: deleted {deleted_signals} signal(s) and "
                f"{deleted_extractions} extraction record(s); re-extracting."
            )

        stats = BatchStats()
        for article in articles:
            outcome = extractor.process_article(
                article.id,
                db=db,
                allow_fetch=not args.no_fetch,
            )
            stats.record(outcome)
            flag = "ok" if outcome.status == "succeeded" else outcome.status
            print(
                f"  [{flag}] {article.id[:8]} | signals={outcome.signals_count} | "
                f"{outcome.text_source or '-'} | body={outcome.body_length} | "
                f"{outcome.error or ''}"
            )

        _print_summary(stats)

        if not args.no_table:
            _print_signal_table(db, [article.id for article in articles])

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
