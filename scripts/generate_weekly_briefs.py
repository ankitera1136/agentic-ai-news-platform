#!/usr/bin/env python3
"""Pre-generate CareerRadar weekly briefs (for example every Monday from cron, or by hand).

Sequential, one user at a time, with CAREER_LLM_CALL_DELAY_SECONDS (default 1.0) between users so
the Groq rate limit is respected. Not part of the scrape job.

Examples:
  python scripts/generate_weekly_briefs.py --user-email student@example.com
  python scripts/generate_weekly_briefs.py --user-email student@example.com --force
  python scripts/generate_weekly_briefs.py --all
  python scripts/generate_weekly_briefs.py --all --force
"""
import argparse
import os
import sys
import time

from sqlalchemy import func

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app.database import SessionLocal, init_database  # noqa: E402
from backend.app.models import User  # noqa: E402
from backend.services.brief import facts as facts_module  # noqa: E402
from backend.services.brief import service  # noqa: E402


def _delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("CAREER_LLM_CALL_DELAY_SECONDS", "1.0")))
    except ValueError:
        return 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CareerRadar weekly briefs")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user-email", help="Generate the brief for one user")
    target.add_argument("--all", action="store_true", help="Generate briefs for every active user")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if this week's brief already exists")
    args = parser.parse_args()

    init_database()
    db = SessionLocal()
    try:
        if args.user_email:
            email = args.user_email.strip().lower()
            user = db.query(User).filter(func.lower(User.email) == email).first()
            if user is None:
                print(f"ERROR: no user with email {args.user_email!r}", file=sys.stderr)
                return 1
            users = [user]
        else:
            users = db.query(User).filter(User.is_active.is_(True)).order_by(User.created_at).all()

        week_start = facts_module.week_start_for(service._now())
        print(f"Week starting {week_start.isoformat()} | {len(users)} user(s) | force={args.force}")

        created = skipped = 0
        delay = _delay_seconds()
        for index, user in enumerate(users):
            exists = db.query(service.WeeklyBrief).filter(
                service.WeeklyBrief.user_id == user.id,
                service.WeeklyBrief.week_start == week_start,
            ).first() is not None
            if exists and not args.force:
                skipped += 1
                print(f"  [skipped] {user.email} | this week's brief already exists (use --force to replace)")
                continue
            row = (service.regenerate_brief if exists else service.get_or_create_brief)(db, user)
            created += 1
            headline = (row.content or {}).get("headline", "")
            print(f"  [{row.source}] {user.email} | {headline}")
            if index < len(users) - 1 and delay > 0:
                time.sleep(delay)

        print(f"Done: {created} generated, {skipped} skipped.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
