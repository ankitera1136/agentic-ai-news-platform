"""
backend/services/brief/migrate.py - Idempotent weekly_briefs schema migration.

Creates the weekly_briefs table (one CareerRadar brief per user per ISO week).
Supports PostgreSQL and SQLite. Safe to run repeatedly; existing rows and all other
tables are left untouched.
"""
import os
import sqlite3
import sys
from typing import Optional


def upgrade(db_path: Optional[str] = None) -> None:
    """Apply the migration idempotently to PostgreSQL or SQLite."""
    if db_path:
        _upgrade_sqlite(db_path)
        return

    try:
        from backend.app.database import engine
        from backend.app.models import WeeklyBrief
        from sqlalchemy import inspect

        WeeklyBrief.__table__.create(bind=engine, checkfirst=True)
        if "weekly_briefs" not in set(inspect(engine).get_table_names()):
            print("[BRIEF MIGRATION] Warning: weekly_briefs still missing after create.")
        else:
            print(f"[BRIEF MIGRATION] Upgrade completed successfully for {engine.dialect.name}.")
    except Exception as e:
        print(f"[BRIEF MIGRATION] Engine migration fallback: {e}")
        _upgrade_sqlite()


def _sqlite_target(db_path: Optional[str]) -> str:
    target = db_path or os.getenv("DATABASE_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "tech_news.db",
    )
    if target.startswith("sqlite:///"):
        target = target.replace("sqlite:///", "")
    return target


def _upgrade_sqlite(db_path: Optional[str] = None) -> None:
    """SQLite-specific migration path used by tests and local fallback."""
    conn = sqlite3.connect(_sqlite_target(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_briefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                week_start DATE NOT NULL,
                facts TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                model_used TEXT,
                generated_at TEXT,
                CONSTRAINT uq_weekly_briefs_user_week UNIQUE (user_id, week_start),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_weekly_briefs_user_id ON weekly_briefs(user_id)")
        conn.commit()
    finally:
        conn.close()


def downgrade(db_path: Optional[str] = None) -> None:
    """Drop the weekly_briefs table only. Users and all other data are untouched."""
    if db_path:
        _downgrade_sqlite(db_path)
        return
    try:
        from backend.app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS weekly_briefs"))
            conn.commit()
    except Exception:
        _downgrade_sqlite()


def _downgrade_sqlite(db_path: Optional[str] = None) -> None:
    conn = sqlite3.connect(_sqlite_target(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS weekly_briefs")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "upgrade"
    db_arg = sys.argv[2] if len(sys.argv) > 2 else None
    if action == "downgrade":
        downgrade(db_arg)
    else:
        upgrade(db_arg)
