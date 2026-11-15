"""
career/migrate.py - Idempotent CareerRadar schema migration.

Creates (or upgrades) the internal tables used by the CareerRadar extractor:
- article_bodies      cached full article text (internal only)
- career_extractions  per-article idempotency ledger
- career_signals      structured career signals

Supports PostgreSQL and SQLite. Safe to run repeatedly; existing rows are preserved.
"""

import os
import sqlite3
import sys
from typing import Optional


def upgrade(db_path: Optional[str] = None) -> None:
    """Apply migrations idempotently to PostgreSQL or SQLite."""
    if db_path:
        _upgrade_sqlite(db_path)
        return

    try:
        from backend.app.database import engine
        from backend.app.models import Base
        from sqlalchemy import inspect

        Base.metadata.create_all(bind=engine)
        _ensure_engine_columns(engine)

        inspector = inspect(engine)
        existing = set(inspector.get_table_names())
        missing = [t for t in ("article_bodies", "career_extractions", "career_signals") if t not in existing]
        if missing:
            print(f"[CAREER MIGRATION] Warning: tables still missing after create_all: {missing}")
        else:
            print(f"[CAREER MIGRATION] Upgrade completed successfully for {engine.dialect.name}.")
    except Exception as e:
        print(f"[CAREER MIGRATION] Engine migration fallback: {e}")
        _upgrade_sqlite()


def _ensure_engine_columns(engine) -> None:
    """Add columns to pre-existing tables, which create_all() will not alter."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "career_signals" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("career_signals")}
    if "timing" not in columns:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE career_signals "
                "ADD COLUMN timing VARCHAR NOT NULL DEFAULT 'current'"
            ))


def _upgrade_sqlite(db_path: Optional[str] = None) -> None:
    """SQLite-specific migration path used by tests and local fallback."""
    target = db_path or os.getenv("DATABASE_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tech_news.db",
    )
    if target.startswith("sqlite:///"):
        target = target.replace("sqlite:///", "")

    conn = sqlite3.connect(target)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_bodies (
                article_id TEXT PRIMARY KEY,
                body_text TEXT,
                fetched_at TEXT,
                status TEXT NOT NULL DEFAULT 'failed',
                http_status INTEGER,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS career_extractions (
                article_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                text_source TEXT NOT NULL DEFAULT 'full_body',
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                model_used TEXT,
                signals_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                extracted_at TEXT,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS career_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_article_id TEXT NOT NULL,
                company TEXT NOT NULL,
                company_raw TEXT,
                action TEXT NOT NULL,
                timing TEXT NOT NULL DEFAULT 'current',
                count INTEGER,
                location_city TEXT,
                location_country TEXT,
                skills_mentioned TEXT,
                confidence REAL NOT NULL,
                evidence_quote TEXT,
                extracted_at TEXT,
                FOREIGN KEY (source_article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)

        # Older CareerRadar databases predate the timing column. ALTER preserves rows
        # and defaults every historical signal to "current".
        signal_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(career_signals)").fetchall()
        }
        if "timing" not in signal_columns:
            cursor.execute(
                "ALTER TABLE career_signals ADD COLUMN timing TEXT NOT NULL DEFAULT 'current'"
            )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_career_signals_article ON career_signals(source_article_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_career_signals_company_action ON career_signals(company, action)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_career_signals_action_time ON career_signals(action, extracted_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_career_extractions_status ON career_extractions(status)")

        conn.commit()
    finally:
        conn.close()


def downgrade(db_path: Optional[str] = None) -> None:
    """Drop CareerRadar tables only. Existing article data is untouched."""
    if db_path:
        _downgrade_sqlite(db_path)
        return

    try:
        from backend.app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            for table in ("career_signals", "career_extractions", "article_bodies"):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            conn.commit()
    except Exception:
        _downgrade_sqlite()


def _downgrade_sqlite(db_path: Optional[str] = None) -> None:
    target = db_path or os.getenv("DATABASE_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tech_news.db",
    )
    if target.startswith("sqlite:///"):
        target = target.replace("sqlite:///", "")

    conn = sqlite3.connect(target)
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS career_signals")
        cursor.execute("DROP TABLE IF EXISTS career_extractions")
        cursor.execute("DROP TABLE IF EXISTS article_bodies")
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
