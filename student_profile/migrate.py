"""
student_profile/migrate.py - Idempotent student profile schema migration.

Creates the student_profiles table (one optional career profile per user).
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
        from backend.app.models import StudentProfile
        from sqlalchemy import inspect

        StudentProfile.__table__.create(bind=engine, checkfirst=True)
        if "student_profiles" not in set(inspect(engine).get_table_names()):
            print("[PROFILE MIGRATION] Warning: student_profiles still missing after create.")
        else:
            print(f"[PROFILE MIGRATION] Upgrade completed successfully for {engine.dialect.name}.")
    except Exception as e:
        print(f"[PROFILE MIGRATION] Engine migration fallback: {e}")
        _upgrade_sqlite()


def _sqlite_target(db_path: Optional[str]) -> str:
    target = db_path or os.getenv("DATABASE_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
            CREATE TABLE IF NOT EXISTS student_profiles (
                user_id TEXT PRIMARY KEY,
                branch TEXT,
                college TEXT,
                year_of_study INTEGER,
                graduation_year INTEGER,
                skills TEXT NOT NULL DEFAULT '[]',
                target_companies TEXT NOT NULL DEFAULT '[]',
                target_roles TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        conn.close()


def downgrade(db_path: Optional[str] = None) -> None:
    """Drop the student_profiles table only. Users and all other data are untouched."""
    if db_path:
        _downgrade_sqlite(db_path)
        return
    try:
        from backend.app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS student_profiles"))
            conn.commit()
    except Exception:
        _downgrade_sqlite()


def _downgrade_sqlite(db_path: Optional[str] = None) -> None:
    conn = sqlite3.connect(_sqlite_target(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS student_profiles")
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
