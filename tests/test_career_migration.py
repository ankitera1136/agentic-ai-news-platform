import os
import sqlite3
import tempfile

import pytest

from career.migrate import downgrade, upgrade

CAREER_TABLES = {"article_bodies", "career_extractions", "career_signals"}
CAREER_INDEXES = {
    "idx_career_signals_article",
    "idx_career_signals_company_action",
    "idx_career_signals_action_time",
    "idx_career_extractions_status",
}


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            summary TEXT,
            llm_summary TEXT,
            published TEXT,
            image_url TEXT,
            source TEXT DEFAULT 'TechNews',
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        INSERT INTO articles (id, title, link, summary)
        VALUES ('art1', 'TCS to hire 500 engineers', 'https://example.com/1', 'TCS hiring')
        """
    )
    conn.commit()
    conn.close()
    yield path
    if os.path.exists(path):
        os.remove(path)


def _tables(path):
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        conn.close()


def _indexes(path):
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
    finally:
        conn.close()


def _columns(path, table):
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_upgrade_creates_career_tables_and_indexes(temp_db):
    upgrade(temp_db)

    tables = _tables(temp_db)
    assert CAREER_TABLES.issubset(tables)
    assert CAREER_INDEXES.issubset(_indexes(temp_db))
    assert "timing" in _columns(temp_db, "career_signals")

    conn = sqlite3.connect(temp_db)
    try:
        row = conn.execute("SELECT id, title FROM articles WHERE id='art1'").fetchone()
    finally:
        conn.close()
    assert row == ("art1", "TCS to hire 500 engineers")


def test_upgrade_adds_timing_column_to_legacy_signals_table(temp_db):
    conn = sqlite3.connect(temp_db)
    try:
        conn.execute(
            """
            CREATE TABLE career_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_article_id TEXT NOT NULL,
                company TEXT NOT NULL,
                company_raw TEXT,
                action TEXT NOT NULL,
                count INTEGER,
                location_city TEXT,
                location_country TEXT,
                skills_mentioned TEXT,
                confidence REAL NOT NULL,
                evidence_quote TEXT,
                extracted_at TEXT,
                FOREIGN KEY (source_article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            INSERT INTO career_signals
                (source_article_id, company, action, confidence, evidence_quote)
            VALUES ('art1', 'Nvidia', 'LAYOFF', 0.95, 'old quote')
            """
        )
        conn.commit()
    finally:
        conn.close()

    upgrade(temp_db)
    upgrade(temp_db)

    assert "timing" in _columns(temp_db, "career_signals")
    conn = sqlite3.connect(temp_db)
    try:
        timing = conn.execute("SELECT timing FROM career_signals WHERE company='Nvidia'").fetchone()[0]
    finally:
        conn.close()
    assert timing == "current"


def test_upgrade_is_idempotent(temp_db):
    upgrade(temp_db)
    upgrade(temp_db)

    assert CAREER_TABLES.issubset(_tables(temp_db))
    conn = sqlite3.connect(temp_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_downgrade_drops_only_career_tables(temp_db):
    upgrade(temp_db)
    downgrade(temp_db)

    tables = _tables(temp_db)
    assert "articles" in tables
    assert not (CAREER_TABLES & tables)

    conn = sqlite3.connect(temp_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_cascade_delete_removes_career_rows(temp_db):
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """
            INSERT INTO article_bodies (article_id, body_text, status, attempts)
            VALUES ('art1', 'body', 'ok', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO career_signals
                (source_article_id, company, action, confidence, evidence_quote)
            VALUES ('art1', 'Tata Consultancy Services', 'HIRING', 0.9, 'TCS to hire 500 engineers')
            """
        )
        conn.commit()

        conn.execute("DELETE FROM articles WHERE id='art1'")
        conn.commit()

        bodies = conn.execute("SELECT COUNT(*) FROM article_bodies").fetchone()[0]
        signals = conn.execute("SELECT COUNT(*) FROM career_signals").fetchone()[0]
        assert bodies == 0
        assert signals == 0
    finally:
        conn.close()
