import os
import sqlite3
import tempfile

import pytest

from student_profile.migrate import downgrade, upgrade


@pytest.fixture
def temp_db():
    """A SQLite file that already has a users table with data, like a live database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT NOT NULL)")
    conn.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
    conn.execute("INSERT INTO users VALUES ('u1', 'a@example.com')")
    conn.execute("INSERT INTO users VALUES ('u2', 'b@example.com')")
    conn.execute("INSERT INTO articles VALUES ('art1', 'Some title')")
    conn.commit()
    conn.close()
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def _tables(path):
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _columns(path, table):
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    finally:
        conn.close()


def test_upgrade_creates_the_student_profiles_table(temp_db):
    assert "student_profiles" not in _tables(temp_db)

    upgrade(temp_db)

    assert "student_profiles" in _tables(temp_db)
    assert _columns(temp_db, "student_profiles") == {
        "user_id", "branch", "college", "year_of_study", "graduation_year",
        "skills", "target_companies", "target_roles", "updated_at",
    }


def test_upgrade_is_idempotent_and_preserves_profile_rows(temp_db):
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "INSERT INTO student_profiles (user_id, branch, skills) VALUES ('u1', 'CSE', '[\"Python\"]')"
    )
    conn.commit()
    conn.close()

    upgrade(temp_db)
    upgrade(temp_db)

    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT user_id, branch, skills FROM student_profiles").fetchall()
    conn.close()
    assert rows == [("u1", "CSE", '["Python"]')]


def test_upgrade_preserves_existing_data_in_other_tables(temp_db):
    upgrade(temp_db)

    conn = sqlite3.connect(temp_db)
    users = conn.execute("SELECT id, email FROM users ORDER BY id").fetchall()
    articles = conn.execute("SELECT id, title FROM articles").fetchall()
    conn.close()
    assert users == [("u1", "a@example.com"), ("u2", "b@example.com")]
    assert articles == [("art1", "Some title")]


def test_list_columns_default_to_an_empty_json_list(temp_db):
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    conn.execute("INSERT INTO student_profiles (user_id) VALUES ('u2')")
    row = conn.execute(
        "SELECT skills, target_companies, target_roles FROM student_profiles"
    ).fetchone()
    conn.close()
    assert row == ("[]", "[]", "[]")


def test_deleting_a_user_cascades_to_the_profile(temp_db):
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO student_profiles (user_id, branch) VALUES ('u1', 'CSE')")
    conn.execute("INSERT INTO student_profiles (user_id, branch) VALUES ('u2', 'ECE')")
    conn.commit()

    conn.execute("DELETE FROM users WHERE id = 'u1'")
    conn.commit()
    remaining = conn.execute("SELECT user_id FROM student_profiles").fetchall()
    conn.close()

    assert remaining == [("u2",)]


def test_downgrade_drops_only_the_profile_table(temp_db):
    upgrade(temp_db)

    downgrade(temp_db)

    assert "student_profiles" not in _tables(temp_db)
    assert {"users", "articles"} <= _tables(temp_db)
    conn = sqlite3.connect(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2
    conn.close()


def test_sqlalchemy_model_and_migration_agree_on_columns(temp_db):
    """The raw-SQLite path must create every column the ORM model declares."""
    from backend.app.models import StudentProfile

    upgrade(temp_db)

    assert {c.name for c in StudentProfile.__table__.columns} == _columns(temp_db, "student_profiles")


def test_engine_path_creates_table_and_is_idempotent(monkeypatch):
    """The SQLAlchemy path (the one PostgreSQL uses) runs twice without error or data loss."""
    from sqlalchemy import create_engine, inspect, text

    import backend.app.database as database
    from backend.app.models import User

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    try:
        User.__table__.create(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO users (id, email, hashed_password) VALUES ('u1', 'a@x.com', 'h')"))
        monkeypatch.setattr(database, "engine", engine)

        upgrade()
        upgrade()

        assert "student_profiles" in set(inspect(engine).get_table_names())
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar() == 1
    finally:
        engine.dispose()
        os.remove(path)
