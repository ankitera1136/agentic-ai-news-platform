import os
import tempfile
from datetime import datetime

import pytest

def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="Run slow integration tests")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Skipped: requires --slow flag and model download")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture
def career_db():
    """Isolated SQLite database with the full current schema and FKs enabled."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    from backend.app.models import Base

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def make_article(career_db):
    from backend.app.models import Article

    def _make(
        article_id="art1",
        title="Acme Robotics expands Bengaluru team",
        link="https://example.com/article",
        summary="Acme Robotics will hire engineers.",
        llm_summary="Acme Robotics is expanding.",
        source="TechCrunch",
        is_copy_of=None,
        created_at=None,
    ):
        article = Article(
            id=article_id,
            title=title,
            link=link,
            summary=summary,
            llm_summary=llm_summary,
            source=source,
            is_copy_of=is_copy_of,
            created_at=created_at or datetime.utcnow(),
        )
        career_db.add(article)
        career_db.commit()
        return article

    return _make
