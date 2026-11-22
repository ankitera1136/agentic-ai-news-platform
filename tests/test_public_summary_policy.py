"""Public endpoints must not return the raw RSS excerpt for restricted sources (HRKatha)."""
import os
import tempfile
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app import webapp
from backend.app.database import get_db
from backend.app.models import Article, Base, Cluster
from backend.services.excerpt import clean_excerpt, public_summary

RAW_EXCERPT = (
    "<p>Tencent has denied reports that it plans to lay off 50 per cent of its workforce, "
    "clarifying that recent management comments were not an indication of mass job cuts. "
    "The clarification follows widespread speculation [...]</p>\n"
    '<p>The post <a href="https://www.hrkatha.com/x">Tencent denies plans to cut half</a> '
    'appeared first on <a href="https://www.hrkatha.com">HR Katha</a>.</p>'
)
LLM = "Tencent says it is not planning mass layoffs."
MINT_RAW = "<p>Mint raw summary stays exactly as stored.</p>"


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _fk(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    db = Session()
    # size 1: the feed only hides members of multi-article clusters
    db.add(Cluster(id=1, primary_article_id="hr-1", size=1))
    db.commit()
    now = datetime.utcnow()
    db.add_all([
        Article(id="hr-1", title="Tencent denies layoffs plan workforce", link="https://hrkatha.example/1",
                summary=RAW_EXCERPT, llm_summary=LLM, source="HRKatha", created_at=now, cluster_id=1),
        # Older row stored before the policy: raw HTML excerpt and no llm_summary.
        Article(id="hr-2", title="PVH cuts jobs workforce globally", link="https://hrkatha.example/2",
                summary=RAW_EXCERPT, llm_summary="", source="HRKatha", created_at=now, cluster_id=1),
        Article(id="mint-1", title="Qualcomm layoffs severance Mint", link="https://mint.example/1",
                summary=MINT_RAW, llm_summary="Mint llm", source="Mint", created_at=now, cluster_id=1),
    ])
    db.commit()
    db.close()

    def override_get_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    webapp.app.dependency_overrides[get_db] = override_get_db
    webapp.FACT_CHECK_CACHE.clear()
    try:
        yield TestClient(webapp.app)
    finally:
        webapp.app.dependency_overrides.pop(get_db, None)
        engine.dispose()
        os.remove(path)


def _assert_clean(text):
    assert "appeared first on" not in text
    assert "<p>" not in text and "<a " not in text
    assert "[...]" not in text


# --- helpers -----------------------------------------------------------------

def test_clean_excerpt_strips_html_marker_and_footer():
    text = clean_excerpt(RAW_EXCERPT)
    _assert_clean(text)
    assert text.startswith("Tencent has denied reports")


def test_clean_excerpt_cuts_at_word_boundary_within_limit():
    text = clean_excerpt(RAW_EXCERPT, 100)
    assert len(text) <= 100
    assert clean_excerpt(RAW_EXCERPT).startswith(text)
    assert not text.endswith(" ")


def test_clean_excerpt_handles_empty_and_short_text():
    assert clean_excerpt(None) == ""
    assert clean_excerpt("<p>Short.</p>", 300) == "Short."


def test_public_summary_policy():
    assert public_summary("Mint", MINT_RAW, "x") == MINT_RAW  # unchanged for other sources
    assert public_summary("Mint", None, "x") == ""
    assert public_summary("HRKatha", RAW_EXCERPT, LLM) == LLM
    fallback = public_summary("HRKatha", RAW_EXCERPT, "")
    _assert_clean(fallback)
    assert 0 < len(fallback) <= 300


# --- endpoints ---------------------------------------------------------------

def test_api_articles_feed(client):
    data = {a["id"]: a for a in client.get("/api/articles/feed").json()["articles"]}
    assert data["hr-1"]["summary"] == LLM
    _assert_clean(data["hr-2"]["summary"])
    assert data["hr-2"]["summary"]
    assert data["mint-1"]["summary"] == MINT_RAW  # other sources untouched
    assert RAW_EXCERPT not in str(data["hr-1"]) and RAW_EXCERPT not in str(data["hr-2"])


def test_articles_endpoint(client):
    data = {a["id"]: a for a in client.get("/articles").json()}
    assert data["hr-1"]["summary"] == LLM
    _assert_clean(data["hr-2"]["summary"])
    assert data["mint-1"]["summary"] == MINT_RAW
    assert data["hr-1"]["llm_summary"] == LLM  # other fields unchanged


def test_home_page_html(client):
    html = client.get("/").text
    assert "appeared first on" not in html
    assert LLM in html


def test_cluster_list(client):
    data = {a["id"]: a for a in client.get("/api/clusters/1/articles").json()}
    assert data["hr-1"]["summary"] == LLM
    _assert_clean(data["hr-2"]["summary"])
    assert data["hr-2"]["summary"]
    assert data["mint-1"]["summary"] == "Mint llm"  # llm_summary first, as before


def test_chat_context_articles(client, monkeypatch):
    seen = {}

    def fake_answer(question, articles, history=None):
        seen["articles"] = articles
        return "ok"

    monkeypatch.setattr(webapp, "get_llm_answer_groq", fake_answer)
    response = client.post("/chat", json={"question": "tencent layoffs workforce pvh qualcomm mint", "history": []})
    assert response.status_code == 200
    by_id = {a["id"]: a for a in seen["articles"]}
    assert by_id["hr-1"]["summary"] == LLM
    _assert_clean(by_id["hr-2"]["summary"])
    assert by_id["mint-1"]["summary"] == MINT_RAW
    assert "appeared first on" not in str(seen["articles"])


def test_cross_check_inputs(client, monkeypatch):
    seen = {}

    def fake_analyze(articles):
        seen["articles"] = articles
        return {"agreement_index": 100.0}

    monkeypatch.setattr(webapp.contradiction_engine, "analyze_story_cluster", fake_analyze)
    assert client.get("/api/articles/hr-2/cross-check").status_code == 200
    by_id = {a["id"]: a for a in seen["articles"]}
    _assert_clean(by_id["hr-2"]["summary"])
    assert by_id["hr-1"]["summary"] == LLM
    assert by_id["mint-1"]["summary"] == MINT_RAW
