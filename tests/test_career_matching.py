"""CareerRadar skill demand and skill-gap matching: service logic and /api/career routes."""
import os
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.auth import get_current_active_user
from backend.app.database import get_db
from backend.app.models import Article, ArticleBody, Base, CareerSignal, StudentProfile, User
from backend.app.routers import career as career_router_module
from backend.services.career import matching

# Fixed "now" for service-level tests: Sunday 2026-11-15. ISO week 45 is Mon Nov 2 to Sun Nov 8.
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
WEEK_45_TUE = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
WEEK_45_WED = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
WEEK_46_TUE = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


class Env:
    def __init__(self, session_factory, current):
        self.session_factory = session_factory
        self.current = current
        self._counter = 0

    def db(self):
        return self.session_factory()

    def add_signal(
        self,
        *,
        company="Acme Robotics",
        action="HIRING",
        timing="current",
        skills=None,
        confidence=0.9,
        when=WEEK_45_TUE,
        title=None,
        count=None,
        quote="Acme will hire engineers.",
        article_id=None,
        source="TechCrunch",
        link=None,
    ):
        self._counter += 1
        article_id = article_id or f"art-{self._counter}"
        db = self.db()
        try:
            article = db.query(Article).filter(Article.id == article_id).first()
            if article is None:
                article = Article(
                    id=article_id,
                    title=title or f"Article {article_id}",
                    link=link or f"https://example.com/{article_id}",
                    summary="s",
                    source=source,
                    published=format_datetime(when),
                    created_at=when,
                )
                db.add(article)
                db.flush()
            signal = CareerSignal(
                source_article_id=article_id,
                company=company,
                action=action,
                timing=timing,
                count=count,
                location_city="Bengaluru",
                location_country="India",
                skills_mentioned=skills or [],
                confidence=confidence,
                evidence_quote=quote,
            )
            db.add(signal)
            db.commit()
            return signal.id
        finally:
            db.close()


def _build_app(session_factory, current=None):
    app = FastAPI()
    app.include_router(career_router_module.router)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    if current is not None:
        async def override_user():
            db = session_factory()
            try:
                return db.query(User).filter(User.id == current["user_id"]).one()
            finally:
                db.close()

        app.dependency_overrides[get_current_active_user] = override_user
    return app


@pytest.fixture
def env():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()
    db.add_all([
        User(id=ALICE, email="alice@example.com", hashed_password="secret-hash-a"),
        User(id=BOB, email="bob@example.com", hashed_password="secret-hash-b"),
    ])
    db.commit()
    db.close()

    current = {"user_id": ALICE}
    environment = Env(session_factory, current)
    environment.client = TestClient(_build_app(session_factory, current))
    environment.anonymous = TestClient(_build_app(session_factory, None))
    try:
        yield environment
    finally:
        engine.dispose()
        os.remove(path)


def _recent(days_ago=3):
    """A timestamp safely inside any default window, for tests that go through the HTTP API."""
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _profile(skills=(), companies=()):
    return SimpleNamespace(skills=list(skills), target_companies=list(companies))


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def test_same_company_action_and_week_from_two_articles_becomes_one_signal(env):
    env.add_signal(company="Acme Robotics", skills=["Python"], confidence=0.80, when=WEEK_45_TUE, article_id="a1", title="First")
    env.add_signal(company="Acme Robotics", skills=["SQL"], confidence=0.95, when=WEEK_45_WED, article_id="a2", title="Second", quote="Best quote")

    db = env.db()
    signals = matching.load_demand_signals(db, days=30, now=NOW)
    db.close()

    assert len(signals) == 1
    signal = signals[0]
    assert signal.confidence == 0.95            # highest confidence kept
    assert signal.evidence_quote == "Best quote"
    assert signal.skills == ["Python", "SQL"] or signal.skills == ["SQL", "Python"]
    assert sorted(signal.skills) == ["Python", "SQL"]  # skills merged
    assert sorted(a["title"] for a in signal.articles) == ["First", "Second"]  # source articles merged
    assert len(signal.signal_ids) == 2


def test_same_company_in_different_weeks_within_14_days_merges(env):
    env.add_signal(when=WEEK_45_TUE, article_id="a1")
    env.add_signal(when=WEEK_46_TUE, article_id="a2")

    db = env.db()
    signals = matching.load_demand_signals(db, days=30, now=NOW)
    db.close()

    assert len(signals) == 1 and len(signals[0].articles) == 2


def test_same_company_more_than_14_days_apart_stays_separate(env):
    env.add_signal(when=datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc), article_id="a1")
    env.add_signal(when=WEEK_46_TUE, article_id="a2")

    db = env.db()
    signals = matching.load_demand_signals(db, days=30, now=NOW)
    db.close()

    assert len(signals) == 2


def test_same_week_different_action_stays_separate(env):
    env.add_signal(action="HIRING", article_id="a1")
    env.add_signal(action="EXPANSION", article_id="a2")

    db = env.db()
    assert len(matching.load_demand_signals(db, days=30, now=NOW)) == 2
    db.close()


def test_company_aliases_dedupe_together(env):
    env.add_signal(company="TCS", skills=["Java"], article_id="a1")
    env.add_signal(company="Tata Consultancy Services", skills=["SQL"], article_id="a2", when=WEEK_45_WED)

    db = env.db()
    signals = matching.load_demand_signals(db, days=30, now=NOW)
    db.close()

    assert len(signals) == 1
    assert signals[0].company == "Tata Consultancy Services"


def test_dedupe_happens_before_counting(env):
    """Two articles about the same company-week must count once in skill demand."""
    env.add_signal(company="Acme", skills=["Python"], article_id="a1")
    env.add_signal(company="Acme", skills=["Python"], article_id="a2", when=WEEK_45_WED)

    db = env.db()
    demand = matching.skill_demand(db, days=30, now=NOW)
    db.close()

    assert demand["skills"][0]["skill"] == "Python"
    assert demand["skills"][0]["signal_count"] == 1
    assert demand["skills"][0]["company_count"] == 1
    assert demand["signals"] == 1
    assert demand["articles"] == 2


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def test_layoff_funding_and_freeze_are_excluded_from_skill_demand(env):
    env.add_signal(company="A", action="LAYOFF", skills=["Cobol"], article_id="a1")
    env.add_signal(company="B", action="FUNDING", skills=["Fortran"], article_id="a2")
    env.add_signal(company="C", action="HIRING_FREEZE", skills=["Perl"], article_id="a3")
    env.add_signal(company="D", action="HIRING", skills=["Python"], article_id="a4")

    db = env.db()
    demand = matching.skill_demand(db, days=30, now=NOW)
    db.close()

    assert [s["skill"] for s in demand["skills"]] == ["Python"]
    assert demand["signals"] == 1


def test_expansion_and_new_office_count_and_past_context_does_not(env):
    env.add_signal(company="A", action="EXPANSION", skills=["Go"], article_id="a1")
    env.add_signal(company="B", action="NEW_OFFICE", timing="planned", skills=["Rust"], article_id="a2")
    env.add_signal(company="C", action="HIRING", timing="past_context", skills=["Ruby"], article_id="a3")

    db = env.db()
    demand = matching.skill_demand(db, days=30, now=NOW)
    db.close()

    assert sorted(s["skill"] for s in demand["skills"]) == ["Go", "Rust"]


def test_signals_outside_the_window_are_ignored(env):
    old = NOW - timedelta(days=40)
    env.add_signal(company="Old Co", skills=["Perl"], when=old, article_id="old")
    env.add_signal(company="New Co", skills=["Python"], when=WEEK_46_TUE, article_id="new")

    db = env.db()
    assert [s["skill"] for s in matching.skill_demand(db, days=30, now=NOW)["skills"]] == ["Python"]
    wider = matching.skill_demand(db, days=90, now=NOW)
    db.close()
    assert sorted(s["skill"] for s in wider["skills"]) == ["Perl", "Python"]


# ---------------------------------------------------------------------------
# Skill demand ranking
# ---------------------------------------------------------------------------

def test_skill_demand_ranks_by_distinct_companies_then_signals(env):
    env.add_signal(company="A", skills=["Python", "SQL"], article_id="a1")
    env.add_signal(company="B", skills=["Python", "SQL"], article_id="a2")
    env.add_signal(company="C", skills=["Python", "Docker"], article_id="a3")
    env.add_signal(company="C", action="EXPANSION", skills=["Docker"], article_id="a4")

    db = env.db()
    skills = matching.skill_demand(db, days=30, now=NOW)["skills"]
    db.close()

    assert [s["skill"] for s in skills] == ["Python", "SQL", "Docker"]
    python, sql, docker = skills
    assert (python["company_count"], python["signal_count"]) == (3, 3)
    assert (sql["company_count"], sql["signal_count"]) == (2, 2)
    assert (docker["company_count"], docker["signal_count"]) == (1, 2)  # one company, two signals
    assert python["companies"] == ["A", "B", "C"]


def test_companies_per_skill_are_capped_at_five(env):
    for i in range(8):
        env.add_signal(company=f"Company {i}", skills=["Python"], article_id=f"a{i}")

    db = env.db()
    entry = matching.skill_demand(db, days=30, now=NOW)["skills"][0]
    db.close()

    assert entry["company_count"] == 8
    assert len(entry["companies"]) == 5


def test_signal_skill_aliases_are_canonicalized_in_demand(env):
    env.add_signal(company="A", skills=["js", "py"], article_id="a1")
    env.add_signal(company="B", skills=["JavaScript", "Python3"], article_id="a2")

    db = env.db()
    skills = matching.skill_demand(db, days=30, now=NOW)["skills"]
    db.close()

    assert sorted(s["skill"] for s in skills) == ["JavaScript", "Python"]
    assert all(s["company_count"] == 2 for s in skills)


def test_skill_demand_limit_and_basis_strings(env):
    for i in range(5):
        env.add_signal(company=f"Co {i}", skills=[f"Skill{i}"], article_id=f"a{i}")

    db = env.db()
    demand = matching.skill_demand(db, days=30, limit=3, now=NOW)
    db.close()

    assert len(demand["skills"]) == 3
    assert demand["basis"] == "Based on 5 signals from 5 articles in the last 30 days."
    assert "not from job postings" in demand["note"]


def test_basis_mentions_signals_without_skills(env):
    env.add_signal(company="A", skills=["Python"], article_id="a1")
    env.add_signal(company="B", skills=[], article_id="a2")

    db = env.db()
    demand = matching.skill_demand(db, days=30, now=NOW)
    db.close()

    assert demand["basis"] == "Based on 2 signals from 2 articles in the last 30 days. 1 of them names specific skills."
    assert demand["signals_with_skills"] == 1


def test_empty_database_gives_an_empty_demand_with_a_clear_basis(env):
    db = env.db()
    demand = matching.skill_demand(db, days=7, now=NOW)
    db.close()

    assert demand["skills"] == []
    assert demand["signals"] == 0
    assert demand["basis"] == "No hiring, expansion or new-office signals were found in the last 7 days."


# ---------------------------------------------------------------------------
# Gap for a profile
# ---------------------------------------------------------------------------

def test_profile_alias_py_matches_signal_python(env):
    env.add_signal(company="Acme", skills=["Python", "Docker"], article_id="a1")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=["py"]), days=30, now=NOW)
    db.close()

    assert gap["empty_profile"] is False
    assert [m["skill"] for m in gap["matched_skills"]] == ["Python"]
    assert gap["matched_skills"][0]["signal_count"] == 1
    assert [m["skill"] for m in gap["missing_skills"]] == ["Docker"]
    signal = gap["relevant_signals"][0]
    assert signal["matched_skills"] == ["Python"]
    assert signal["missing_skills"] == ["Docker"]


def test_profile_alias_matches_alias_stored_on_the_signal(env):
    env.add_signal(company="Acme", skills=["k8s"], article_id="a1")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=["Kubernetes"]), days=30, now=NOW)
    db.close()

    assert [m["skill"] for m in gap["matched_skills"]] == ["Kubernetes"]


def test_target_company_match_tcs_vs_full_name(env):
    env.add_signal(company="Tata Consultancy Services", skills=[], article_id="a1")
    env.add_signal(company="Other Corp", skills=["Rust"], article_id="a2")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=[], companies=["TCS"]), days=30, now=NOW)
    db.close()

    assert gap["empty_profile"] is False
    assert [s["company"] for s in gap["relevant_signals"]] == ["Tata Consultancy Services"]
    assert gap["relevant_signals"][0]["matches_target_company"] is True


def test_target_company_match_works_the_other_way_round(env):
    env.add_signal(company="TCS", skills=[], article_id="a1")

    db = env.db()
    gap = matching.gap_for_profile(
        db, _profile(companies=["Tata Consultancy Services"]), days=30, now=NOW
    )
    db.close()

    assert len(gap["relevant_signals"]) == 1


def test_missing_skills_are_the_demanded_skills_the_user_lacks_in_ranked_order(env):
    env.add_signal(company="A", skills=["Python", "SQL", "Docker"], article_id="a1")
    env.add_signal(company="B", skills=["Python", "SQL"], article_id="a2")
    env.add_signal(company="C", skills=["Python", "Rust"], article_id="a3")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=["SQL"]), days=30, now=NOW)
    db.close()

    # Demand order: Python (3 companies), SQL (2), then Docker and Rust (1 each, alphabetical).
    assert [m["skill"] for m in gap["missing_skills"]] == ["Python", "Docker", "Rust"]
    assert [m["skill"] for m in gap["matched_skills"]] == ["SQL"]
    assert "SQL" not in [m["skill"] for m in gap["missing_skills"]]


def test_missing_skills_respect_the_limit(env):
    env.add_signal(company="A", skills=[f"Skill{i}" for i in range(15)], article_id="a1")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=["Other"]), days=30, missing_limit=4, now=NOW)
    db.close()

    assert len(gap["missing_skills"]) == 4


def test_relevant_signals_exclude_unrelated_ones_and_put_target_companies_first(env):
    env.add_signal(company="Skill Match Systems", skills=["Python", "Go"], confidence=0.99, article_id="a1")
    env.add_signal(company="Unrelated Systems", skills=["Fortran"], article_id="a2")
    env.add_signal(company="Dream Systems", skills=["Java"], confidence=0.5, article_id="a3")

    db = env.db()
    gap = matching.gap_for_profile(
        db, _profile(skills=["Python"], companies=["Dream Systems"]), days=30, now=NOW
    )
    db.close()

    assert [s["company"] for s in gap["relevant_signals"]] == ["Dream Systems", "Skill Match Systems"]


def test_only_stored_skills_are_reported_nothing_is_invented(env):
    env.add_signal(company="Acme", skills=["Python"], article_id="a1")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=["Java"], companies=["Acme"]), days=30, now=NOW)
    db.close()

    assert [m["skill"] for m in gap["missing_skills"]] == ["Python"]
    assert gap["matched_skills"] == []
    assert gap["relevant_signals"][0]["skills"] == ["Python"]


def test_none_profile_and_blank_profile_are_empty(env):
    env.add_signal(company="Acme", skills=["Python"], article_id="a1")
    db = env.db()
    for profile in (None, _profile(), _profile(skills=["  "], companies=[""])):
        gap = matching.gap_for_profile(db, profile, days=30, now=NOW)
        assert gap["empty_profile"] is True
        assert gap["top_demand"][0]["skill"] == "Python"
        assert gap["relevant_signals"] == []
    db.close()


def test_evidence_quote_is_truncated_to_200_characters(env):
    env.add_signal(company="Acme", skills=["Python"], quote="x" * 500, article_id="a1")

    db = env.db()
    gap = matching.gap_for_profile(db, _profile(skills=["Python"]), days=30, now=NOW)
    db.close()

    quote = gap["relevant_signals"][0]["evidence_quote"]
    assert len(quote) == 200
    assert quote.endswith("\u2026")


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

def test_skill_demand_is_public_and_returns_ranked_skills(env):
    env.add_signal(company="A", skills=["Python"], when=_recent(), article_id="a1")

    response = env.anonymous.get("/api/career/skill-demand")

    assert response.status_code == 200
    body = response.json()
    assert body["skills"][0]["skill"] == "Python"
    assert body["days"] == 30
    assert body["basis"].startswith("Based on 1 signal from 1 article")
    assert "note" in body


def test_my_gap_requires_authentication(env):
    response = env.anonymous.get("/api/career/my-gap")
    assert response.status_code == 401


def test_my_gap_with_no_profile_returns_200_empty_profile_and_top_demand(env):
    env.add_signal(company="A", skills=["Python", "SQL"], when=_recent(), article_id="a1")

    response = env.client.get("/api/career/my-gap")

    assert response.status_code == 200
    body = response.json()
    assert body["empty_profile"] is True
    assert [s["skill"] for s in body["top_demand"]] == ["Python", "SQL"] or sorted(
        s["skill"] for s in body["top_demand"]
    ) == ["Python", "SQL"]
    assert body["matched_skills"] == []
    assert body["relevant_signals"] == []


def test_my_gap_uses_the_signed_in_users_own_profile_only(env):
    env.add_signal(company="Acme", skills=["Python", "Rust"], when=_recent(), article_id="a1")
    db = env.db()
    db.add(StudentProfile(user_id=ALICE, skills=["py"], target_companies=[], target_roles=[]))
    db.add(StudentProfile(user_id=BOB, skills=["Rust"], target_companies=[], target_roles=[]))
    db.commit()
    db.close()

    env.current["user_id"] = ALICE
    alice = env.client.get("/api/career/my-gap").json()
    env.current["user_id"] = BOB
    bob = env.client.get("/api/career/my-gap").json()

    assert [m["skill"] for m in alice["matched_skills"]] == ["Python"]
    assert [m["skill"] for m in alice["missing_skills"]] == ["Rust"]
    assert [m["skill"] for m in bob["matched_skills"]] == ["Rust"]
    assert [m["skill"] for m in bob["missing_skills"]] == ["Python"]


def test_my_gap_response_has_no_user_data(env):
    env.add_signal(company="Acme", skills=["Python"], when=_recent(), article_id="a1")
    db = env.db()
    db.add(StudentProfile(user_id=ALICE, skills=["Python"], target_companies=["Acme"], target_roles=[]))
    db.commit()
    db.close()

    text = env.client.get("/api/career/my-gap").text

    assert "alice@example.com" not in text
    assert "secret-hash" not in text
    assert ALICE not in text


def test_no_response_contains_body_text(env):
    secret = "FULL-ARTICLE-BODY-SENTINEL-DO-NOT-LEAK"
    env.add_signal(company="Acme", skills=["Python"], when=_recent(), article_id="a1")
    db = env.db()
    db.add(ArticleBody(article_id="a1", body_text=secret, status="ok", attempts=1))
    db.add(StudentProfile(user_id=ALICE, skills=["Python"], target_companies=["Acme"], target_roles=[]))
    db.commit()
    db.close()

    for url in ("/api/career/skill-demand", "/api/career/my-gap"):
        text = env.client.get(url).text
        assert "body_text" not in text
        assert secret not in text


def test_signal_cards_expose_only_the_allowed_article_fields(env):
    env.add_signal(
        company="Acme", skills=["Python"], when=_recent(), article_id="a1",
        title="Acme hiring", source="TechCrunch", link="https://example.com/a1", count=200,
    )
    db = env.db()
    db.add(StudentProfile(user_id=ALICE, skills=["Python"], target_companies=[], target_roles=[]))
    db.commit()
    db.close()

    signal = env.client.get("/api/career/my-gap").json()["relevant_signals"][0]

    assert set(signal["articles"][0]) == {"title", "link", "source"}
    assert signal["articles"][0] == {
        "title": "Acme hiring", "link": "https://example.com/a1", "source": "TechCrunch",
    }
    assert signal["count"] == 200
    assert signal["location"] == {"city": "Bengaluru", "country": "India"}
    assert {"company", "action", "timing", "skills", "confidence", "evidence_quote"} <= set(signal)


@pytest.mark.parametrize("query", [
    "days=0", "days=91", "days=-5", "days=abc", "limit=0", "limit=51", "limit=abc",
])
def test_invalid_skill_demand_parameters_return_422(env, query):
    assert env.anonymous.get(f"/api/career/skill-demand?{query}").status_code == 422


@pytest.mark.parametrize("query", ["days=0", "days=91", "days=abc"])
def test_invalid_my_gap_days_returns_422(env, query):
    assert env.client.get(f"/api/career/my-gap?{query}").status_code == 422


@pytest.mark.parametrize("query", ["days=1&limit=1", "days=90&limit=50", "days=7", "limit=20"])
def test_boundary_parameters_are_accepted(env, query):
    assert env.anonymous.get(f"/api/career/skill-demand?{query}").status_code == 200


# ---------------------------------------------------------------------------
# Wiring and page
# ---------------------------------------------------------------------------

def test_routes_and_page_are_wired_into_the_real_app():
    from backend.app.webapp import app

    client = TestClient(app)
    assert client.get("/api/career/my-gap").status_code == 401  # route exists and is protected
    page = client.get("/career")
    assert page.status_code == 200
    assert "CareerRadar" in page.text
    assert "inferred from news mentions" in page.text.lower()


def test_career_page_never_inserts_api_data_with_innerhtml():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", "career.html")
    html = open(path, encoding="utf-8").read()
    assert "innerHTML" not in html
    assert "insertAdjacentHTML" not in html


@pytest.mark.parametrize("template", ["index.html", "dashboard.html", "chat.html"])
def test_navbars_link_to_careerradar(template):
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", template)
    html = open(path, encoding="utf-8").read()
    assert 'href="/career"' in html
    assert "CareerRadar" in html
