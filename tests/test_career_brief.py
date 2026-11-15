"""CareerRadar weekly brief: facts, validated writer, template fallback, caching, API, script."""
import importlib.util
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.auth import get_current_active_user
from backend.app.database import get_db
from backend.app.models import (
    Article, ArticleBody, Base, CareerSignal, StudentProfile, User, WeeklyBrief,
)
from backend.app.routers import brief as brief_router_module
from backend.services.brief import facts as facts_module
from backend.services.brief import generator as generator_module
from backend.services.brief import service
from backend.services.brief.migrate import downgrade, upgrade
from backend.services.brief.resources import LEARNING_RESOURCES, links_for

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"
QUOTE = "A long supporting quote that must never be returned by brief endpoints."


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class FakeGroq:
    """Stands in for call_groq. Queue items: a dict/str (the model's answer), an Exception, or a
    callable(facts) -> dict. When the queue is empty, `default` is used."""

    def __init__(self):
        self.calls = []
        self.queue = []
        self.default = valid_output

    def __call__(self, api_key, payload, timeout=25, preferred_model=None):
        self.calls.append({"payload": payload, "model": preferred_model})
        item = self.queue.pop(0) if self.queue else self.default
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(self.facts(payload))
        return (item if isinstance(item, str) else json.dumps(item)), "fake-model"

    @staticmethod
    def facts(payload):
        content = payload["messages"][1]["content"]
        return json.loads(content.split("FACTS (JSON):\n", 1)[1])


def valid_output(facts):
    """A well-behaved writer answer built only from the facts it was given."""
    skills = [s["skill"] for s in (facts.get("missing_skills") or facts.get("demand_skills") or [])][:2]
    growing = [c["company"] for c in facts["market"]["growing"][:2]]
    return {
        "headline": f"Focus on {' and '.join(skills)} this week" if skills else "This week in hiring news",
        "summary": f"These skills are named in news mentions over the last {facts['window_days']} days.",
        "focus_skills": [{"skill": s, "why": "Named in hiring and expansion news mentions."} for s in skills],
        "watch_list": [f"{c}: hiring in news mentions" for c in growing],
        "caution": "This is inferred from news mentions, not job postings.",
    }


class Env:
    def __init__(self, session_factory, current):
        self.session_factory = session_factory
        self.current = current
        self._n = 0

    def db(self):
        return self.session_factory()

    def add_signal(self, *, company, action="HIRING", timing="current", skills=(), days_ago=1,
                   city=None, country=None, count=None, article_id=None):
        self._n += 1
        article_id = article_id or f"art-{self._n}"
        when = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=self._n % 5)
        db = self.db()
        try:
            db.add(Article(
                id=article_id, title=f"SECRET-TITLE {company} {article_id}",
                link=f"https://example.com/{article_id}", summary="s", source="TechCrunch",
                published=format_datetime(when), created_at=when,
            ))
            db.flush()
            db.add(CareerSignal(
                source_article_id=article_id, company=company, action=action, timing=timing,
                count=count, location_city=city, location_country=country,
                skills_mentioned=list(skills), confidence=0.9, evidence_quote=QUOTE,
            ))
            db.commit()
        finally:
            db.close()

    def seed_market(self):
        """5 current signals: Docker (3 companies), Kubernetes (2), Terraform (2, no learning link)."""
        self.add_signal(company="Acme Systems", skills=["Python", "Docker", "Kubernetes"], days_ago=1, count=100)
        self.add_signal(company="Beta Systems", skills=["Docker", "Terraform"], days_ago=2)
        self.add_signal(company="Gamma Systems", action="EXPANSION", skills=["Docker", "Kubernetes"], days_ago=2)
        self.add_signal(company="Delta Systems", action="LAYOFF", days_ago=3, count=300)
        self.add_signal(company="Epsilon Systems", action="NEW_OFFICE", skills=["Terraform", "Rust"], days_ago=3)

    def set_profile(self, user_id, skills=(), companies=()):
        db = self.db()
        try:
            db.add(StudentProfile(user_id=user_id, skills=list(skills), target_companies=list(companies), target_roles=[]))
            db.commit()
        finally:
            db.close()

    def briefs(self, user_id=None):
        db = self.db()
        try:
            q = db.query(WeeklyBrief)
            if user_id:
                q = q.filter(WeeklyBrief.user_id == user_id)
            return q.all()
        finally:
            db.close()


def _build_app(session_factory, current=None):
    app = FastAPI()
    app.include_router(brief_router_module.router)

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
def fake(monkeypatch):
    groq = FakeGroq()
    monkeypatch.setattr(generator_module, "call_groq", groq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("CAREER_LLM_MODEL", raising=False)
    return groq


@pytest.fixture
def env(fake):
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

    service.regenerate_limiter.reset()
    current = {"user_id": ALICE}
    environment = Env(session_factory, current)
    environment.engine = engine
    environment.client = TestClient(_build_app(session_factory, current))
    environment.anonymous = TestClient(_build_app(session_factory, None))
    try:
        yield environment
    finally:
        service.regenerate_limiter.reset()
        engine.dispose()
        os.remove(path)


def _profile(skills=(), companies=()):
    from types import SimpleNamespace
    return SimpleNamespace(skills=list(skills), target_companies=list(companies))


# ---------------------------------------------------------------------------
# Facts builder
# ---------------------------------------------------------------------------

def test_top_missing_skills_are_ranked_by_demand_and_exclude_skills_the_user_has(env):
    env.seed_market()
    db = env.db()
    facts = facts_module.build_facts(db, _profile(skills=["py", "Docker"]))
    db.close()

    names = [m["skill"] for m in facts["missing_skills"]]
    assert "Python" not in names and "Docker" not in names       # "py" aliases to Python
    assert names == ["Kubernetes", "Terraform", "Rust"]           # by companies, signals, then name
    kube = facts["missing_skills"][0]
    assert kube["demand_signals"] == 2 and kube["company_count"] == 2
    assert kube["companies"] == ["Acme Systems", "Gamma Systems"]


def test_missing_skills_are_capped_at_three_with_companies_and_sources_capped(env):
    env.seed_market()
    for i in range(5):
        env.add_signal(company=f"Extra {i} Systems", skills=["Docker"], days_ago=1)
    db = env.db()
    facts = facts_module.build_facts(db, _profile(skills=["Rust"]))
    db.close()

    assert len(facts["missing_skills"]) == 3
    docker = facts["missing_skills"][0]
    assert docker["skill"] == "Docker"
    assert len(docker["companies"]) == 3
    assert len(docker["sources"]) == 2
    assert all(set(s) == {"title", "link"} for s in docker["sources"])


def test_learning_links_exist_only_for_skills_with_an_entry(env):
    env.seed_market()
    db = env.db()
    facts = facts_module.build_facts(db, _profile(skills=["Python"]))
    db.close()

    by_skill = {m["skill"]: m for m in facts["missing_skills"]}
    assert by_skill["Docker"]["learning_links"] == links_for("Docker")
    assert by_skill["Docker"]["learning_links"][0]["url"].startswith("https://docs.docker.com")
    assert "Terraform" not in LEARNING_RESOURCES
    assert by_skill["Terraform"]["learning_links"] == []


def test_every_learning_link_is_an_https_url_with_one_or_two_entries():
    for skill, entries in LEARNING_RESOURCES.items():
        assert 1 <= len(entries) <= 2, skill
        for title, url in entries:
            assert title and url.startswith("https://"), (skill, url)
    assert links_for("No Such Skill") == []


def test_matched_skills_relevant_signals_and_market_snapshot(env):
    env.seed_market()
    db = env.db()
    facts = facts_module.build_facts(db, _profile(skills=["Python", "Kubernetes"], companies=["Acme Systems"]))
    db.close()

    assert [m["skill"] for m in facts["matched_skills"]] == ["Kubernetes", "Python"] or \
        sorted(m["skill"] for m in facts["matched_skills"]) == ["Kubernetes", "Python"]
    assert len(facts["relevant_signals"]) <= 3
    assert facts["relevant_signals"][0]["company"] == "Acme Systems"
    assert facts["relevant_signals"][0]["matches_target_company"] is True

    growing = {c["company"]: c for c in facts["market"]["growing"]}
    cutting = {c["company"]: c for c in facts["market"]["cutting"]}
    # Four companies are growing with 1 signal each; the snapshot keeps the top 3 (ties by name).
    assert list(growing) == ["Acme Systems", "Beta Systems", "Epsilon Systems"]
    assert set(cutting) == {"Delta Systems"}
    # Headcount only appears with "from X of Y signals"; absent when no signal stated a number.
    assert growing["Acme Systems"]["reported_hires"] == {"total": 100, "signals_with_count": 1, "signals": 1}
    assert growing["Beta Systems"]["reported_hires"] is None
    assert cutting["Delta Systems"]["reported_layoffs"] == {"total": 300, "signals_with_count": 1, "signals": 1}


def test_no_profile_gives_market_only_facts(env):
    env.seed_market()
    db = env.db()
    facts = facts_module.build_facts(db, None)
    db.close()

    assert facts["empty_profile"] is True
    assert "profile" not in facts
    assert "missing_skills" not in facts and "relevant_signals" not in facts and "matched_skills" not in facts
    assert len(facts["demand_skills"]) == 3
    assert facts["market"]["growing"]


def test_seven_day_window_is_used_when_there_are_at_least_five_signals(env):
    env.seed_market()                                   # 5 signals in the last 3 days
    db = env.db()
    facts = facts_module.build_facts(db, _profile(skills=["Python"]))
    db.close()

    assert facts["window_days"] == 7 and facts["window_fallback"] is False
    assert "uses the last 30 days" not in facts["basis"]


def test_thirty_day_window_fallback_is_stated_in_the_basis(env):
    env.add_signal(company="Acme Systems", skills=["Docker"], days_ago=1)
    env.add_signal(company="Beta Systems", skills=["Docker"], days_ago=1)
    env.add_signal(company="Old Systems", skills=["Go"], days_ago=20)
    db = env.db()
    facts = facts_module.build_facts(db, _profile(skills=["Python"]))
    db.close()

    assert facts["window_days"] == 30 and facts["window_fallback"] is True
    assert "The last 7 days had only 2 hiring or layoff signals" in facts["basis"]
    assert "uses the last 30 days" in facts["basis"]
    assert {m["skill"] for m in facts["missing_skills"]} == {"Docker", "Go"}


def test_llm_view_contains_no_urls_or_article_titles(env):
    env.seed_market()
    db = env.db()
    full = facts_module.build_facts(db, _profile(skills=["Python"], companies=["Acme Systems"]))
    db.close()

    view_text = json.dumps(facts_module.llm_view(full))
    assert "http" not in view_text
    assert "SECRET-TITLE" not in view_text
    assert "learning_links" not in view_text and '"sources"' not in view_text
    assert "http" in json.dumps(full)                   # the full facts do keep them for code to attach


def test_week_start_is_the_monday_of_the_iso_week():
    sunday = datetime(2026, 11, 15, 23, 59, tzinfo=timezone.utc)
    monday = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    assert facts_module.week_start_for(sunday).isoformat() == "2026-11-09"
    assert facts_module.week_start_for(monday).isoformat() == "2026-11-09"
    assert facts_module.week_start_for(sunday + timedelta(hours=2)).isoformat() == "2026-11-16"


# ---------------------------------------------------------------------------
# Validator and template
# ---------------------------------------------------------------------------

def _view(env, profile=None):
    db = env.db()
    try:
        return facts_module.llm_view(facts_module.build_facts(db, profile))
    finally:
        db.close()


def _good(**overrides):
    out = {
        "headline": "Focus on Docker and Kubernetes this week",
        "summary": "Docker is named in 3 signals in news mentions over the last 7 days.",
        "focus_skills": [{"skill": "Docker", "why": "Named in 3 hiring signals in news mentions."}],
        "watch_list": ["Acme Systems: hiring in news mentions"],
        "caution": "This is inferred from news mentions, not job postings.",
    }
    out.update(overrides)
    return out


def test_validator_accepts_a_brief_that_only_uses_the_facts(env):
    env.seed_market()
    view = _view(env, _profile(skills=["Python"]))
    assert generator_module.validate_brief(_good(), view) == []


@pytest.mark.parametrize("override,fragment", [
    ({"headline": "Docker appears in 47 signals"}, "number 47"),
    ({"summary": "Microsoft is hiring a lot this week."}, "Microsoft"),
    ({"summary": "You should also learn Angular."}, "Angular"),
    ({"summary": "See https://docs.docker.com/ for more."}, "URL"),
    ({"summary": "See docs.docker.com for more."}, "URL"),
    ({"summary": "Visit www.example.org today."}, "URL"),
    ({"summary": "Learn Docker in 3 weeks."}, "time estimate"),
    ({"summary": "Learn Docker in a few weeks."}, "time estimate"),
    ({"summary": "Give it two months."}, "time estimate"),
    ({"summary": "Expect a salary of 5 lakh."}, "money"),
    ({"summary": "Offers pay $100 on average."}, "money"),
    ({"summary": "Hiring peaked in November."}, "date"),
    ({"summary": "The Zebra platform is trending."}, "Zebra"),
    ({"focus_skills": [{"skill": "Terraformz", "why": "x"}]}, "not one of the missing"),
    ({"headline": "x" * 121}, "longer than 120"),
    ({"summary": "y" * 401}, "longer than 400"),
    ({"focus_skills": [{"skill": "Docker", "why": "z" * 201}]}, "why must be"),
    ({"watch_list": ["a", "b", "c", "d"]}, "watch_list"),
    ({"focus_skills": [{"skill": "Docker", "why": "a"}] * 4}, "focus_skills must be"),
    ({"caution": ""}, "caution"),
])
def test_validator_rejects_anything_not_in_the_facts(env, override, fragment):
    env.seed_market()
    view = _view(env, _profile(skills=["Python"]))
    problems = generator_module.validate_brief(_good(**override), view)
    assert problems, override
    assert any(fragment.lower() in p.lower() for p in problems), problems


def test_focus_skills_must_be_skills_the_user_could_learn_not_ones_they_have(env):
    env.seed_market()
    view = _view(env, _profile(skills=["Python"]))
    mine = generator_module.validate_brief(
        _good(focus_skills=[{"skill": "Python", "why": "You already know it."}]), view)
    unknown = generator_module.validate_brief(
        _good(focus_skills=[{"skill": "Angular", "why": "x"}]), view)
    assert any("not one of the missing or demanded skills" in p for p in mine)
    assert any("not one of the missing or demanded skills" in p for p in unknown)
    assert generator_module.validate_brief(
        _good(focus_skills=[{"skill": "Docker", "why": "Named in 3 signals."}]), view) == []


def test_validator_allows_the_window_length_but_not_other_durations(env):
    env.seed_market()
    view = _view(env, _profile(skills=["Python"]))        # window_days == 7
    ok = generator_module.validate_brief(_good(summary="Covers the last 7 days of news mentions."), view)
    bad = generator_module.validate_brief(_good(summary="Docker is named in 3 signals over 2 weeks."), view)
    assert ok == []
    assert any("time estimate" in p for p in bad)


def test_validator_rejects_non_objects_and_bad_shapes(env):
    env.seed_market()
    view = _view(env, None)
    assert generator_module.validate_brief([], view)
    assert generator_module.validate_brief({"headline": "x"}, view)
    assert generator_module.validate_brief(_good(watch_list="not a list"), view)


@pytest.mark.parametrize("with_profile", [True, False])
def test_the_template_brief_passes_its_own_validator(env, with_profile):
    env.seed_market()
    db = env.db()
    full = facts_module.build_facts(db, _profile(skills=["Python"], companies=["Acme Systems"]) if with_profile else None)
    db.close()

    raw = generator_module.build_template_raw(full)
    assert generator_module.validate_brief(raw, facts_module.llm_view(full)) == []
    assert len(raw["headline"]) <= 120 and len(raw["summary"]) <= 400 and len(raw["watch_list"]) <= 3


# ---------------------------------------------------------------------------
# Writer behaviour
# ---------------------------------------------------------------------------

def test_valid_llm_output_is_stored_with_source_llm_and_links_attached_by_code(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"], companies=["Acme Systems"])

    response = env.client.get("/api/career/brief")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert len(fake.calls) == 1
    content = body["content"]
    assert content["headline"].startswith("Focus on Docker and Kubernetes")
    docker = next(f for f in content["focus_skills"] if f["skill"] == "Docker")
    assert docker["learning_links"][0]["url"].startswith("https://docs.docker.com")   # from the static dict
    assert docker["demand_signals"] == 3 and docker["company_count"] == 3
    assert all(s["link"].startswith("https://example.com/") for s in docker["sources"])
    assert body["links"]["Docker"]["learning"] == docker["learning_links"]
    # Skill demand counts the 4 hiring/expansion/new-office signals (the layoff is not skill demand).
    assert body["basis"].startswith("Based on 4 signals from 4 articles in the last 7 days.")
    row = env.briefs(ALICE)[0]
    assert row.source == "llm" and row.model_used == "fake-model"


def test_the_stored_facts_are_exactly_what_the_writer_saw(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])

    env.client.get("/api/career/brief")

    sent = fake.facts(fake.calls[0]["payload"])
    assert env.briefs(ALICE)[0].facts == sent
    assert "http" not in json.dumps(sent)


def test_the_writer_cannot_add_links_or_other_keys(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    sneaky = valid_output(None) if False else None

    def answer(facts):
        out = valid_output(facts)
        out["learning_links"] = [{"title": "Evil", "url": "https://evil.example.net"}]
        out["focus_skills"][0]["url"] = "https://evil.example.net"
        return out

    fake.queue.append(answer)
    body = env.client.get("/api/career/brief").json()

    assert body["source"] == "llm"
    assert "evil.example.net" not in json.dumps(body)


def test_writer_prompt_and_settings(env, fake, monkeypatch):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    monkeypatch.setenv("CAREER_LLM_MODEL", "my-model")

    env.client.get("/api/career/brief")

    call = fake.calls[0]
    assert call["model"] == "my-model"
    system = call["payload"]["messages"][0]["content"]
    assert "ONLY the facts" in system and "news mentions" in system and "URL" in system
    assert "salaries" in system and "time estimates" in system


@pytest.mark.parametrize("bad_text", [
    "Docker appears in 47 signals this week",           # invented number
    "Microsoft is hiring Docker engineers",              # invented company
    "Read docs at https://docs.docker.com/get-started",  # invented URL
])
def test_invented_content_is_rejected_retried_once_then_the_template_is_used(env, fake, bad_text):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])

    def bad(facts):
        out = valid_output(facts)
        out["summary"] = bad_text
        return out

    fake.queue.extend([bad, bad])
    body = env.client.get("/api/career/brief").json()

    assert len(fake.calls) == 2                          # one retry, no more
    assert body["source"] == "template"
    assert bad_text not in json.dumps(body)
    assert env.briefs(ALICE)[0].source == "template" and env.briefs(ALICE)[0].model_used is None
    retry_prompt = fake.calls[1]["payload"]["messages"][-1]["content"]
    assert "rejected" in retry_prompt


def test_a_good_answer_on_the_retry_is_accepted(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    fake.queue.append(lambda facts: {**valid_output(facts), "summary": "Docker appears in 99 signals."})

    body = env.client.get("/api/career/brief").json()

    assert len(fake.calls) == 2
    assert body["source"] == "llm"


@pytest.mark.parametrize("junk", ["not json at all", "[]", '{"headline": "only a headline"}', ""])
def test_unparseable_or_malformed_answers_fall_back_to_the_template(env, fake, junk):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    fake.queue.extend([junk, junk])

    body = env.client.get("/api/career/brief").json()

    assert body["source"] == "template"
    assert len(fake.calls) == 2


def test_fenced_json_answers_are_accepted(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    fake.queue.append(lambda facts: "```json\n" + json.dumps(valid_output(facts)) + "\n```")

    assert env.client.get("/api/career/brief").json()["source"] == "llm"


def test_call_groq_raising_uses_the_template_without_retrying(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    fake.queue.append(RuntimeError("All Groq models exhausted"))

    response = env.client.get("/api/career/brief")

    assert response.status_code == 200
    assert response.json()["source"] == "template"
    assert len(fake.calls) == 1
    assert response.json()["content"]["headline"].startswith("Focus on")


def test_missing_api_key_uses_the_template_without_calling_the_llm(env, fake, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    env.seed_market()

    body = env.client.get("/api/career/brief").json()

    assert body["source"] == "template"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# API: caching, regenerate, empty profile, history
# ---------------------------------------------------------------------------

def test_a_second_get_in_the_same_week_does_not_call_the_llm_again(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])

    first = env.client.get("/api/career/brief").json()
    second = env.client.get("/api/career/brief").json()

    assert len(fake.calls) == 1
    assert first == second
    assert len(env.briefs(ALICE)) == 1


def test_a_new_week_creates_a_new_brief(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    db = env.db()
    user = db.query(User).filter(User.id == ALICE).one()
    now = datetime.now(timezone.utc)
    this_week = service.get_or_create_brief(db, user, now=now).week_start
    next_week = service.get_or_create_brief(db, user, now=now + timedelta(days=7)).week_start
    db.close()

    assert this_week != next_week
    assert (next_week - this_week).days == 7
    assert len(fake.calls) == 2
    assert len(env.briefs(ALICE)) == 2


def test_regenerate_replaces_the_row_and_calls_the_llm(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    first = env.client.get("/api/career/brief").json()
    fake.queue.append(lambda facts: {**valid_output(facts), "headline": "A brand new headline"})

    second = env.client.post("/api/career/brief/regenerate")

    assert second.status_code == 200
    assert second.json()["content"]["headline"] == "A brand new headline"
    assert second.json()["week_start"] == first["week_start"]
    assert len(fake.calls) == 2
    rows = env.briefs(ALICE)
    assert len(rows) == 1 and rows[0].content["headline"] == "A brand new headline"
    assert env.client.get("/api/career/brief").json()["content"]["headline"] == "A brand new headline"


def test_regenerate_is_rate_limited_to_five_per_hour(env, fake, monkeypatch):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(service.regenerate_limiter, "clock", lambda: clock["now"])

    statuses = [env.client.post("/api/career/brief/regenerate").status_code for _ in range(5)]
    blocked = env.client.post("/api/career/brief/regenerate")

    assert statuses == [200] * 5
    assert blocked.status_code == 429
    assert "at most 5 times per hour" in blocked.json()["detail"]
    assert int(blocked.headers["Retry-After"]) > 0
    calls_after_block = len(fake.calls)
    assert calls_after_block == 5                         # the blocked request never reached the LLM

    # The limit is per user.
    env.current["user_id"] = BOB
    assert env.client.post("/api/career/brief/regenerate").status_code == 200
    env.current["user_id"] = ALICE

    # And it expires.
    clock["now"] += 3601
    assert env.client.post("/api/career/brief/regenerate").status_code == 200


def test_get_does_not_count_toward_the_regenerate_limit(env, fake, monkeypatch):
    env.seed_market()
    monkeypatch.setattr(service.regenerate_limiter, "clock", lambda: 5.0)
    for _ in range(8):
        assert env.client.get("/api/career/brief").status_code == 200
    assert env.client.post("/api/career/brief/regenerate").status_code == 200


def test_empty_profile_returns_a_market_only_brief(env, fake):
    env.seed_market()

    response = env.client.get("/api/career/brief")

    assert response.status_code == 200
    body = response.json()
    assert body["content"]["empty_profile"] is True
    assert body["content"]["market"]["growing"]
    assert body["content"]["matched_skills"] == [] and body["content"]["relevant_signals"] == []
    stored = env.briefs(ALICE)[0].facts
    assert stored["empty_profile"] is True and "profile" not in stored
    assert "Career Profile" in body["content"]["summary"] or body["source"] == "llm"


def test_empty_profile_with_no_signals_still_returns_200(env, fake):
    body = env.client.get("/api/career/brief")
    assert body.status_code == 200
    assert body.json()["content"]["empty_profile"] is True
    assert body.json()["content"]["focus_skills"] == []


def test_history_lists_only_week_headline_and_source(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])
    db = env.db()
    user = db.query(User).filter(User.id == ALICE).one()
    now = datetime.now(timezone.utc)
    for weeks in range(3):
        service.get_or_create_brief(db, user, now=now - timedelta(days=7 * weeks))
    db.close()

    history = env.client.get("/api/career/brief/history").json()["briefs"]

    assert len(history) == 3
    assert [set(h) for h in history] == [{"week_start", "headline", "source"}] * 3
    assert [h["week_start"] for h in history] == sorted((h["week_start"] for h in history), reverse=True)
    assert len(env.client.get("/api/career/brief/history?limit=2").json()["briefs"]) == 2


@pytest.mark.parametrize("query", ["limit=0", "limit=27", "limit=abc"])
def test_history_limit_is_validated(env, query):
    assert env.client.get(f"/api/career/brief/history?{query}").status_code == 422


# ---------------------------------------------------------------------------
# Auth, isolation, cascade, privacy
# ---------------------------------------------------------------------------

def test_unauthenticated_requests_return_401(env, fake):
    assert env.anonymous.get("/api/career/brief").status_code == 401
    assert env.anonymous.post("/api/career/brief/regenerate").status_code == 401
    assert env.anonymous.get("/api/career/brief/history").status_code == 401
    assert fake.calls == [] and env.briefs() == []


def test_user_a_cannot_see_user_bs_brief(env, fake):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"], companies=["Acme Systems"])
    env.set_profile(BOB, skills=["Rust"], companies=["Epsilon Systems"])

    env.current["user_id"] = ALICE
    alice = env.client.get("/api/career/brief").json()
    env.current["user_id"] = BOB
    bob = env.client.get("/api/career/brief").json()
    bob_history = env.client.get("/api/career/brief/history").json()["briefs"]

    assert len(env.briefs()) == 2
    assert {r.user_id for r in env.briefs()} == {ALICE, BOB}
    assert [m["skill"] for m in alice["content"]["matched_skills"]] == ["Python"]
    assert [m["skill"] for m in bob["content"]["matched_skills"]] == ["Rust"]
    assert alice["content"]["relevant_signals"][0]["company"] == "Acme Systems"
    assert bob["content"]["relevant_signals"][0]["company"] == "Epsilon Systems"
    assert len(bob_history) == 1
    # Alice's profile never reaches Bob's response, and Bob regenerating leaves Alice's row alone.
    env.client.post("/api/career/brief/regenerate")
    alice_row = env.briefs(ALICE)[0]
    assert alice_row.generated_at is not None
    assert "Rust" not in json.dumps(alice_row.facts["profile"])


def test_deleting_the_user_cascades_their_briefs_at_the_database_level(env, fake):
    env.seed_market()
    env.client.get("/api/career/brief")
    assert len(env.briefs(ALICE)) == 1

    db = env.db()
    db.query(User).filter(User.id == ALICE).delete(synchronize_session=False)   # bypasses ORM cascade
    db.commit()
    db.close()

    assert env.briefs(ALICE) == []


def test_deleting_the_user_through_the_orm_removes_their_briefs(env, fake):
    env.seed_market()
    env.client.get("/api/career/brief")

    db = env.db()
    db.delete(db.query(User).filter(User.id == ALICE).one())
    db.commit()
    db.close()

    assert env.briefs(ALICE) == []


def test_one_brief_per_user_per_week_is_enforced_by_the_database(env, fake):
    from sqlalchemy.exc import IntegrityError

    env.seed_market()
    env.client.get("/api/career/brief")
    existing = env.briefs(ALICE)[0]

    db = env.db()
    db.add(WeeklyBrief(user_id=ALICE, week_start=existing.week_start, facts={}, content={}, source="template"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.close()


def test_no_response_contains_body_text_quotes_titles_or_user_data(env, fake):
    secret = "FULL-ARTICLE-BODY-SENTINEL-DO-NOT-LEAK"
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"], companies=["Acme Systems"])
    db = env.db()
    db.add(ArticleBody(article_id="art-1", body_text=secret, status="ok", attempts=1))
    db.commit()
    db.close()

    texts = [
        env.client.get("/api/career/brief").text,
        env.client.post("/api/career/brief/regenerate").text,
        env.client.get("/api/career/brief/history").text,
    ]
    for text in texts:
        assert "body_text" not in text and secret not in text
        assert "evidence_quote" not in text and QUOTE not in text
        assert "alice@example.com" not in text and "secret-hash" not in text and ALICE not in text
    # The writer never saw article titles either.
    assert all("SECRET-TITLE" not in json.dumps(c["payload"]) for c in fake.calls)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT NOT NULL)")
    conn.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
    conn.execute("INSERT INTO users VALUES ('u1', 'a@example.com'), ('u2', 'b@example.com')")
    conn.execute("INSERT INTO articles VALUES ('art1', 'Some title')")
    conn.commit()
    conn.close()
    try:
        yield path
    finally:
        os.remove(path)


def _tables(path):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def _columns(path, table):
    conn = sqlite3.connect(path)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_migration_creates_the_table_and_runs_twice_safely(temp_db):
    assert "weekly_briefs" not in _tables(temp_db)
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "INSERT INTO weekly_briefs (user_id, week_start, facts, content, source) "
        "VALUES ('u1', '2026-11-09', '{}', '{}', 'template')"
    )
    conn.commit()
    conn.close()

    upgrade(temp_db)
    upgrade(temp_db)

    conn = sqlite3.connect(temp_db)
    assert conn.execute("SELECT user_id, week_start, source FROM weekly_briefs").fetchall() == [
        ("u1", "2026-11-09", "template")
    ]
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
    conn.close()


def test_migration_enforces_uniqueness_and_cascade(temp_db):
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO weekly_briefs (user_id, week_start, facts, content, source) VALUES ('u1','2026-11-09','{}','{}','llm')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO weekly_briefs (user_id, week_start, facts, content, source) VALUES ('u1','2026-11-09','{}','{}','llm')")
    conn.execute("INSERT INTO weekly_briefs (user_id, week_start, facts, content, source) VALUES ('u2','2026-11-09','{}','{}','llm')")
    conn.commit()
    conn.execute("DELETE FROM users WHERE id = 'u1'")
    conn.commit()
    assert conn.execute("SELECT user_id FROM weekly_briefs").fetchall() == [("u2",)]
    conn.close()


def test_migration_has_an_index_and_matches_the_orm_columns(temp_db):
    upgrade(temp_db)
    conn = sqlite3.connect(temp_db)
    indexes = {r[1] for r in conn.execute("PRAGMA index_list(weekly_briefs)")}
    conn.close()
    assert "ix_weekly_briefs_user_id" in indexes
    assert {c.name for c in WeeklyBrief.__table__.columns} == _columns(temp_db, "weekly_briefs")


def test_downgrade_drops_only_the_brief_table(temp_db):
    upgrade(temp_db)
    downgrade(temp_db)
    assert "weekly_briefs" not in _tables(temp_db)
    assert {"users", "articles"} <= _tables(temp_db)


def test_engine_path_creates_the_table_and_is_idempotent(monkeypatch):
    from sqlalchemy import inspect, text
    import backend.app.database as database

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

        assert "weekly_briefs" in set(inspect(engine).get_table_names())
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar() == 1
    finally:
        engine.dispose()
        os.remove(path)


# ---------------------------------------------------------------------------
# Batch script
# ---------------------------------------------------------------------------

def _load_script():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_weekly_briefs.py")
    spec = importlib.util.spec_from_file_location("generate_weekly_briefs_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_script(env, monkeypatch, argv):
    module = _load_script()
    monkeypatch.setattr(module, "SessionLocal", env.session_factory)
    monkeypatch.setattr(module, "init_database", lambda: None)
    monkeypatch.setenv("CAREER_LLM_CALL_DELAY_SECONDS", "0")
    monkeypatch.setattr("sys.argv", ["generate_weekly_briefs.py"] + argv)
    return module.main()


def test_script_generates_for_one_user_and_skips_existing_unless_forced(env, fake, monkeypatch, capsys):
    env.seed_market()
    env.set_profile(ALICE, skills=["Python"])

    assert _run_script(env, monkeypatch, ["--user-email", "ALICE@example.com"]) == 0
    assert len(env.briefs(ALICE)) == 1 and len(fake.calls) == 1
    assert "[llm] alice@example.com" in capsys.readouterr().out

    assert _run_script(env, monkeypatch, ["--user-email", "alice@example.com"]) == 0
    assert len(fake.calls) == 1                                  # skipped, no new LLM call
    assert "[skipped]" in capsys.readouterr().out

    assert _run_script(env, monkeypatch, ["--user-email", "alice@example.com", "--force"]) == 0
    assert len(fake.calls) == 2 and len(env.briefs(ALICE)) == 1  # replaced in place


def test_script_all_covers_every_active_user_sequentially(env, fake, monkeypatch, capsys):
    env.seed_market()
    db = env.db()
    db.add(User(id="33333333-3333-3333-3333-333333333333", email="inactive@example.com",
                hashed_password="h", is_active=False))
    db.commit()
    db.close()

    assert _run_script(env, monkeypatch, ["--all"]) == 0

    assert {r.user_id for r in env.briefs()} == {ALICE, BOB}
    assert len(fake.calls) == 2
    assert "Done: 2 generated, 0 skipped." in capsys.readouterr().out


def test_script_reports_an_unknown_user(env, fake, monkeypatch, capsys):
    assert _run_script(env, monkeypatch, ["--user-email", "nobody@example.com"]) == 1
    assert "no user with email" in capsys.readouterr().err
    assert fake.calls == []


def test_script_waits_between_users_using_the_configured_delay(env, fake, monkeypatch):
    env.seed_market()
    module = _load_script()
    monkeypatch.setattr(module, "SessionLocal", env.session_factory)
    monkeypatch.setattr(module, "init_database", lambda: None)
    monkeypatch.setenv("CAREER_LLM_CALL_DELAY_SECONDS", "2.5")
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("sys.argv", ["generate_weekly_briefs.py", "--all"])

    assert module.main() == 0

    assert sleeps == [2.5]                                       # between the two users, not after the last


def test_brief_generation_is_not_wired_into_the_scrape_job():
    path = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "webapp.py")
    source = open(path, encoding="utf-8").read()
    scrape = source[source.index("def _run_scrape_sync"):source.index("async def scheduled_news_scrape_job")]
    assert "brief" not in scrape.lower()


# ---------------------------------------------------------------------------
# Wiring and page
# ---------------------------------------------------------------------------

def test_routes_are_wired_into_the_real_app():
    from backend.app.webapp import app

    client = TestClient(app)
    assert client.get("/api/career/brief").status_code == 401
    assert client.post("/api/career/brief/regenerate").status_code == 401
    assert client.get("/api/career/brief/history").status_code == 401
    page = client.get("/career")
    assert page.status_code == 200 and "This Week" in page.text


def _page():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", "career.html")
    return open(path, encoding="utf-8").read()


def test_this_week_tab_has_the_required_elements():
    html = _page()
    assert 'id="tabBtnWeek"' in html and 'id="panelWeek"' in html
    for marker in ("AI-written", "Auto template", "Regenerate", "Past briefs", "Skills to focus on",
                   "Watch list", "Market snapshot", "Inferred from news mentions",
                   "/api/career/brief/regenerate", "/api/career/brief/history", "#week"):
        assert marker in html, marker
    assert 'a.rel = \'noopener noreferrer\'' in html and "a.target = '_blank'" in html
    assert "/dashboard" in html                                   # empty-profile call to action
    assert "Log in" in html                                       # logged-out call to action


def test_this_week_tab_never_inserts_api_data_with_innerhtml():
    html = _page()
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "document.write" not in html


def test_existing_tabs_are_unchanged():
    html = _page()
    for marker in ('id="demandContent"', 'id="gapContent"', 'id="marketMap"', 'id="boardContent"',
                   'id="trendBox"', "Skills in demand", "Hiring vs Layoffs", "Hiring / expanding"):
        assert marker in html, marker


def test_page_has_no_window_property_that_collides_with_an_element_id():
    import re

    html = _page()
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    names = set(re.findall(r"\bwindow\.([A-Za-z_$][\w$]*)", html))
    assert ids & names == set()
