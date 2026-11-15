"""Student profile API: GET/PUT/DELETE /api/profile, validation, isolation and cascade."""
import os
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.auth import get_current_active_user
from backend.app.database import get_db
from backend.app.models import Base, StudentProfile, User
from backend.app.routers import profile as profile_module

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


class Env:
    """A temp SQLite DB, an app with only the profile router, and a switchable current user."""

    def __init__(self, engine, session_factory, current):
        self.engine = engine
        self.session_factory = session_factory
        self.current = current

    def db(self):
        return self.session_factory()

    def rows(self):
        db = self.db()
        try:
            return db.query(StudentProfile).all()
        finally:
            db.close()


def _build_app(session_factory, current=None):
    app = FastAPI()
    app.include_router(profile_module.router)
    app.state.limiter = profile_module.limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
        User(id=ALICE, email="alice@example.com", hashed_password="secret-hash-a", full_name="Alice"),
        User(id=BOB, email="bob@example.com", hashed_password="secret-hash-b", full_name="Bob"),
    ])
    db.commit()
    db.close()

    profile_module.limiter.reset()
    current = {"user_id": ALICE}
    app = _build_app(session_factory, current)
    environment = Env(engine, session_factory, current)
    environment.client = TestClient(app)
    environment.app_factory = lambda with_user=True: _build_app(
        session_factory, current if with_user else None
    )
    try:
        yield environment
    finally:
        profile_module.limiter.reset()
        engine.dispose()
        if os.path.exists(path):
            os.remove(path)


def _full_payload(**overrides):
    payload = {
        "branch": "Computer Science",
        "college": "NIT Kurukshetra",
        "year_of_study": 4,
        "graduation_year": 2027,
        "skills": ["Python", "SQL"],
        "target_companies": ["Google"],
        "target_roles": ["Backend Developer"],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# GET / PUT / DELETE behaviour
# ---------------------------------------------------------------------------

def test_get_without_profile_returns_empty_default_and_creates_no_row(env):
    response = env.client.get("/api/profile")

    assert response.status_code == 200
    assert response.json() == {
        "branch": None,
        "college": None,
        "year_of_study": None,
        "graduation_year": None,
        "skills": [],
        "target_companies": [],
        "target_roles": [],
        "updated_at": None,
    }
    assert env.rows() == []


def test_put_then_get_round_trip(env):
    put = env.client.put("/api/profile", json=_full_payload())
    assert put.status_code == 200

    got = env.client.get("/api/profile").json()

    assert got["branch"] == "Computer Science"
    assert got["college"] == "NIT Kurukshetra"
    assert got["year_of_study"] == 4
    assert got["graduation_year"] == 2027
    assert got["skills"] == ["Python", "SQL"]
    assert got["target_companies"] == ["Google"]
    assert got["target_roles"] == ["Backend Developer"]
    assert got["updated_at"] is not None


def test_put_twice_updates_the_same_row(env):
    env.client.put("/api/profile", json=_full_payload())
    second = env.client.put(
        "/api/profile",
        json=_full_payload(branch="Electronics", skills=["Rust"], target_roles=[]),
    )

    assert second.status_code == 200
    rows = env.rows()
    assert len(rows) == 1
    assert rows[0].user_id == ALICE
    assert rows[0].branch == "Electronics"
    assert rows[0].skills == ["Rust"]
    assert rows[0].target_roles == []  # PUT is a full replace, not a merge


def test_put_with_only_some_fields_clears_the_rest(env):
    env.client.put("/api/profile", json=_full_payload())
    env.client.put("/api/profile", json={"branch": "Mechanical"})

    got = env.client.get("/api/profile").json()
    assert got["branch"] == "Mechanical"
    assert got["college"] is None
    assert got["year_of_study"] is None
    assert got["skills"] == []


def test_delete_removes_the_row_and_is_idempotent(env):
    env.client.put("/api/profile", json=_full_payload())
    assert len(env.rows()) == 1

    first = env.client.delete("/api/profile")
    second = env.client.delete("/api/profile")

    assert first.status_code == 204
    assert second.status_code == 204
    assert env.rows() == []
    assert env.client.get("/api/profile").json()["skills"] == []


def test_response_contains_only_profile_fields(env):
    env.client.put("/api/profile", json=_full_payload())
    body = env.client.get("/api/profile").json()

    assert set(body) == {
        "branch", "college", "year_of_study", "graduation_year",
        "skills", "target_companies", "target_roles", "updated_at",
    }
    text = str(body).lower()
    assert "alice@example.com" not in text
    assert "secret-hash" not in text
    assert ALICE not in text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_skills_are_trimmed_deduped_case_insensitively_and_aliases_normalized(env):
    skills = [
        " js ", "JavaScript", "javascript", "", "   ",
        "py", "python3", "Golang", "React.js", "reactjs", "NodeJS", "node",
        "ML", "k8s", "Postgres", "Figma", "figma",
    ]
    response = env.client.put("/api/profile", json=_full_payload(skills=skills))

    assert response.status_code == 200
    assert response.json()["skills"] == [
        "JavaScript", "Python", "Go", "React", "Node.js",
        "Machine Learning", "Kubernetes", "PostgreSQL", "Figma",
    ]


def test_unknown_skill_keeps_its_cleaned_original_spelling(env):
    response = env.client.put("/api/profile", json=_full_payload(skills=["  Apache   Kafka "]))
    assert response.json()["skills"] == ["Apache Kafka"]


def test_skill_limits(env):
    fifty = [f"skill{i}" for i in range(50)]
    assert env.client.put("/api/profile", json=_full_payload(skills=fifty)).status_code == 200

    fifty_one = [f"skill{i}" for i in range(51)]
    assert env.client.put("/api/profile", json=_full_payload(skills=fifty_one)).status_code == 422

    assert env.client.put("/api/profile", json=_full_payload(skills=["x" * 40])).status_code == 200
    assert env.client.put("/api/profile", json=_full_payload(skills=["x" * 41])).status_code == 422


def test_duplicates_do_not_count_toward_the_skill_limit(env):
    skills = ["Python"] * 80
    response = env.client.put("/api/profile", json=_full_payload(skills=skills))
    assert response.status_code == 200
    assert response.json()["skills"] == ["Python"]


def test_company_and_role_limits(env):
    twenty = [f"Company {i}" for i in range(20)]
    assert env.client.put("/api/profile", json=_full_payload(target_companies=twenty)).status_code == 200
    twenty_one = [f"Company {i}" for i in range(21)]
    assert env.client.put("/api/profile", json=_full_payload(target_companies=twenty_one)).status_code == 422

    ten = [f"Role {i}" for i in range(10)]
    assert env.client.put("/api/profile", json=_full_payload(target_roles=ten)).status_code == 200
    eleven = [f"Role {i}" for i in range(11)]
    assert env.client.put("/api/profile", json=_full_payload(target_roles=eleven)).status_code == 422

    assert env.client.put("/api/profile", json=_full_payload(target_roles=["r" * 60])).status_code == 200
    assert env.client.put("/api/profile", json=_full_payload(target_roles=["r" * 61])).status_code == 422


@pytest.mark.parametrize("year", [0, 7, -1, "3", 3.5, True])
def test_bad_year_of_study_is_rejected(env, year):
    response = env.client.put("/api/profile", json=_full_payload(year_of_study=year))
    assert response.status_code == 422


@pytest.mark.parametrize("year", [1, 6])
def test_year_of_study_bounds_are_accepted(env, year):
    assert env.client.put("/api/profile", json=_full_payload(year_of_study=year)).status_code == 200


@pytest.mark.parametrize("year", [2019, 2041, "2027", 2027.5])
def test_bad_graduation_year_is_rejected(env, year):
    response = env.client.put("/api/profile", json=_full_payload(graduation_year=year))
    assert response.status_code == 422


@pytest.mark.parametrize("year", [2020, 2040])
def test_graduation_year_bounds_are_accepted(env, year):
    assert env.client.put("/api/profile", json=_full_payload(graduation_year=year)).status_code == 200


def test_unknown_fields_are_rejected(env):
    response = env.client.put("/api/profile", json=_full_payload(is_admin=True))
    assert response.status_code == 422
    assert env.rows() == []


def test_skills_must_be_a_list_of_strings(env):
    assert env.client.put("/api/profile", json=_full_payload(skills="python")).status_code == 422
    assert env.client.put("/api/profile", json=_full_payload(skills=["ok", 5])).status_code == 422


def test_control_characters_are_stripped_from_all_strings(env):
    response = env.client.put(
        "/api/profile",
        json=_full_payload(
            branch="C\u0007S\u200bE",
            college="NIT\u0000 Kurukshetra\ttest",
            skills=["Py\u200bthon", "Dock\u0001er"],
            target_companies=["Goo\u0008gle"],
            target_roles=["Back\u0000end\nDeveloper"],
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["branch"] == "CSE"
    assert body["college"] == "NIT Kurukshetra test"
    assert body["skills"] == ["Python", "Docker"]
    assert body["target_companies"] == ["Google"]
    assert body["target_roles"] == ["Backend Developer"]


def test_blank_branch_and_college_become_null(env):
    response = env.client.put("/api/profile", json=_full_payload(branch="   ", college=""))
    assert response.json()["branch"] is None
    assert response.json()["college"] is None


def test_tcs_and_full_name_become_the_same_canonical_company(env):
    response = env.client.put(
        "/api/profile",
        json=_full_payload(target_companies=["TCS", "Tata Consultancy Services", "tcs ltd"]),
    )

    assert response.status_code == 200
    assert response.json()["target_companies"] == ["Tata Consultancy Services"]


def test_companies_use_the_careerradar_canonical_names(env):
    from backend.services.career.normalizer import normalize_company

    names = ["TCS", "Infosys Ltd", "Acme Robotics Pvt Ltd"]
    response = env.client.put("/api/profile", json=_full_payload(target_companies=names))

    assert response.json()["target_companies"] == [normalize_company(n).canonical for n in names]


# ---------------------------------------------------------------------------
# Auth, isolation, cascade, rate limit, wiring
# ---------------------------------------------------------------------------

def test_unauthenticated_requests_return_401(env):
    client = TestClient(env.app_factory(with_user=False))

    assert client.get("/api/profile").status_code == 401
    assert client.put("/api/profile", json=_full_payload()).status_code == 401
    assert client.delete("/api/profile").status_code == 401
    assert env.rows() == []


def test_one_user_cannot_read_or_overwrite_another_users_profile(env):
    env.current["user_id"] = ALICE
    env.client.put("/api/profile", json=_full_payload(branch="Alice Branch", skills=["Python"]))

    env.current["user_id"] = BOB
    bobs_view = env.client.get("/api/profile").json()
    assert bobs_view["branch"] is None
    assert bobs_view["skills"] == []

    env.client.put("/api/profile", json=_full_payload(branch="Bob Branch", skills=["Rust"]))
    env.client.delete("/api/profile")  # Bob deleting must not touch Alice's row

    env.current["user_id"] = ALICE
    alices_view = env.client.get("/api/profile").json()
    assert alices_view["branch"] == "Alice Branch"
    assert alices_view["skills"] == ["Python"]
    assert [row.user_id for row in env.rows()] == [ALICE]


def test_two_users_get_separate_rows(env):
    env.current["user_id"] = ALICE
    env.client.put("/api/profile", json=_full_payload(branch="A"))
    env.current["user_id"] = BOB
    env.client.put("/api/profile", json=_full_payload(branch="B"))

    assert sorted((r.user_id, r.branch) for r in env.rows()) == [(ALICE, "A"), (BOB, "B")]


def test_deleting_the_user_cascades_the_profile_at_the_database_level(env):
    env.client.put("/api/profile", json=_full_payload())
    assert len(env.rows()) == 1

    db = env.db()
    # Bulk delete bypasses ORM cascades, so this proves ON DELETE CASCADE works in the DB.
    db.query(User).filter(User.id == ALICE).delete(synchronize_session=False)
    db.commit()
    db.close()

    assert env.rows() == []


def test_deleting_the_user_through_the_orm_removes_the_profile(env):
    env.client.put("/api/profile", json=_full_payload())

    db = env.db()
    db.delete(db.query(User).filter(User.id == ALICE).one())
    db.commit()
    db.close()

    assert env.rows() == []


def test_put_is_rate_limited(env):
    statuses = [env.client.put("/api/profile", json=_full_payload()).status_code for _ in range(31)]

    assert statuses[:30] == [200] * 30
    assert statuses[30] == 429


def test_router_is_included_in_the_real_app():
    from backend.app.webapp import app

    # Without a token the real auth dependency answers 401, so the route exists and is protected.
    response = TestClient(app).get("/api/profile")
    assert response.status_code == 401
