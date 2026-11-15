"""CareerRadar market views: geo normalization, company board, heatmap and weekly trend."""
import os
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.database import get_db
from backend.app.models import Article, ArticleBody, Base, CareerSignal
from backend.app.routers import career as career_router_module
from backend.services.career import geo, market, matching

# Fixed "now": Sunday 2026-11-15 (ISO week 46 runs Mon Nov 9 to Sun Nov 15; week 45 is Nov 2 to 8).
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
W45_TUE = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
W45_WED = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
W46_TUE = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
W44_TUE = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)


class Env:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self._counter = 0

    def db(self):
        return self.session_factory()

    def add_signal(
        self,
        *,
        company="Acme Robotics",
        action="HIRING",
        timing="current",
        city="Bengaluru",
        country="India",
        count=None,
        confidence=0.9,
        when=W45_TUE,
        article_id=None,
        title=None,
        source="TechCrunch",
        link=None,
    ):
        self._counter += 1
        article_id = article_id or f"art-{self._counter}"
        db = self.db()
        try:
            if db.query(Article).filter(Article.id == article_id).first() is None:
                db.add(Article(
                    id=article_id,
                    title=title or f"Article {article_id}",
                    link=link or f"https://example.com/{article_id}",
                    summary="s",
                    source=source,
                    published=format_datetime(when),
                    created_at=when,
                ))
                db.flush()
            signal = CareerSignal(
                source_article_id=article_id,
                company=company,
                action=action,
                timing=timing,
                count=count,
                location_city=city,
                location_country=country,
                skills_mentioned=[],
                confidence=confidence,
                evidence_quote="A long supporting quote that must never be returned by market endpoints.",
            )
            db.add(signal)
            db.commit()
            return signal.id
        finally:
            db.close()


def _build_app(session_factory):
    app = FastAPI()
    app.include_router(career_router_module.router)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
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
    environment = Env(session_factory)
    environment.client = TestClient(_build_app(session_factory))   # no auth override: endpoints are public
    try:
        yield environment
    finally:
        engine.dispose()
        os.remove(path)


def _last_week(day_offset):
    """A time inside last week (Monday + offset), safely within every default window."""
    now = datetime.now(timezone.utc)
    monday_this_week = (now - timedelta(days=now.weekday())).replace(hour=9, minute=0, second=0, microsecond=0)
    return monday_this_week - timedelta(days=7) + timedelta(days=day_offset)


# ---------------------------------------------------------------------------
# geo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("India", "India"), ("IN", "India"), ("in", "India"), ("Republic of India", "India"),
    ("USA", "United States"), ("U.S.", "United States"), ("US", "United States"),
    ("United States of America", "United States"),
    ("UK", "United Kingdom"), ("U.K.", "United Kingdom"), ("Great Britain", "United Kingdom"),
    ("UAE", "United Arab Emirates"), ("singapore", "Singapore"), ("Germany", "Germany"),
    ("Canada", "Canada"), ("Australia", "Australia"),
    ("  Atlantis  ", "Atlantis"),            # unknown keeps the cleaned original
    ("Narnia\u200b", "Narnia"),
])
def test_normalize_country(raw, expected):
    assert geo.normalize_country(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_normalize_country_empty_is_none(raw):
    assert geo.normalize_country(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("Bangalore", "Bengaluru"), ("bengaluru", "Bengaluru"), ("Gurgaon", "Gurugram"),
    ("Bombay", "Mumbai"), ("Madras", "Chennai"), ("Calcutta", "Kolkata"),
    ("New Delhi", "Delhi"), ("Delhi", "Delhi"), ("Pune", "Pune"), ("Hyderabad", "Hyderabad"),
    ("Noida", "Noida"), ("NYC", "New York"), ("Bengaluru, Karnataka", "Bengaluru"),
    ("Springfield", "Springfield"),           # unknown keeps the cleaned original
])
def test_normalize_city(raw, expected):
    assert geo.normalize_city(raw) == expected


def test_coordinate_tables_cover_india_and_global_cities():
    indian = [c for c, country in geo.CITY_COUNTRY.items() if country == "India"]
    global_ = [c for c, country in geo.CITY_COUNTRY.items() if country != "India"]
    assert len(indian) >= 40
    assert len(global_) >= 30
    for city in ("San Francisco", "Seattle", "Austin", "New York", "London", "Berlin",
                 "Singapore", "Dubai", "Toronto", "Sydney"):
        assert city in geo.CITY_COORDS
    for lat, lng in geo.CITY_COORDS.values():
        assert -90 <= lat <= 90 and -180 <= lng <= 180


def test_resolve_location_rules():
    known = geo.resolve_location("Bangalore", None)
    assert (known.city, known.country, known.precision) == ("Bengaluru", "India", "city")
    assert known.country_inferred is True and known.plottable

    unknown_city = geo.resolve_location("Springfield", "India")
    assert unknown_city.lat is None and unknown_city.lng is None and not unknown_city.plottable
    assert unknown_city.stated is True

    country_only = geo.resolve_location(None, "USA")
    assert country_only.precision == "country" and country_only.country == "United States"
    assert (country_only.lat, country_only.lng) == geo.COUNTRY_CENTROIDS["United States"]

    conflicting = geo.resolve_location("Hyderabad", "Pakistan")   # same name, different place
    assert not conflicting.plottable

    nothing = geo.resolve_location(None, None)
    assert nothing.stated is False and not nothing.plottable

    unknown_country = geo.resolve_location(None, "Atlantis")
    assert unknown_country.stated is True and not unknown_country.plottable


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def test_heatmap_plots_known_cities_with_aggregated_counts(env):
    env.add_signal(company="A", city="Bangalore", country="IN", article_id="a1")
    env.add_signal(company="B", city="Bengaluru", country="India", action="LAYOFF", article_id="a2")
    env.add_signal(company="C", city="Pune", country="India", article_id="a3")

    db = env.db()
    data = market.heatmap_points(db, days=30, now=NOW)
    db.close()

    by_city = {p["city"]: p for p in data["points"]}
    assert set(by_city) == {"Bengaluru", "Pune"}
    bengaluru = by_city["Bengaluru"]
    assert bengaluru["signal_count"] == 2
    assert (bengaluru["growth_count"], bengaluru["cuts_count"]) == (1, 1)
    assert bengaluru["kind"] == "mixed"
    assert bengaluru["precision"] == "city"
    assert (bengaluru["lat"], bengaluru["lng"]) == geo.CITY_COORDS["Bengaluru"]
    assert bengaluru["companies"] == ["A", "B"]
    assert by_city["Pune"]["kind"] == "growth"


def test_unknown_city_has_no_coordinates_and_is_listed_as_unmapped(env):
    env.add_signal(company="A", city="Springfield", country="India", article_id="a1")
    env.add_signal(company="B", city="Pune", country="India", article_id="a2")

    db = env.db()
    data = market.heatmap_points(db, days=30, now=NOW)
    db.close()

    assert [p["city"] for p in data["points"]] == ["Pune"]
    assert data["unmapped_locations"] == [{"city": "Springfield", "country": "India", "signal_count": 1}]
    assert all(p["lat"] is not None and p["lng"] is not None for p in data["points"])


def test_signals_without_location_are_counted_and_not_plotted(env):
    env.add_signal(company="A", city=None, country=None, article_id="a1")
    env.add_signal(company="B", city=None, country=None, article_id="a2")
    env.add_signal(company="C", city="Pune", country="India", article_id="a3")

    db = env.db()
    data = market.heatmap_points(db, days=30, now=NOW)
    in_india = market.heatmap_points(db, days=30, country="India", now=NOW)
    db.close()

    assert data["location_not_stated"] == 2
    assert [p["city"] for p in data["points"]] == ["Pune"]
    assert data["signals"] == 1
    assert "2 more signals in this period state no location and are not plotted." in data["basis"]
    assert in_india["location_not_stated"] == 2        # still reported under a country filter


def test_single_unlocated_signal_uses_singular_wording(env):
    env.add_signal(company="A", city=None, country=None, article_id="a1")

    db = env.db()
    basis = market.heatmap_points(db, days=30, now=NOW)["basis"]
    db.close()

    assert "1 more signal in this period states no location and is not plotted." in basis


def test_country_only_signal_is_plotted_at_the_country_centroid(env):
    env.add_signal(company="A", city=None, country="USA", article_id="a1")

    db = env.db()
    data = market.heatmap_points(db, days=30, now=NOW)
    db.close()

    point = data["points"][0]
    assert point["precision"] == "country"
    assert point["city"] is None and point["country"] == "United States"
    assert (point["lat"], point["lng"]) == geo.COUNTRY_CENTROIDS["United States"]


def test_unknown_country_only_signal_is_unmapped_not_plotted(env):
    env.add_signal(company="A", city=None, country="Atlantis", article_id="a1")

    db = env.db()
    data = market.heatmap_points(db, days=30, now=NOW)
    db.close()

    assert data["points"] == []
    assert data["unmapped_locations"] == [{"city": None, "country": "Atlantis", "signal_count": 1}]


def test_heatmap_group_filter(env):
    env.add_signal(company="A", action="HIRING", article_id="a1")
    env.add_signal(company="B", action="LAYOFF", article_id="a2")
    env.add_signal(company="C", action="HIRING_FREEZE", article_id="a3")

    db = env.db()
    growth = market.heatmap_points(db, days=30, action_group="growth", now=NOW)
    cuts = market.heatmap_points(db, days=30, action_group="cuts", now=NOW)
    everything = market.heatmap_points(db, days=30, action_group="all", now=NOW)
    db.close()

    assert growth["points"][0]["companies"] == ["A"] and growth["points"][0]["kind"] == "growth"
    assert cuts["points"][0]["companies"] == ["B", "C"] and cuts["points"][0]["kind"] == "cuts"
    assert everything["points"][0]["signal_count"] == 3


def test_heatmap_points_cap_companies_at_five(env):
    for i in range(8):
        env.add_signal(company=f"Company {i}", article_id=f"a{i}")

    db = env.db()
    point = market.heatmap_points(db, days=30, now=NOW)["points"][0]
    db.close()

    assert point["company_count"] == 8
    assert len(point["companies"]) == 5


def test_country_filter_india_excludes_other_countries(env):
    env.add_signal(company="Indian Systems", city="Bengaluru", country="India", article_id="a1")
    env.add_signal(company="American Systems", city="Seattle", country="United States", article_id="a2")
    env.add_signal(company="British Systems", city=None, country="UK", article_id="a3")

    db = env.db()
    india = market.heatmap_points(db, days=30, country="India", now=NOW)
    glob = market.heatmap_points(db, days=30, country=None, now=NOW)
    board_india = market.company_board(db, days=30, country="India", now=NOW)
    trend_india = market.weekly_trend(db, weeks=4, country="India", now=NOW)
    db.close()

    assert [p["city"] for p in india["points"]] == ["Bengaluru"]
    assert india["excluded_by_country_filter"] == 2
    assert india["country"] == "India"
    assert len(glob["points"]) == 3
    assert [c["company"] for c in board_india["companies"]] == ["Indian Systems"]
    assert trend_india["totals"]["growth"] == 1


def test_city_only_signal_counts_as_its_cities_country(env):
    env.add_signal(company="A", city="Bangalore", country=None, article_id="a1")

    db = env.db()
    india = market.heatmap_points(db, days=30, country="India", now=NOW)
    db.close()

    assert india["signals"] == 1
    assert india["country_inferred_from_city"] == 1


@pytest.mark.parametrize("value", [None, "", "Global", "global", " all "])
def test_global_country_values_mean_no_filter(value):
    assert market.country_filter(value) is None


# ---------------------------------------------------------------------------
# Dedupe and eligibility
# ---------------------------------------------------------------------------

def test_same_company_action_and_week_from_two_articles_counts_once(env):
    env.add_signal(company="Acme", city="Pune", count=100, confidence=0.8, when=W45_TUE, article_id="a1", title="First")
    env.add_signal(company="Acme", city="Pune", count=120, confidence=0.95, when=W45_WED, article_id="a2", title="Second")

    db = env.db()
    board = market.company_board(db, days=30, now=NOW)
    heat = market.heatmap_points(db, days=30, now=NOW)
    trend = market.weekly_trend(db, weeks=4, now=NOW)
    db.close()

    assert board["companies"][0]["growth"] == 1
    assert board["signals"] == 1
    assert board["articles"] == 2
    assert board["companies"][0]["reported_hires"] == {"total": 120, "signals_with_count": 1, "signals": 1}
    assert heat["points"][0]["signal_count"] == 1
    assert trend["totals"]["growth"] == 1
    assert sorted(a["title"] for a in board["companies"][0]["articles"]) == ["First", "Second"]


def test_same_company_in_different_weeks_counts_twice(env):
    env.add_signal(company="Acme", when=W45_TUE, article_id="a1")
    env.add_signal(company="Acme", when=W46_TUE, article_id="a2")

    db = env.db()
    board = market.company_board(db, days=30, now=NOW)
    db.close()

    assert board["companies"][0]["growth"] == 2


def test_past_context_and_other_actions_are_excluded(env):
    env.add_signal(company="Past", timing="past_context", article_id="a1")
    env.add_signal(company="Funded", action="FUNDING", article_id="a2")
    env.add_signal(company="Other", action="OTHER", article_id="a3")
    env.add_signal(company="Real", article_id="a4")

    db = env.db()
    board = market.company_board(db, days=30, now=NOW)
    heat = market.heatmap_points(db, days=30, now=NOW)
    trend = market.weekly_trend(db, weeks=4, now=NOW)
    db.close()

    assert [c["company"] for c in board["companies"]] == ["Real"]
    assert heat["signals"] == 1
    assert trend["totals"] == {"growth": 1, "cuts": 0}


def test_planned_signals_are_included(env):
    env.add_signal(company="Planner", timing="planned", action="NEW_OFFICE", article_id="a1")

    db = env.db()
    assert market.company_board(db, days=30, now=NOW)["companies"][0]["growth"] == 1
    db.close()


def test_dedupe_matches_the_step_3_matching_logic(env):
    """market.py repeats matching.py's grouping rules; both must pick the same winners."""
    env.add_signal(company="TCS", confidence=0.7, when=W45_TUE, article_id="a1")
    env.add_signal(company="Tata Consultancy Services", confidence=0.9, when=W45_WED, article_id="a2")
    env.add_signal(company="Acme", action="EXPANSION", when=W46_TUE, article_id="a3")
    env.add_signal(company="Acme", action="NEW_OFFICE", when=W46_TUE, article_id="a4")
    env.add_signal(company="Old", when=NOW - timedelta(days=60), article_id="a5")

    db = env.db()
    expected = matching.load_demand_signals(db, days=30, now=NOW)
    cutoff = NOW - timedelta(days=30)
    actual = [s for s in market._load_market_signals(db, cutoff, NOW) if s.action in matching.DEMAND_ACTIONS]
    db.close()

    def shape(signals):
        return sorted(
            (s.id, s.company, s.action, s.confidence, tuple(sorted(s.signal_ids)),
             tuple(sorted(a["id"] for a in s.articles)))
            for s in signals
        )

    assert shape(actual) == shape(expected)
    assert len(actual) == 3


# ---------------------------------------------------------------------------
# Company board
# ---------------------------------------------------------------------------

def test_reported_hires_sum_only_signals_with_a_count_and_say_how_many(env):
    env.add_signal(company="Acme", count=100, when=W44_TUE, article_id="a1")
    env.add_signal(company="Acme", count=None, when=W45_TUE, article_id="a2")
    env.add_signal(company="Acme", count=50, when=W46_TUE, article_id="a3")

    db = env.db()
    acme = market.company_board(db, days=30, now=NOW)["companies"][0]
    db.close()

    assert acme["growth"] == 3
    assert acme["reported_hires"] == {"total": 150, "signals_with_count": 2, "signals": 3}


def test_reported_hires_is_none_not_zero_when_no_signal_has_a_count(env):
    env.add_signal(company="Acme", count=None, article_id="a1")

    db = env.db()
    acme = market.company_board(db, days=30, now=NOW)["companies"][0]
    db.close()

    assert acme["reported_hires"] == {"total": None, "signals_with_count": 0, "signals": 1}
    assert acme["reported_layoffs"] == {"total": None, "signals_with_count": 0, "signals": 0}


def test_reported_layoffs_sum_only_layoff_signals_with_counts(env):
    env.add_signal(company="Acme", action="LAYOFF", count=300, when=W45_TUE, article_id="a1")
    env.add_signal(company="Acme", action="LAYOFF", count=None, when=W46_TUE, article_id="a2")
    env.add_signal(company="Acme", action="HIRING_FREEZE", count=40, when=W46_TUE, article_id="a3")

    db = env.db()
    acme = market.company_board(db, days=30, now=NOW)["companies"][0]
    db.close()

    assert acme["cuts"] == 3
    assert acme["reported_layoffs"] == {"total": 300, "signals_with_count": 1, "signals": 2}


def test_net_signal_growing_cutting_and_mixed(env):
    # Growing: 2 growth vs 1 cut
    env.add_signal(company="Grower", when=W44_TUE, article_id="g1")
    env.add_signal(company="Grower", when=W45_TUE, article_id="g2")
    env.add_signal(company="Grower", action="LAYOFF", when=W46_TUE, article_id="g3")
    # Cutting: 1 growth vs 2 cuts
    env.add_signal(company="Cutter", action="LAYOFF", when=W44_TUE, article_id="c1")
    env.add_signal(company="Cutter", action="HIRING_FREEZE", when=W45_TUE, article_id="c2")
    env.add_signal(company="Cutter", when=W46_TUE, article_id="c3")
    # Mixed: equal
    env.add_signal(company="Balanced", when=W45_TUE, article_id="b1")
    env.add_signal(company="Balanced", action="LAYOFF", when=W45_TUE, article_id="b2")

    db = env.db()
    board = {c["company"]: c for c in market.company_board(db, days=30, now=NOW)["companies"]}
    db.close()

    assert board["Grower"]["net_signal"] == "growing"
    assert board["Cutter"]["net_signal"] == "cutting"
    assert board["Balanced"]["net_signal"] == "mixed"
    assert (board["Grower"]["growth"], board["Grower"]["cuts"]) == (2, 1)


def test_board_limit_sorting_and_article_cap(env):
    for week_offset, when in enumerate([W44_TUE, W45_TUE, W46_TUE]):
        env.add_signal(company="Busy", when=when, article_id=f"busy{week_offset}")
    env.add_signal(company="Quiet", when=W45_TUE, article_id="quiet")
    for i in range(4):
        env.add_signal(
            company="ManySources", action="EXPANSION", when=W45_TUE, article_id=f"m{i}",
            title=f"Source {i}",
        )

    db = env.db()
    board = market.company_board(db, days=30, limit=2, now=NOW)
    db.close()

    assert [c["company"] for c in board["companies"]] == ["Busy", "ManySources"]
    assert len(board["companies"]) == 2
    many = [c for c in market.company_board(env.db(), days=30, now=NOW)["companies"] if c["company"] == "ManySources"][0]
    assert len(many["articles"]) == 3                                  # capped at 3 sources
    assert set(many["articles"][0]) == {"title", "link", "source"}      # nothing else leaks


def test_company_aliases_merge_on_the_board(env):
    env.add_signal(company="TCS", when=W44_TUE, article_id="a1")
    env.add_signal(company="Tata Consultancy Services", when=W46_TUE, article_id="a2")

    db = env.db()
    companies = market.company_board(db, days=30, now=NOW)["companies"]
    db.close()

    assert [c["company"] for c in companies] == ["Tata Consultancy Services"]
    assert companies[0]["growth"] == 2


def test_basis_strings_and_notes(env):
    env.add_signal(company="A", article_id="a1")
    env.add_signal(company="B", action="LAYOFF", article_id="a2")

    db = env.db()
    board = market.company_board(db, days=30, now=NOW)
    india = market.company_board(db, days=7, country="India", now=NOW)
    empty = market.company_board(db, days=1, now=NOW)
    db.close()

    assert board["basis"] == "Based on 2 signals from 2 articles in the last 30 days."
    assert "news coverage, not official hiring data" in board["note"]
    assert "not totals" in board["headcount_note"]
    assert empty["basis"] == "No hiring or layoff signals were found in the last day."
    assert india["country"] == "India"


# ---------------------------------------------------------------------------
# Weekly trend
# ---------------------------------------------------------------------------

def test_weekly_trend_zero_fills_empty_weeks(env):
    env.add_signal(company="A", when=W45_TUE, article_id="a1")
    env.add_signal(company="B", action="LAYOFF", when=W46_TUE, article_id="a2")

    db = env.db()
    trend = market.weekly_trend(db, weeks=4, now=NOW)
    db.close()

    assert [p["week"] for p in trend["series"]] == ["2026-W43", "2026-W44", "2026-W45", "2026-W46"]
    assert [p["start"] for p in trend["series"]] == ["2026-10-19", "2026-10-26", "2026-11-02", "2026-11-09"]
    assert [p["growth"] for p in trend["series"]] == [0, 0, 1, 0]
    assert [p["cuts"] for p in trend["series"]] == [0, 0, 0, 1]
    assert trend["totals"] == {"growth": 1, "cuts": 1}
    assert "published" in trend["week_basis"]


def test_weekly_trend_with_no_signals_is_all_zeros(env):
    db = env.db()
    trend = market.weekly_trend(db, weeks=3, now=NOW)
    db.close()

    assert len(trend["series"]) == 3
    assert all(p["growth"] == 0 and p["cuts"] == 0 for p in trend["series"])
    assert trend["signals"] == 0
    assert trend["basis"].startswith("No hiring or layoff signals")


def test_weekly_trend_ignores_signals_older_than_the_window(env):
    env.add_signal(company="Old", when=NOW - timedelta(days=70), article_id="a1")
    env.add_signal(company="New", when=W46_TUE, article_id="a2")

    db = env.db()
    trend = market.weekly_trend(db, weeks=4, now=NOW)
    db.close()

    assert trend["totals"]["growth"] == 1


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

def test_endpoints_are_public_and_return_data(env):
    env.add_signal(company="Acme", city="Pune", country="India", count=100, when=_last_week(1), article_id="a1")
    env.add_signal(company="Beta", action="LAYOFF", city="Seattle", country="USA", when=_last_week(2), article_id="a2")

    board = env.client.get("/api/career/company-board?days=30&country=India&limit=30")
    heat = env.client.get("/api/career/heatmap?days=30&country=India&group=all")
    trend = env.client.get("/api/career/trend?weeks=8&country=India")

    assert board.status_code == heat.status_code == trend.status_code == 200
    assert [c["company"] for c in board.json()["companies"]] == ["Acme"]
    assert [p["city"] for p in heat.json()["points"]] == ["Pune"]
    assert trend.json()["totals"] == {"growth": 1, "cuts": 0}
    assert len(trend.json()["series"]) == 8

    everyone = env.client.get("/api/career/company-board").json()
    assert sorted(c["company"] for c in everyone["companies"]) == ["Acme", "Beta"]


def test_no_response_contains_body_text_or_quotes(env):
    secret = "FULL-ARTICLE-BODY-SENTINEL-DO-NOT-LEAK"
    env.add_signal(company="Acme", city="Pune", country="India", when=_last_week(1), article_id="a1")
    db = env.db()
    db.add(ArticleBody(article_id="a1", body_text=secret, status="ok", attempts=1))
    db.commit()
    db.close()

    for url in ("/api/career/company-board", "/api/career/heatmap", "/api/career/trend"):
        text = env.client.get(url).text
        assert "body_text" not in text
        assert secret not in text
        assert "evidence_quote" not in text
        assert "must never be returned" not in text


@pytest.mark.parametrize("url", [
    "/api/career/company-board?days=0", "/api/career/company-board?days=91",
    "/api/career/company-board?days=abc", "/api/career/company-board?limit=0",
    "/api/career/company-board?limit=51",
    "/api/career/heatmap?days=0", "/api/career/heatmap?days=91",
    "/api/career/heatmap?group=everything", "/api/career/heatmap?group=",
    "/api/career/trend?weeks=0", "/api/career/trend?weeks=27", "/api/career/trend?weeks=abc",
])
def test_invalid_parameters_return_422(env, url):
    assert env.client.get(url).status_code == 422


@pytest.mark.parametrize("url", [
    "/api/career/company-board?days=1&limit=1", "/api/career/company-board?days=90&limit=50",
    "/api/career/heatmap?days=90&group=growth", "/api/career/heatmap?group=cuts",
    "/api/career/trend?weeks=1", "/api/career/trend?weeks=26",
    "/api/career/heatmap?country=Global", "/api/career/trend?country=",
])
def test_boundary_and_global_parameters_are_accepted(env, url):
    assert env.client.get(url).status_code == 200


def test_routes_and_page_are_wired_into_the_real_app():
    from backend.app.webapp import app

    client = TestClient(app)
    assert client.get("/career").status_code == 200
    # Public endpoints answer without a token (they read the real local DB, so only check shape).
    for url in ("/api/career/heatmap?days=7", "/api/career/trend?weeks=2", "/api/career/company-board?days=7"):
        assert client.get(url).status_code == 200


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _page():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", "career.html")
    return open(path, encoding="utf-8").read()


def test_page_has_the_new_tab_filters_map_chart_and_notes():
    html = _page()
    assert "Hiring vs Layoffs" in html
    assert 'id="tabBtnSkills"' in html and 'id="tabBtnMarket"' in html
    assert "leaflet@1.9.4" in html and "chart.js@4" in html
    assert "tile.openstreetmap.org" in html and "OpenStreetMap" in html
    assert "News coverage, not official data" in html
    assert 'value="India" selected' in html and "Global" in html
    assert 'data-group="growth"' in html and 'data-group="cuts"' in html
    assert "no stated location" in html
    assert "Hiring / expanding" in html and "Layoffs / freezes" in html


def test_page_degrades_gracefully_when_a_cdn_fails():
    html = _page()
    assert "its library did not load" in html           # map and chart fallback messages
    # the map (try/catch around the library load) and the chart (.catch) both have a failure path
    assert html.count("catch (err)") + html.count(".catch(") >= 2
    assert "integrity" in html


def test_page_never_inserts_api_data_with_innerhtml():
    html = _page()
    assert "innerHTML" not in html
    assert "insertAdjacentHTML" not in html
    assert "document.write" not in html


def test_skills_tab_content_is_unchanged():
    html = _page()
    for marker in ('id="demandContent"', 'id="gapContent"', 'id="daysSelector"',
                   "Skills in demand", "Your skill gap", "Inferred from news mentions"):
        assert marker in html


def test_page_does_not_use_window_properties_that_collide_with_element_ids():
    """Browsers expose every element id as a window property. window.marketMap was the <div>
    itself, which made the map skip its own creation and show the fallback message."""
    import re

    html = _page()
    element_ids = set(re.findall(r'\bid="([^"]+)"', html))
    window_names = set(re.findall(r"\bwindow\.([A-Za-z_$][\w$]*)", html))
    assert element_ids & window_names == set()
    assert "window.marketMap" not in html


def test_deep_link_waits_for_the_page_script_before_starting_the_market_tab():
    """/career#market must not run before the market state (declared with let/const) exists."""
    import re

    html = _page()
    assert re.search(r"DOMContentLoaded'[^)]*\)\s*=>\s*\{\s*if \(window\.location\.hash === '#market'\)", html)


def test_map_failures_distinguish_a_missing_library_from_a_drawing_error():
    html = _page()
    assert "its library did not load" in html
    assert "loaded but could not be drawn" in html
