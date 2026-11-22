"""CareerRadar pre-filter wiring (CAREER_PREFILTER_ENABLED)."""
import json

import pytest

from backend.app.models import ArticleBody, CareerExtraction, CareerSignal
from backend.services.career import body_fetcher
from backend.services.career import extractor as extractor_module
from backend.services.career import filters as filters_module
from backend.services.career.body_fetcher import FetchResult
from backend.services.career.extractor import CareerExtractor

BODY_TEXT = (
    "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India. "
    "The roles will focus on Python and Kubernetes. The company added that "
    "the new team will work on warehouse automation and robotics."
)
SIGNAL = {
    "company": "Acme Robotics Pvt Ltd",
    "action": "HIRING",
    "timing": "current",
    "count": "500",
    "location": {"city": "Bengaluru", "country": "India"},
    "skills_mentioned": ["Python", "Kubernetes"],
    "confidence": 0.9,
    "evidence_quote": "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India.",
}

NOT_CAREER_TITLE = "The OpenAI board member who fired Sam Altman says we still can't trust AI CEOs"
CAREER_TITLE = "Acme Robotics to hire 500 engineers in Bengaluru"


@pytest.fixture
def probes(monkeypatch):
    """Count body fetches and LLM calls; the LLM answers with one valid signal."""
    state = {"fetch": 0, "llm": 0}

    def fake_fetch(url, **kwargs):
        state["fetch"] += 1
        return FetchResult("ok", body_text=BODY_TEXT, http_status=200)

    def fake_call(api_key, payload, timeout=25, preferred_model=None):
        state["llm"] += 1
        return json.dumps([SIGNAL]), "test-model"

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)
    monkeypatch.setattr(extractor_module, "call_groq", fake_call)
    return state


def _extractor():
    return CareerExtractor(api_key="test-key", fetch_delay_seconds=0, llm_call_delay_seconds=0)


def _ledger(db, article_id):
    return db.query(CareerExtraction).filter_by(article_id=article_id).one()


def test_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv("CAREER_PREFILTER_ENABLED", raising=False)
    assert extractor_module.prefilter_enabled() is False


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("ON", True), ("false", False), ("", False), ("0", False)])
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", value)
    assert extractor_module.prefilter_enabled() is expected


def test_flag_off_is_unchanged_non_career_article_is_extracted(career_db, make_article, probes, monkeypatch):
    monkeypatch.delenv("CAREER_PREFILTER_ENABLED", raising=False)
    article = make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="Opinion piece.")
    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)
    assert outcome.status == "succeeded"
    assert probes == {"fetch": 1, "llm": 1}
    row = _ledger(career_db, "a1")
    assert row.text_source == "full_body"
    # The hash is exactly what the pre-prefilter code computed.
    assert row.content_hash == extractor_module.compute_content_hash(
        "full_body", extractor_module.text_for_llm(f"{NOT_CAREER_TITLE}\n\n{BODY_TEXT}")
    )


def test_flag_on_skips_non_career_article_without_http_or_llm(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    article = make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="Opinion piece.")
    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)
    assert outcome.status == "skipped_not_career"
    assert probes == {"fetch": 0, "llm": 0}
    row = _ledger(career_db, "a1")
    assert (row.status, row.signals_count, row.text_source) == ("succeeded", 0, "skipped_not_career")
    assert career_db.query(ArticleBody).count() == 0
    assert career_db.query(CareerSignal).count() == 0


def test_flag_on_lets_career_article_through(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    article = make_article(article_id="a1", title=CAREER_TITLE, summary="Acme Robotics is growing.")
    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)
    assert outcome.status == "succeeded"
    assert outcome.signals_count == 1
    assert probes == {"fetch": 1, "llm": 1}
    assert _ledger(career_db, "a1").text_source == "full_body"


def test_career_signal_in_summary_only_passes(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    article = make_article(article_id="a1", title="Acme Robotics update", summary="Acme Robotics will hire 500 engineers.")
    assert _extractor().process_article(article.id, db=career_db, allow_fetch=True).status == "succeeded"


def test_roundup_check_still_runs_first(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    article = make_article(article_id="a1", title="Weekly quiz: test your tech knowledge", summary="")
    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)
    assert outcome.status == "skipped_roundup"
    assert _ledger(career_db, "a1").text_source == "skipped_roundup"


def test_skipped_rows_do_not_reopen_on_normal_rerun(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="Opinion piece.")
    extractor = _extractor()
    extractor.process_pending_articles(db=career_db, limit=5, allow_fetch=True)
    first = _ledger(career_db, "a1")
    first_state = (first.content_hash, first.attempts, first.extracted_at)
    stats = extractor.process_pending_articles(db=career_db, limit=5, allow_fetch=True)
    again = _ledger(career_db, "a1")
    assert (again.content_hash, again.attempts, again.extracted_at) == first_state
    assert stats.skipped_not_career == 1
    assert probes == {"fetch": 0, "llm": 0}


def test_version_bump_reopens_skipped_rows(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="Opinion piece.")
    extractor = _extractor()
    extractor.process_article("a1", db=career_db, allow_fetch=True)
    old_hash = _ledger(career_db, "a1").content_hash

    # New version, same regex outcome: row is re-evaluated, re-written with a new hash, still no LLM.
    monkeypatch.setattr(extractor_module, "FILTER_VERSION", "test-filter-bump-1")
    outcome = extractor.process_article("a1", db=career_db, allow_fetch=True)
    assert outcome.status == "skipped_not_career"
    row = _ledger(career_db, "a1")
    assert row.content_hash != old_hash
    assert row.attempts == 2
    assert probes == {"fetch": 0, "llm": 0}

    # New version whose rules now accept the article: it goes on to a real extraction.
    monkeypatch.setattr(extractor_module, "FILTER_VERSION", "test-filter-bump-2")
    monkeypatch.setattr(extractor_module, "career_relevance", lambda title, summary=None: "layoff")
    outcome = extractor.process_article("a1", db=career_db, allow_fetch=True)
    assert outcome.status == "succeeded"
    assert probes == {"fetch": 1, "llm": 1}
    assert _ledger(career_db, "a1").text_source == "full_body"


def test_toggling_flag_off_keeps_existing_full_body_hashes(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    make_article(article_id="a1", title=CAREER_TITLE, summary="Acme is growing.")
    extractor = _extractor()
    extractor.process_article("a1", db=career_db, allow_fetch=True)
    hash_on = _ledger(career_db, "a1").content_hash

    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "false")
    outcome = extractor.process_article("a1", db=career_db, allow_fetch=True)
    assert outcome.status == "skipped" and outcome.error == "unchanged"
    assert _ledger(career_db, "a1").content_hash == hash_on
    assert probes == {"fetch": 1, "llm": 1}


def test_existing_extraction_is_not_overwritten_by_the_prefilter(career_db, make_article, probes, monkeypatch):
    monkeypatch.delenv("CAREER_PREFILTER_ENABLED", raising=False)
    make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="Opinion piece.")
    extractor = _extractor()
    extractor.process_article("a1", db=career_db, allow_fetch=True)
    before = _ledger(career_db, "a1").content_hash
    signals_before = career_db.query(CareerSignal).count()

    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    outcome = extractor.process_article("a1", db=career_db, allow_fetch=True)
    assert outcome.status == "skipped" and outcome.error == "already_processed"
    row = _ledger(career_db, "a1")
    assert row.content_hash == before and row.text_source == "full_body"
    assert career_db.query(CareerSignal).count() == signals_before


def test_skipped_rows_are_reopened_when_flag_is_turned_off(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="Opinion piece.")
    extractor = _extractor()
    extractor.process_article("a1", db=career_db, allow_fetch=True)
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "false")
    outcome = extractor.process_article("a1", db=career_db, allow_fetch=True)
    assert outcome.status == "succeeded"
    assert _ledger(career_db, "a1").text_source == "full_body"


def test_batch_stats_and_dict_count_the_new_skip(career_db, make_article, probes, monkeypatch):
    monkeypatch.setenv("CAREER_PREFILTER_ENABLED", "true")
    make_article(article_id="a1", title=NOT_CAREER_TITLE, summary="x")
    make_article(article_id="a2", title="A family tree of software job titles", summary="x")
    make_article(article_id="a3", title=CAREER_TITLE, summary="Acme is growing.")
    stats = _extractor().process_pending_articles(db=career_db, limit=5, allow_fetch=True)
    assert stats.skipped_not_career == 2
    assert stats.processed == 1  # skipped articles do not use up the LLM batch limit
    assert stats.as_dict()["skipped_not_career"] == 2


def test_backfill_summary_prints_the_new_counter(capsys):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backfill_career_signals.py"
    spec = importlib.util.spec_from_file_location("backfill_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = extractor_module.BatchStats()
    stats.skipped_not_career = 3
    module._print_summary(stats)
    assert "skipped (not career):      3" in capsys.readouterr().out


def test_filter_version_comes_from_filters_module():
    assert extractor_module.FILTER_VERSION == filters_module.FILTER_VERSION
