"""Estimate guard: a count in a hedged sentence is not a stated count."""
import json

import pytest

from backend.app.models import CareerSignal
from backend.services.career import body_fetcher
from backend.services.career import extractor as extractor_module
from backend.services.career.body_fetcher import FetchResult
from backend.services.career.extractor import CareerExtractor
from backend.services.career.grounding import count_is_estimated, estimate_cue_in_count_sentence

ESTIMATED = [
    (1000, "While the exact number of job cuts has not been confirmed, initial estimates suggest that around 1,000 employees may be affected."),
    (1000, "Netflix is reportedly cutting nearly 1,000 jobs, or about 6 per cent of its workforce."),
    (500, "The company is estimated to lay off 500 people."),
    (500, "Estimates put the cuts at 500 jobs."),
    (850, "Qantas will cut approximately 850 jobs."),
    (850, "Qantas will cut around 850 jobs."),
    (300, "As many as 300 roles could go, the memo said."),
    (300, "Up to 300 employees may lose their jobs."),
    (1200, "The firm will cut about 1,200 jobs."),
    (400, "The number of layoffs, 400, has not been confirmed."),
    (400, "A figure of 400 employees is unconfirmed."),
    (60000, "Microsoft is offering exit packages to 60,000 employees, though uptake is not confirmed."),
]

STATED = [
    (1040, "PVH plans to eliminate 1,040 jobs, or 4 per cent of its global workforce."),
    (1040, "The company will lay off 1,040 employees."),
    (1271, "Canva has announced that it is reducing its workforce by about 22 per cent, affecting 1,271 employees."),
    (850, "Qantas plans to eliminate 850 jobs. Analysts say the savings could be large."),   # hedge is in another sentence
    (80, "Innoviz Technologies plans to lay off 80 employees, representing around 20 per cent of its workforce"),
    (1100, "The move will eliminate 1,100 jobs, based on the 12,300 employees eBay had at the end of last year."),
]


@pytest.mark.parametrize("count,quote", ESTIMATED)
def test_hedged_sentences_make_the_count_an_estimate(count, quote):
    assert count_is_estimated(count, quote), quote


@pytest.mark.parametrize("count,quote", STATED)
def test_stated_counts_are_kept(count, quote):
    assert not count_is_estimated(count, quote), quote


def test_none_and_empty_inputs():
    assert not count_is_estimated(None, "around 1,000 jobs")
    assert not count_is_estimated(1000, "")
    assert not count_is_estimated(1000, None)


def test_count_not_in_quote_is_left_alone():
    assert not count_is_estimated(2000, "Reportedly around 1,000 jobs will go.")


def test_percentage_is_not_the_count():
    assert not count_is_estimated(20, "The company will cut 80 jobs, about 20 per cent of staff.")


def test_cue_is_reported():
    assert estimate_cue_in_count_sentence(1000, "Nearly 1,000 jobs will go.") == "nearly"


BODY = ("Acme Robotics Pvt Ltd said it will lay off 1,040 employees in Pune, India. "
        "Separately, Beta Corp is reportedly cutting nearly 1,000 jobs in Chennai, India.")


def _run(career_db, make_article, monkeypatch, signals):
    article = make_article(article_id="a1", title="Acme and Beta layoffs", summary="Acme will lay off staff.")
    monkeypatch.setattr(body_fetcher, "fetch_article_body",
                        lambda url, **k: FetchResult("ok", body_text=BODY, http_status=200))
    monkeypatch.setattr(extractor_module, "call_groq", lambda *a, **k: (json.dumps(signals), "m"))
    outcome = CareerExtractor(api_key="k", fetch_delay_seconds=0, llm_call_delay_seconds=0).process_article(
        article.id, db=career_db, allow_fetch=True)
    return outcome, {s.company: s for s in career_db.query(CareerSignal).all()}


def test_extractor_nulls_estimated_counts_and_counts_the_drop(career_db, make_article, monkeypatch):
    def sig(company, count, quote, city):
        return {"company": company, "action": "LAYOFF", "timing": "current", "count": count,
                "location": {"city": city, "country": "India"}, "skills_mentioned": [],
                "confidence": 0.9, "evidence_quote": quote}
    signals = [
        sig("Acme Robotics Pvt Ltd", "1040", "Acme Robotics Pvt Ltd said it will lay off 1,040 employees in Pune, India.", "Pune"),
        sig("Beta Corp", "1000", "Separately, Beta Corp is reportedly cutting nearly 1,000 jobs in Chennai, India.", "Chennai"),
    ]
    outcome, rows = _run(career_db, make_article, monkeypatch, signals)
    assert outcome.status == "succeeded"
    acme = next(r for c, r in rows.items() if c.startswith("Acme"))
    assert acme.count == 1040
    assert rows["Beta"].count is None   # "Corp" is a stripped legal suffix
    assert outcome.dropped.as_dict()["estimated_count"] == 1


def test_backfill_summary_prints_estimated_count(capsys):
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backfill_career_signals.py"
    spec = importlib.util.spec_from_file_location("backfill_script4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = extractor_module.BatchStats()
    stats.dropped.add("estimated_count", 2)
    module._print_summary(stats)
    assert "estimated_count: 2" in capsys.readouterr().out

