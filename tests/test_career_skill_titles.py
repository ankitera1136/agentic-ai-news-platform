"""Job titles are not skills: dropped, or stripped from technologies."""
import json

import pytest

from backend.services.career import body_fetcher
from backend.services.career import extractor as extractor_module
from backend.services.career.body_fetcher import FetchResult
from backend.services.career.extractor import CareerExtractor
from backend.services.career.grounding import DropCounters, clean_job_title_skill, clean_job_title_skills


def test_whole_title_strings_are_dropped():
    cleaned, dropped = clean_job_title_skills(["engineer", "senior engineer", "software engineer"])
    assert cleaned == [] and dropped == 3


@pytest.mark.parametrize(
    "title",
    ["developer", "Developers", "manager", "Analyst", "intern", "associate", "lead", "director",
     "executive", "staff", "employee", "worker", "engineers", "Employees", "Workers",
     "Senior Software Engineer", "Product manager", "Data analyst"],
)
def test_generic_titles_are_dropped(title):
    assert clean_job_title_skill(title) is None


def test_technologies_are_kept():
    cleaned, dropped = clean_job_title_skills(["Python", "Kubernetes", "C++", "React", "Lead generation"])
    assert cleaned == ["Python", "Kubernetes", "C++", "React", "Lead generation"]
    assert dropped == 0


@pytest.mark.parametrize(
    "raw,expected",
    [("Python developer", "Python"), ("Java Developers", "Java"),
     ("Machine Learning engineer", "Machine Learning"), ("Senior Python developer", "Python")],
)
def test_title_suffix_is_removed_from_technologies(raw, expected):
    assert clean_job_title_skill(raw) == expected


def test_empty_input():
    assert clean_job_title_skills([]) == ([], 0)
    assert clean_job_title_skills(None) == ([], 0)


def test_drop_counter_exists():
    counters = DropCounters()
    counters.add("job_title_skill", 2)
    assert counters.as_dict()["job_title_skill"] == 2


BODY = (
    "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India. "
    "The roles will focus on Python and Kubernetes. A senior engineer will lead each team. "
    "The company added that the new team will work on warehouse automation and robotics."
)


def test_extractor_drops_titles_keeps_technologies_and_counts(career_db, make_article, monkeypatch):
    article = make_article(article_id="a1", title="Acme Robotics to hire 500 engineers", summary="Acme will hire.")
    monkeypatch.setattr(
        body_fetcher, "fetch_article_body",
        lambda url, **k: FetchResult("ok", body_text=BODY, http_status=200),
    )
    signal = {
        "company": "Acme Robotics Pvt Ltd", "action": "HIRING", "timing": "current", "count": "500",
        "location": {"city": "Bengaluru", "country": "India"},
        "skills_mentioned": ["engineer", "senior engineer", "Python developer", "Python", "Kubernetes"],
        "confidence": 0.9,
        "evidence_quote": "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India.",
    }
    monkeypatch.setattr(extractor_module, "call_groq", lambda *a, **k: (json.dumps([signal]), "test-model"))
    outcome = CareerExtractor(api_key="k", fetch_delay_seconds=0, llm_call_delay_seconds=0).process_article(
        article.id, db=career_db, allow_fetch=True
    )
    assert outcome.status == "succeeded"
    from backend.app.models import CareerSignal
    row = career_db.query(CareerSignal).one()
    assert row.skills_mentioned == ["Python", "Kubernetes"]
    assert outcome.dropped.as_dict()["job_title_skill"] == 2


def test_backfill_summary_prints_job_title_drops(capsys):
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backfill_career_signals.py"
    spec = importlib.util.spec_from_file_location("backfill_script2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = extractor_module.BatchStats()
    stats.dropped.add("job_title_skill", 3)
    module._print_summary(stats)
    assert "job_title_skill: 3" in capsys.readouterr().out
