"""Prompt v5 rules, generic skill denylist and layoff skills."""
import json

import pytest

from backend.app.models import CareerSignal
from backend.services.career import body_fetcher
from backend.services.career import extractor as extractor_module
from backend.services.career.body_fetcher import FetchResult
from backend.services.career.extractor import CareerExtractor
from backend.services.career.grounding import drop_generic_skills, is_generic_skill
from backend.services.career.prompt import CAREER_SYSTEM_PROMPT, PROMPT_VERSION

PROMPT = CAREER_SYSTEM_PROMPT.lower()


def test_prompt_version_is_v5():
    assert PROMPT_VERSION == "v5"


def test_prompt_voluntary_exit_rule():
    assert "voluntary exit" in PROMPT and "buyout" in PROMPT and "separation" in PROMPT
    assert "only if the article says employees were actually let go" in PROMPT
    assert "set count to null" in PROMPT
    assert "eligible or offered" in PROMPT


def test_prompt_location_rule_skips_possible_places():
    assert "only give a location where the article states the event happens or is confirmed" in PROMPT
    assert "could also be affected" in PROMPT
    assert "may also see cuts" in PROMPT


def test_prompt_skills_are_technologies_only():
    assert "technologies, tools, programming languages, frameworks" in PROMPT
    for forbidden in ("finance", "procurement", "hr", "human resources", "marketing", "legal", "digital"):
        assert forbidden in PROMPT  # named as examples of what NOT to include
    assert "never include job functions, departments or role titles" in PROMPT
    assert "[] for every layoff and hiring_freeze" in PROMPT


@pytest.mark.parametrize(
    "skill",
    ["digital", "Finance", "human resources", "HR", "Procurement", "marketing", "Legal", "sales",
     "operations", "AI models", "artificial intelligence (AI)", "AI", "Finance skills", "data"],
)
def test_generic_skills_are_dropped(skill):
    assert is_generic_skill(skill)


@pytest.mark.parametrize(
    "skill",
    ["Python", "Kubernetes", "SAP", "Salesforce", "machine learning", "Generative AI", "LLM",
     "C++", "AWS", "Data engineering", "React", "Rust"],
)
def test_real_skills_are_kept(skill):
    assert not is_generic_skill(skill)


def test_drop_generic_skills_counts():
    kept, dropped = drop_generic_skills(["Python", "finance", "AI models", "Kubernetes", "digital"])
    assert kept == ["Python", "Kubernetes"] and dropped == 3


BODY = (
    "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India. "
    "The roles will focus on Python and Kubernetes plus finance and digital work. "
    "Separately Acme Robotics said it will cut 200 jobs in Pune, India, in finance and procurement."
)


def _run(career_db, make_article, monkeypatch, signals):
    article = make_article(article_id="a1", title="Acme hires and cuts", summary="Acme will hire.")
    monkeypatch.setattr(
        body_fetcher, "fetch_article_body",
        lambda url, **k: FetchResult("ok", body_text=BODY, http_status=200),
    )
    monkeypatch.setattr(extractor_module, "call_groq", lambda *a, **k: (json.dumps(signals), "m"))
    outcome = CareerExtractor(api_key="k", fetch_delay_seconds=0, llm_call_delay_seconds=0).process_article(
        article.id, db=career_db, allow_fetch=True
    )
    return outcome, {s.action: s for s in career_db.query(CareerSignal).all()}


def _signal(action, quote, skills, **extra):
    base = {
        "company": "Acme Robotics Pvt Ltd", "action": action, "timing": "current", "count": None,
        "location": {"city": "Pune", "country": "India"}, "skills_mentioned": skills,
        "confidence": 0.9, "evidence_quote": quote,
    }
    base.update(extra)
    return base


def test_generic_skills_dropped_and_counted_layoff_skills_cleared(career_db, make_article, monkeypatch):
    signals = [
        _signal("HIRING", "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India.",
                ["Python", "Kubernetes", "finance", "digital"], count="500",
                location={"city": "Bengaluru", "country": "India"}),
        _signal("LAYOFF", "Separately Acme Robotics said it will cut 200 jobs in Pune, India, in finance and procurement.",
                ["procurement", "Python"], count="200"),
    ]
    outcome, rows = _run(career_db, make_article, monkeypatch, signals)
    assert outcome.status == "succeeded"
    assert rows["HIRING"].skills_mentioned == ["Python", "Kubernetes"]
    assert rows["LAYOFF"].skills_mentioned == []                  # layoffs store no skills
    assert outcome.dropped.as_dict()["generic_skill"] >= 3      # finance, digital, procurement


def test_hiring_freeze_skills_are_cleared(career_db, make_article, monkeypatch):
    signals = [_signal("HIRING_FREEZE", "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India.",
                       ["Python"])]
    _, rows = _run(career_db, make_article, monkeypatch, signals)
    assert rows["HIRING_FREEZE"].skills_mentioned == []


def test_backfill_summary_prints_generic_skill_drops(capsys):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backfill_career_signals.py"
    spec = importlib.util.spec_from_file_location("backfill_script3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = extractor_module.BatchStats()
    stats.dropped.add("generic_skill", 5)
    module._print_summary(stats)
    assert "generic_skill: 5" in capsys.readouterr().out
