import pytest
import json

from backend.app.models import ArticleBody, CareerExtraction, CareerSignal
from backend.services.career import body_fetcher
from backend.services.career import extractor as extractor_module
from backend.services.career.body_fetcher import FetchResult
from backend.services.career.extractor import CareerExtractor, default_batch_limit

BODY_TEXT = (
    "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India. "
    "The roles will focus on Python and Kubernetes. The company added that "
    "the new team will work on warehouse automation and robotics."
)


def _signal(**overrides):
    item = {
        "company": "Acme Robotics Pvt Ltd",
        "action": "HIRING",
        "timing": "current",
        "count": "500",
        "location": {"city": "Bengaluru", "country": "India"},
        "skills_mentioned": ["Python", "Kubernetes", "Rust"],
        "confidence": 0.9,
        "evidence_quote": (
            "Acme Robotics Pvt Ltd said it will hire 500 engineers in Bengaluru, India."
        ),
    }
    item.update(overrides)
    return item


def _mock_body(monkeypatch, result=None, counter=None):
    result = result or FetchResult("ok", body_text=BODY_TEXT, http_status=200)

    def fake_fetch(url, **kwargs):
        if counter is not None:
            counter["count"] += 1
        return result

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)


def _mock_llm(monkeypatch, responses, calls=None):
    queue = list(responses)

    def fake_call(api_key, payload, timeout=25, preferred_model=None):
        if calls is not None:
            calls.append(payload)
        if not queue:
            raise AssertionError("Unexpected extra LLM call")
        value = queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, "test-model"

    monkeypatch.setattr(extractor_module, "call_groq", fake_call)


def _extractor():
    return CareerExtractor(api_key="test-key", fetch_delay_seconds=0, llm_call_delay_seconds=0)


def _extraction(db, article_id="art1"):
    return db.query(CareerExtraction).filter_by(article_id=article_id).one()


def test_valid_extraction_is_grounded_and_persisted(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [json.dumps([_signal()])])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.signals_count == 1
    assert outcome.text_source == "full_body"

    signal = career_db.query(CareerSignal).one()
    assert signal.source_article_id == article.id
    assert signal.company == "Acme Robotics"
    assert signal.company_raw == "Acme Robotics Pvt Ltd"
    assert signal.action == "HIRING"
    assert signal.timing == "current"
    assert signal.count == 500
    assert signal.location_city == "Bengaluru"
    assert signal.location_country == "India"
    assert signal.skills_mentioned == ["Python", "Kubernetes"]
    assert signal.confidence == 0.9
    assert "500 engineers" in signal.evidence_quote

    extraction = _extraction(career_db)
    assert extraction.status == "succeeded"
    assert extraction.signals_count == 1
    assert extraction.text_source == "full_body"
    assert extraction.attempts == 1
    assert extraction.model_used == "test-model"

    body = career_db.query(ArticleBody).filter_by(article_id=article.id).one()
    assert body.status == "ok"
    assert body.body_text == BODY_TEXT


def test_zero_signals_marks_extraction_succeeded(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, ["[]"])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.signals_count == 0
    assert career_db.query(CareerSignal).count() == 0
    assert _extraction(career_db).signals_count == 0


def test_second_run_is_idempotent_and_makes_no_llm_call(career_db, make_article, monkeypatch):
    article = make_article()
    body_calls = {"count": 0}
    llm_calls = []
    _mock_body(monkeypatch, counter=body_calls)
    _mock_llm(monkeypatch, [json.dumps([_signal()])], calls=llm_calls)

    extractor = _extractor()
    first = extractor.process_article(article.id, db=career_db, allow_fetch=True)
    second = extractor.process_article(article.id, db=career_db, allow_fetch=True)

    assert first.status == "succeeded"
    assert second.status == "skipped"
    assert second.error == "unchanged"
    assert len(llm_calls) == 1
    assert body_calls["count"] == 1
    assert career_db.query(CareerSignal).count() == 1


def test_invalid_json_retries_once_then_succeeds(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    calls = []
    _mock_llm(monkeypatch, ["not json at all", "[]"], calls=calls)

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert len(calls) == 2
    assert _extraction(career_db).attempts == 1


def test_invalid_json_twice_marks_failed(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    calls = []
    _mock_llm(monkeypatch, ["not json", "still not json"], calls=calls)

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "failed"
    assert len(calls) == 2

    extraction = _extraction(career_db)
    assert extraction.status == "failed"
    assert extraction.attempts == 1
    assert "invalid_json_after_retry" in extraction.error


def test_failed_extraction_stops_after_retry_limit(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    calls = []
    _mock_llm(monkeypatch, [], calls=calls)

    extraction = CareerExtraction(
        article_id=article.id,
        content_hash="same-hash",
        text_source="full_body",
        status="failed",
        attempts=3,
    )
    career_db.add(extraction)
    career_db.commit()

    # Force the hash to match by reusing the extractor's own computation.
    from backend.services.career.extractor import compute_content_hash, text_for_llm
    from backend.services.career.body_fetcher import ensure_article_body

    context = ensure_article_body(career_db, article, allow_fetch=True, delay_seconds=0)
    extraction.content_hash = compute_content_hash(
        context.text_source, text_for_llm(context.text)
    )
    career_db.commit()

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "skipped"
    assert outcome.error == "retry_limit_reached"
    assert calls == []


def test_invalid_action_is_dropped(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [json.dumps([_signal(action="FIRING")])])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.signals_count == 0
    assert outcome.dropped.invalid_action == 1


def test_invalid_timing_is_dropped(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [json.dumps([_signal(timing="someday")])])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.signals_count == 0
    assert outcome.dropped.invalid_timing == 1


def test_verge_retrospective_layoff_is_dropped(career_db, make_article, monkeypatch):
    """The Jensen Huang founder-CEO essay mentions an old layoff as biography."""
    article = make_article(
        article_id="verge-history",
        title="Jensen Huang and the decline of the visionary showman founder CEO",
        summary="A look at founder CEOs and the mythology around them.",
        llm_summary="Essay on founder-led companies.",
        link="https://example.com/verge-history",
    )
    body = (
        "Jensen Huang and the decline of the visionary showman founder CEO. "
        "In the early years, he had to lay off over half of Nvidia's roughly 100 staffers. "
        "The company later became one of the most valuable in the world."
    )
    _mock_body(monkeypatch, FetchResult("ok", body_text=body, http_status=200))
    _mock_llm(
        monkeypatch,
        [json.dumps([_signal(
            company="Nvidia",
            action="LAYOFF",
            timing="past_context",
            count=None,
            location=None,
            skills_mentioned=[],
            confidence=0.95,
            evidence_quote="he had to lay off over half of Nvidia's roughly 100 staffers",
        )])],
    )

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.signals_count == 0
    assert outcome.dropped.past_context == 1
    assert career_db.query(CareerSignal).count() == 0


def test_hindu_retrospective_resignation_is_dropped(career_db, make_article, monkeypatch):
    """The Hindu piece recounts a past resignation as background, not current news."""
    article = make_article(
        article_id="hindu-history",
        title="Jensen Huang | AI rainmaker meets his Waterloo",
        summary="A profile of Jensen Huang.",
        llm_summary="Retrospective profile.",
        link="https://example.com/hindu-history",
    )
    body = (
        "By then, the controversy had come to a head, forcing Mr. Huang to resign as CEO of "
        "Nvidia, the company he had built from scratch. Nvidia's board chose a successor."
    )
    _mock_body(monkeypatch, FetchResult("ok", body_text=body, http_status=200))
    _mock_llm(
        monkeypatch,
        [json.dumps([_signal(
            company="Nvidia",
            action="OTHER",
            timing="past_context",
            count=None,
            location=None,
            skills_mentioned=[],
            confidence=0.90,
            evidence_quote=(
                "By then, the controversy had come to a head, forcing Mr. Huang to resign "
                "as CEO of Nvidia, the company he had built from scratch."
            ),
        )])],
    )

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.signals_count == 0
    assert outcome.dropped.past_context == 1
    assert career_db.query(CareerSignal).count() == 0


def test_current_layoff_announcement_is_extracted(career_db, make_article, monkeypatch):
    """A fresh layoff announcement must survive the historical-context filter."""
    article = make_article(
        article_id="current-layoff",
        title="Zeta Systems to lay off 300 employees in Austin",
        summary="The restructuring affects engineering and support roles.",
        llm_summary="Zeta Systems announced layoffs.",
        link="https://example.com/current-layoff",
    )
    body = (
        "Zeta Systems said on Tuesday it will lay off 300 employees in Austin, United States, "
        "as part of a restructuring. The company said the cuts affect engineering and support roles."
    )
    _mock_body(monkeypatch, FetchResult("ok", body_text=body, http_status=200))
    _mock_llm(
        monkeypatch,
        [json.dumps([_signal(
            company="Zeta Systems",
            action="LAYOFF",
            timing="current",
            count="300",
            location={"city": "Austin", "country": "United States"},
            skills_mentioned=[],
            confidence=0.93,
            evidence_quote=(
                "Zeta Systems said on Tuesday it will lay off 300 employees in Austin, "
                "United States, as part of a restructuring."
            ),
        )])],
    )

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.signals_count == 1
    assert outcome.dropped.past_context == 0
    signal = career_db.query(CareerSignal).one()
    assert signal.company == "Zeta Systems"
    assert signal.action == "LAYOFF"
    assert signal.timing == "current"
    assert signal.count == 300
    assert signal.location_city == "Austin"
    assert signal.location_country == "United States"


def test_low_confidence_is_dropped(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [json.dumps([_signal(confidence=0.2)])])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.signals_count == 0
    assert outcome.dropped.low_confidence == 1


def test_hallucinated_evidence_is_dropped(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(
        monkeypatch,
        [json.dumps([_signal(evidence_quote="This sentence is not in the article at all.")])],
    )

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.signals_count == 0
    assert outcome.dropped.evidence_quote == 1


def test_ungrounded_skill_is_removed(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [json.dumps([_signal()])])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    signal = career_db.query(CareerSignal).one()
    assert signal.skills_mentioned == ["Python", "Kubernetes"]
    assert outcome.dropped.skill_not_grounded == 1


def test_tcs_alias_is_normalized(career_db, make_article, monkeypatch):
    article = make_article()
    summary = "TCS will hire 200 engineers in Pune, India. The roles require Java."
    article.summary = summary
    article.llm_summary = "TCS is hiring."
    career_db.commit()

    _mock_body(
        monkeypatch,
        FetchResult(
            "ok",
            body_text="TCS will hire 200 engineers in Pune, India. The roles require Java.",
            http_status=200,
        ),
    )
    _mock_llm(
        monkeypatch,
        [json.dumps([_signal(
            company="TCS",
            count="200",
            location={"city": "Pune", "country": "India"},
            skills_mentioned=["Java"],
            evidence_quote="TCS will hire 200 engineers in Pune, India.",
        )])],
    )

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    signal = career_db.query(CareerSignal).one()
    assert signal.company == "Tata Consultancy Services"
    assert signal.company_raw == "TCS"
    assert outcome.unknown_companies == []


def test_copy_article_is_skipped(career_db, make_article, monkeypatch):
    original = make_article(article_id="original")
    copy = make_article(article_id="copy", is_copy_of=original.id)
    _mock_body(monkeypatch)
    llm_calls = []
    _mock_llm(monkeypatch, [], calls=llm_calls)

    outcome = _extractor().process_article(copy.id, db=career_db, allow_fetch=True)

    assert outcome.status == "skipped"
    assert outcome.error == "copy_article"
    assert llm_calls == []
    assert career_db.query(CareerExtraction).count() == 0


def test_changed_content_replaces_old_signals(career_db, make_article, monkeypatch):
    article = make_article()
    body_calls = {"count": 0}
    llm_calls = []
    _mock_body(monkeypatch, counter=body_calls)
    _mock_llm(
        monkeypatch,
        [
            json.dumps([_signal()]),
            json.dumps([_signal(
                company="TCS",
                action="LAYOFF",
                count="200",
                location={"city": "Pune", "country": "India"},
                skills_mentioned=[],
                confidence=0.95,
                evidence_quote="TCS will lay off 200 employees in Pune, India.",
            )]),
        ],
        calls=llm_calls,
    )

    extractor = _extractor()
    first = extractor.process_article(article.id, db=career_db, allow_fetch=True)
    assert first.signals_count == 1
    assert career_db.query(CareerSignal).one().company == "Acme Robotics"

    body = career_db.query(ArticleBody).filter_by(article_id=article.id).one()
    body.body_text = (
        "TCS will lay off 200 employees in Pune, India. "
        "The company said the roles were affected by automation."
    )
    career_db.commit()

    second = extractor.process_article(article.id, db=career_db, allow_fetch=True)

    assert second.status == "succeeded"
    assert second.signals_count == 1
    assert len(llm_calls) == 2
    rows = career_db.query(CareerSignal).all()
    assert len(rows) == 1
    assert rows[0].company == "Tata Consultancy Services"
    assert rows[0].action == "LAYOFF"
    assert rows[0].count == 200
    extraction = _extraction(career_db)
    assert extraction.text_source == "full_body"
    assert extraction.signals_count == 1


def test_fetch_failure_falls_back_to_headline_only_and_emits_nothing(career_db, make_article, monkeypatch):
    """When only the stored summary is available, no LLM call is made and no signals are stored."""
    article = make_article(
        summary="TCS will hire 200 engineers in Pune, India. The roles require Java.",
    )
    _mock_body(monkeypatch, FetchResult("failed", error="timeout"))
    llm_calls = []
    _mock_llm(monkeypatch, [], calls=llm_calls)

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.text_source == "stored_summary"
    assert outcome.signals_count == 0
    assert outcome.error == "headline_only"
    assert llm_calls == []
    assert career_db.query(CareerSignal).count() == 0

    extraction = _extraction(career_db)
    assert extraction.status == "succeeded"
    assert extraction.text_source == "stored_summary"
    assert extraction.signals_count == 0
    body = career_db.query(ArticleBody).filter_by(article_id=article.id).one()
    assert body.status == "failed"
    assert body.attempts == 1


def test_paywalled_fetch_is_headline_only_and_emits_nothing(career_db, make_article, monkeypatch):
    article = make_article(
        summary="TCS will hire 200 engineers in Pune, India.",
    )
    _mock_body(monkeypatch, FetchResult("paywalled", http_status=402, error="auth_required"))
    llm_calls = []
    _mock_llm(monkeypatch, [], calls=llm_calls)

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.text_source == "stored_summary"
    assert outcome.signals_count == 0
    assert llm_calls == []
    assert career_db.query(ArticleBody).one().status == "paywalled"


def test_cache_only_mode_skips_articles_without_cached_body(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    llm_calls = []
    _mock_llm(monkeypatch, [], calls=llm_calls)

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=False)

    assert outcome.status == "no_body"
    assert outcome.error == "body_not_cached"
    assert llm_calls == []
    assert career_db.query(CareerExtraction).count() == 0


def test_cache_only_mode_processes_cached_body(career_db, make_article, monkeypatch):
    article = make_article()
    career_db.add(
        ArticleBody(
            article_id=article.id,
            body_text=BODY_TEXT,
            status="ok",
            attempts=1,
            http_status=200,
        )
    )
    career_db.commit()

    _mock_body(monkeypatch)
    llm_calls = []
    _mock_llm(monkeypatch, [json.dumps([_signal()])], calls=llm_calls)

    stats = _extractor().process_pending_articles(
        db=career_db, limit=1, allow_fetch=False
    )

    assert stats.succeeded == 1
    assert stats.processed == 1
    assert len(llm_calls) == 1
    assert career_db.query(CareerSignal).count() == 1
    assert _extraction(career_db).text_source == "full_body"


def test_vercel_batch_limit(monkeypatch):
    monkeypatch.delenv("CAREER_EXTRACTION_BATCH_LIMIT", raising=False)
    monkeypatch.delenv("CAREER_EXTRACTION_BATCH_LIMIT_VERCEL", raising=False)

    monkeypatch.setenv("VERCEL", "1")
    assert default_batch_limit() == 3

    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("NOW_REGION", raising=False)
    assert default_batch_limit() == 25


# ---------------------------------------------------------------------------
# Prompt v4 rules, filters and rate limiting
# ---------------------------------------------------------------------------

def test_prompt_version_is_part_of_content_hash(monkeypatch):
    """Changing PROMPT_VERSION must change the hash so articles are re-extracted."""
    base = extractor_module.compute_content_hash("full_body", "same text")
    monkeypatch.setattr(extractor_module, "PROMPT_VERSION", "vNEXT")
    assert extractor_module.compute_content_hash("full_body", "same text") != base


def test_prompt_version_is_v4():
    from backend.services.career.prompt import PROMPT_VERSION

    assert PROMPT_VERSION == "v5"


def test_prompt_does_not_offer_other_action():
    """OTHER is removed from the action list the model sees."""
    from backend.services.career.prompt import CAREER_SYSTEM_PROMPT as prompt

    assert "OTHER" not in prompt
    assert "Only these six actions exist" in prompt
    assert "regulator guidelines" in prompt


def test_prompt_rule_count_must_come_from_announcement_not_estimate():
    """An executive's estimate must give count=null and no HIRING signal."""
    from backend.services.career.prompt import CAREER_SYSTEM_PROMPT as prompt

    assert "ANNOUNCED hiring or layoff" in prompt
    assert "executive's estimate" in prompt
    assert "I need to onboard 30 lakh people a year" in prompt
    assert "count MUST be null" in prompt
    assert "emit no HIRING or LAYOFF signal" in prompt


def test_prompt_rule_roundups_are_skipped():
    from backend.services.career.prompt import CAREER_SYSTEM_PROMPT as prompt

    assert "ROUNDUPS" in prompt
    for kind in ("roundup", "weekly quiz", "newsletter", "deals list", "listicle"):
        assert kind in prompt


def test_other_action_is_dropped_and_counted(career_db, make_article, monkeypatch):
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(
        monkeypatch,
        [json.dumps([
            _signal(action="OTHER"),
            _signal(action="HIRING"),
        ])],
    )

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "succeeded"
    assert outcome.dropped.other_action == 1
    assert outcome.signals_count == 1
    assert career_db.query(CareerSignal).one().action == "HIRING"


def test_old_other_rows_remain_valid_in_the_schema(career_db, make_article):
    """OTHER stays in ACTIONS and in the table so rows written by older versions load fine."""
    from backend.services.career.schemas import ACTIONS

    article = make_article()
    assert "OTHER" in ACTIONS
    career_db.add(CareerSignal(
        source_article_id=article.id, company="Nvidia", action="OTHER",
        timing="current", confidence=0.9,
    ))
    career_db.commit()
    assert career_db.query(CareerSignal).one().action == "OTHER"


@pytest.mark.parametrize("title", [
    "Space, sustainability, AI: test your business creativity with Edition 258 of our weekly quiz!",
    "The Download newsletter: AI hiring edition",
    "Startup funding roundup: 12 deals this week",
    "Weekly funding report",
    "Black Friday deals on laptops",
    "Indian Listed New-Age Tech Company Tracker: Market Cap, Revenue & More",
    "Top 10 AI startups to watch",
    "The 7 best tech jobs in 2027",
    "New-Age Tech Stocks Continue Rally; E2E Network, Turtlemint Lead Gains",
    "Sensex surges as IT stocks rally",
])
def test_roundup_titles_are_detected(title):
    from backend.services.career.filters import roundup_reason

    assert roundup_reason(title) is not None


@pytest.mark.parametrize("title", [
    "Zeta Systems to lay off 300 employees in Austin",
    "TCS will hire 40,000 freshers this year",
    "Eternal reworking 2030 EV delivery target amid fleet expansion",
    "Anup Gupta To Exit Nexus Venture Partners After 16 Years",
    "Ajinkya Rahane Picks Up Stake In D2C Cookware Brand Ember",
    "",
    None,
])
def test_normal_titles_are_not_roundups(title):
    from backend.services.career.filters import roundup_reason

    assert roundup_reason(title) is None


def test_roundup_title_skips_fetch_and_llm_and_marks_ledger(career_db, make_article, monkeypatch):
    article = make_article(
        article_id="quiz-1",
        title="Test your business creativity with Edition 258 of our weekly quiz!",
        link="https://example.com/quiz",
    )
    fetch_counter = {"count": 0}
    _mock_body(monkeypatch, counter=fetch_counter)
    llm_calls = []
    _mock_llm(monkeypatch, [], calls=llm_calls)

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.status == "skipped_roundup"
    assert fetch_counter["count"] == 0
    assert llm_calls == []
    extraction = _extraction(career_db, "quiz-1")
    assert extraction.status == "succeeded"
    assert extraction.signals_count == 0
    assert extraction.text_source == "skipped_roundup"
    assert career_db.query(CareerSignal).count() == 0
    assert career_db.query(ArticleBody).count() == 0


def test_roundup_skip_removes_existing_signals_and_is_counted_in_stats(career_db, make_article, monkeypatch):
    article = make_article(
        article_id="tracker-1",
        title="Indian Listed New-Age Tech Company Tracker: Market Cap, Revenue & More",
        link="https://example.com/tracker",
    )
    career_db.add(CareerSignal(
        source_article_id=article.id, company="Zomato", action="FUNDING",
        timing="current", confidence=0.9,
    ))
    career_db.commit()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [])

    stats = _extractor().process_pending_articles(db=career_db, limit=5, allow_fetch=True)

    assert stats.skipped_roundup == 1
    assert stats.processed == 0
    assert stats.as_dict()["skipped_roundup"] == 1
    assert career_db.query(CareerSignal).count() == 0

    # Second run is idempotent and still reported as a roundup skip.
    stats = _extractor().process_pending_articles(db=career_db, limit=5, allow_fetch=True)
    assert stats.skipped_roundup == 1
    assert career_db.query(CareerExtraction).count() == 1


def test_headline_only_is_counted_in_stats(career_db, make_article, monkeypatch):
    make_article(summary="TCS will hire 200 engineers in Pune, India.")
    _mock_body(monkeypatch, FetchResult("failed", error="timeout"))
    _mock_llm(monkeypatch, [])

    stats = _extractor().process_pending_articles(db=career_db, limit=5, allow_fetch=True)

    assert stats.headline_only == 1
    assert stats.zero_signal == 1
    assert stats.as_dict()["headline_only"] == 1


def test_full_body_low_confidence_signal_is_not_capped(career_db, make_article, monkeypatch):
    """The old 0.6 cap and the 0.7 headline gate are gone; confidence is stored as reported."""
    article = make_article()
    _mock_body(monkeypatch)
    _mock_llm(monkeypatch, [json.dumps([_signal(confidence=0.95)])])

    outcome = _extractor().process_article(article.id, db=career_db, allow_fetch=True)

    assert outcome.text_source == "full_body"
    assert career_db.query(CareerSignal).one().confidence == 0.95
    assert not hasattr(outcome.dropped, "summary_low_confidence")
    assert not hasattr(outcome.dropped, "confidence_capped")


def test_llm_call_delay_spaces_consecutive_calls(monkeypatch):
    sleeps = []
    clock = {"now": 100.0}
    monkeypatch.setattr(extractor_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(extractor_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        extractor_module, "call_groq", lambda *a, **k: ("[]", "test-model")
    )
    extractor = CareerExtractor(api_key="k", fetch_delay_seconds=0, llm_call_delay_seconds=1.0)

    extractor._call_llm([{"role": "user", "content": "a"}])
    assert sleeps == []  # first call never waits

    clock["now"] = 100.25
    extractor._call_llm([{"role": "user", "content": "b"}])
    assert sleeps == [0.75]


def test_llm_call_delay_reads_env_with_default(monkeypatch):
    monkeypatch.delenv("CAREER_LLM_CALL_DELAY_SECONDS", raising=False)
    assert CareerExtractor(api_key="k").llm_call_delay_seconds == 1.0
    monkeypatch.setenv("CAREER_LLM_CALL_DELAY_SECONDS", "2.5")
    assert CareerExtractor(api_key="k").llm_call_delay_seconds == 2.5
    monkeypatch.setenv("CAREER_LLM_CALL_DELAY_SECONDS", "not-a-number")
    assert CareerExtractor(api_key="k").llm_call_delay_seconds == 1.0


def test_rejected_configured_model_warns_once_and_is_dropped(caplog):
    """A configured model that Groq rejected is warned about once, then not requested again."""
    extractor = CareerExtractor(api_key="test-key", model="llama3-8b-8192", fetch_delay_seconds=0)

    with caplog.at_level("WARNING", logger="career_radar"):
        extractor._note_model_fallback("openai/gpt-oss-120b")
        assert extractor.model is None
        extractor._note_model_fallback("openai/gpt-oss-120b")

    warnings = [r for r in caplog.records if "was rejected by Groq" in r.getMessage()]
    assert len(warnings) == 1
    assert "llama3-8b-8192" in warnings[0].getMessage()
