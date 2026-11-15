import pytest

from backend.services.career.grounding import (
    count_appears_in_text,
    ground_count,
    ground_location,
    ground_skills,
    parse_count_expression,
)
from backend.services.career.normalizer import normalize_company


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("40,000", 40000),
        ("40k", 40000),
        ("forty thousand", 40000),
        ("4 lakh", 400000),
        ("4,00,000", 400000),
        ("1.5 lakh", 150000),
        ("about 500", 500),
        ("500 employees", 500),
        ("12", 12),
        (250, 250),
        (250.0, 250),
    ],
)
def test_parse_count_formats(raw, expected):
    assert parse_count_expression(raw) == (expected, None)


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("10% of workforce", "percentage"),
        ("10 percent", "percentage"),
        ("$40,000", "not_people"),
        ("forty thousand dollars", "not_people"),
        ("hundreds", "invalid"),
        ("a majority", "invalid"),
        (250.5, "invalid"),
        (-5, "invalid"),
    ],
)
def test_parse_count_rejects_non_people_values(raw, reason):
    value, actual_reason = parse_count_expression(raw)
    assert value is None
    assert actual_reason == reason


@pytest.mark.parametrize(
    "text,count,expected",
    [
        ("The company cut 40,000 jobs.", 40000, True),
        ("The company cut 40k jobs.", 40000, True),
        ("The company cut forty thousand jobs.", 40000, True),
        ("The company cut 4 lakh jobs.", 400000, True),
        ("The company cut 10% of its workforce.", 10, False),
        ("The company cut 10% of its workforce.", 40000, False),
    ],
)
def test_count_grounding_formats(text, count, expected):
    grounded, _ = count_appears_in_text(count, text)
    assert grounded is expected


def test_percentage_count_becomes_null():
    count, reason = ground_count("10%", "The firm will cut 10% of its workforce.", "")
    assert count is None
    assert reason == "percentage"


def test_ungrounded_count_becomes_null():
    count, reason = ground_count(500, "The firm will hire engineers in Pune.", "")
    assert count is None
    assert reason == "not_grounded"


def test_count_falls_back_to_source_when_not_in_quote():
    source = "Background: 500 users joined. The firm will hire 120 engineers."
    count, reason = ground_count(
        500,
        source,
        "The firm will hire 120 engineers.",
    )
    # 500 is in the full text, so the evidence-first check falls back to source.
    assert count == 500
    assert reason is None


def test_skills_must_appear_in_text():
    source = "The team uses Python and Kubernetes. Rust is not mentioned as a requirement."
    kept, dropped = ground_skills(["Python", "Kubernetes", "COBOL"], source)
    assert kept == ["Python", "Kubernetes"]
    assert dropped == ["COBOL"]


def test_skill_phrase_is_case_insensitive():
    kept, dropped = ground_skills(["python", "Machine Learning"], "Python and machine learning roles.")
    assert kept == ["python", "Machine Learning"]
    assert dropped == []


def test_location_parts_must_appear():
    city, country, dropped = ground_location("Bengaluru", "India", "Hiring in Bengaluru, India.")
    assert (city, country, dropped) == ("Bengaluru", "India", False)

    city, country, dropped = ground_location("Mumbai", "India", "Hiring in Bengaluru, India.")
    assert (city, country, dropped) == (None, "India", True)


def test_country_alias_is_grounded():
    city, country, dropped = ground_location("Austin", "United States", "Expanding in Austin, US.")
    assert (city, country, dropped) == ("Austin", "United States", False)


@pytest.mark.parametrize(
    "raw,canonical,matched",
    [
        ("TCS", "Tata Consultancy Services", True),
        ("Tata Consultancy Services Ltd.", "Tata Consultancy Services", True),
        ("Infosys Technologies Pvt Ltd", "Infosys", True),
        ("HCLTech", "HCL Technologies", True),
        ("L&T Infotech", "LTIMindtree", True),
        ("Acme Robotics Pvt Ltd", "Acme Robotics", False),
    ],
)
def test_company_normalization(raw, canonical, matched):
    result = normalize_company(raw)
    assert result.canonical == canonical
    assert result.matched_alias is matched
