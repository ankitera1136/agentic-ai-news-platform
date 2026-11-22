"""Deterministic grounding and validation helpers for CareerRadar.

Every check here is intentionally conservative: values that cannot be tied back to
the text sent to the LLM are dropped or set to null rather than guessed.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.services.claim_extractor import verify_quote_grounding

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_for_match(text: Optional[str]) -> str:
    """Lowercase and collapse whitespace while preserving tech-skill punctuation."""
    if not text:
        return ""
    value = str(text).lower()
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"[^a-z0-9\s+#.&/-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def verify_quote(quote: str, source_text: str, threshold: float = 0.70) -> bool:
    """Keep the existing 70% token-overlap fallback used by claim_extractor."""
    return verify_quote_grounding(quote, source_text, threshold=threshold)


# ---------------------------------------------------------------------------
# Number parsing (count formats)
# ---------------------------------------------------------------------------

_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "k": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "cr": 10_000_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
}

_PEOPLE_UNITS = {
    "people", "person", "persons", "employee", "employees", "staff",
    "worker", "workers", "job", "jobs", "role", "roles", "headcount",
    "engineer", "engineers", "developer", "developers", "worker",
}

_PERCENT_UNITS = {"%", "percent", "per cent"}

_CURRENCY_MARKERS = re.compile(
    r"[$₹€£]|\b(?:usd|inr|dollars?|rupees?|crore rupees|lakh rupees|"
    r"funding|valuation|revenue|profit|contract|deal|investment)\b",
    re.IGNORECASE,
)

_APPROX_PREFIXES = re.compile(
    r"^(?:about|around|approximately|approx\.?|nearly|roughly|some|"
    r"over|more than|at least|up to|as many as|close to)\s+",
    re.IGNORECASE,
)

_DIGIT_MENTION_RE = re.compile(
    r"(?<![\w.])(?P<num>\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)"
    r"(?P<unit>\s*(?:%|percent|per\s+cent|k|thousand|lakhs?|lacs?|crores?|cr|"
    r"million|mn|m|billion|bn|b|people|persons?|employees?|staff|workers?|"
    r"jobs?|roles?|headcount|engineers?|developers?))?",
    re.IGNORECASE,
)

_WORD_TOKEN = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|lakhs|"
    r"lac|lacs|crore|crores|million|billion|dozen|a|an|and)"
)
_WORD_PHRASE_RE = re.compile(rf"\b{_WORD_TOKEN}(?:[\s-]+{_WORD_TOKEN}){{0,6}}\b", re.IGNORECASE)


def _to_int(value: float) -> Optional[int]:
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return None


def parse_word_number(phrase: str) -> Optional[int]:
    """Parse number words such as 'forty thousand', 'four lakh', 'a dozen'."""
    tokens = [t for t in re.split(r"[\s-]+", phrase.lower().strip()) if t]
    if not tokens:
        return None
    if "dozen" in tokens:
        return 12 if len(tokens) <= 2 else None

    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token in ("and",):
            continue
        if token in ("a", "an"):
            current += 1
            saw_number = True
        elif token in _WORD_UNITS:
            current += _WORD_UNITS[token]
            saw_number = True
        elif token in _SCALES:
            scale = _SCALES[token]
            if current == 0:
                current = 1
            if scale == 100:
                current *= 100
            else:
                total += current * scale
                current = 0
            saw_number = True
        else:
            return None
    if not saw_number:
        return None
    return total + current


def parse_count_expression(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Convert an LLM count value into an integer, or explain why it is not a count.

    Returns (count, reason) where reason is one of: percentage, not_people,
    invalid, or None on success.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, "invalid"
    if isinstance(value, int):
        return (value, None) if value >= 0 else (None, "invalid")
    if isinstance(value, float):
        converted = _to_int(value)
        return (converted, None) if converted is not None and converted >= 0 else (None, "invalid")

    text = str(value).strip()
    if not text:
        return None, "invalid"
    lowered = text.lower()

    if re.search(r"%|\bpercent\b|\bper\s+cent\b", lowered):
        return None, "percentage"
    if _CURRENCY_MARKERS.search(lowered):
        return None, "not_people"

    cleaned = _APPROX_PREFIXES.sub("", lowered).strip().strip(".,;")
    cleaned = cleaned.replace(",", "")

    digit = _DIGIT_MENTION_RE.search(cleaned)
    if digit:
        number = float(digit.group("num"))
        unit = (digit.group("unit") or "").strip().lower()
        if unit in _PERCENT_UNITS:
            return None, "percentage"
        if unit in _SCALES:
            number *= _SCALES[unit]
        elif unit and unit not in _PEOPLE_UNITS:
            return None, "invalid"
        converted = _to_int(number)
        return (converted, None) if converted is not None and converted >= 0 else (None, "invalid")

    word_value = parse_word_number(cleaned)
    if word_value is not None:
        return word_value, None
    return None, "invalid"


@dataclass(frozen=True)
class NumberMention:
    value: int
    is_percentage: bool


def iter_number_mentions(text: str) -> Iterable[NumberMention]:
    """Yield numeric mentions, including 40,000 / 40k / 4 lakh / forty thousand."""
    if not text:
        return
    for match in _DIGIT_MENTION_RE.finditer(text):
        number = float(match.group("num").replace(",", ""))
        unit = (match.group("unit") or "").strip().lower()
        is_percentage = unit in _PERCENT_UNITS
        if unit in _SCALES:
            number *= _SCALES[unit]
        converted = _to_int(number)
        if converted is not None:
            yield NumberMention(converted, is_percentage)

    for match in _WORD_PHRASE_RE.finditer(text):
        phrase = match.group(0)
        # Skip single filler words such as "a" / "and".
        if phrase.strip().lower() in ("a", "an", "and"):
            continue
        converted = parse_word_number(phrase)
        if converted is not None:
            yield NumberMention(converted, False)


def count_appears_in_text(count: int, text: str) -> Tuple[bool, Optional[str]]:
    """Return (grounded, reason). reason='percentage' when the only equal mention is a percentage."""
    percentage_match = False
    for mention in iter_number_mentions(text or ""):
        if mention.value == count:
            if mention.is_percentage:
                percentage_match = True
            else:
                return True, None
    return False, ("percentage" if percentage_match else "not_grounded")


def ground_count(
    raw_count: Any,
    source_text: str,
    evidence_quote: str = "",
) -> Tuple[Optional[int], Optional[str]]:
    """Parse and ground a count against the evidence quote first, then full text."""
    parsed, reason = parse_count_expression(raw_count)
    if reason:
        return None, reason
    if parsed is None:
        return None, None

    if evidence_quote:
        grounded, evidence_reason = count_appears_in_text(parsed, evidence_quote)
        if grounded:
            return parsed, None
        if evidence_reason == "percentage":
            return None, "percentage"

    grounded, text_reason = count_appears_in_text(parsed, source_text)
    if grounded:
        return parsed, None
    if text_reason == "percentage":
        return None, "percentage"
    return None, "not_grounded"


# ---------------------------------------------------------------------------
# Estimates are not counts
# ---------------------------------------------------------------------------
# A number in a hedged sentence ("estimates suggest around 1,000 employees", "reportedly
# cutting nearly 1,000 jobs", "has not been confirmed") is an estimate, not a stated count.
# Cues that hedge the whole sentence.
ESTIMATE_CUES = re.compile(
    r"\b(?:estimates?|estimated|reportedly|could|may)\b"
    r"|\bnot\s+(?:yet\s+)?been\s+confirmed\b|\bnot\s+(?:yet\s+)?confirmed\b|\bunconfirmed\b",
    re.IGNORECASE,
)
# Cues that hedge only the number right after them ("nearly 1,000 jobs"), not another number in
# the same sentence ("about 22 per cent, affecting 1,271 employees").
_NUMBER_HEDGES = re.compile(r"\b(?:around|approximately|nearly|about)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c\u2018(])")


def _first_mention_after(text: str, start: int) -> Optional["NumberMention"]:
    """The first number mentioned in `text[start:]` (digits or words), or None."""
    tail = text[start:]
    spans = []
    for m in _DIGIT_MENTION_RE.finditer(tail):
        number = float(m.group("num").replace(",", ""))
        unit = (m.group("unit") or "").strip().lower()
        pct = unit in _PERCENT_UNITS
        if unit in _SCALES:
            number *= _SCALES[unit]
        converted = _to_int(number)
        if converted is not None:
            spans.append((m.start(), NumberMention(converted, pct)))
    for m in _WORD_PHRASE_RE.finditer(tail):
        if m.group(0).strip().lower() in ("a", "an", "and"):
            continue
        converted = parse_word_number(m.group(0))
        if converted is not None:
            spans.append((m.start(), NumberMention(converted, False)))
    return min(spans, key=lambda x: x[0])[1] if spans else None


def estimate_cue_in_count_sentence(count: Optional[int], quote: Optional[str]) -> Optional[str]:
    """The estimate cue found in a sentence of `quote` that states `count`, else None.

    Only sentences that contain the number are checked. Sentence-wide cues (estimate,
    reportedly, could, may, "not been confirmed") apply anywhere in that sentence; around /
    approximately / nearly / about apply only when the count is the number they introduce.
    A count that is not in the quote is left alone.
    """
    if count is None or not quote:
        return None
    for sentence in _SENTENCE_SPLIT.split(quote):
        if not any(m.value == count and not m.is_percentage for m in iter_number_mentions(sentence)):
            continue
        cue = ESTIMATE_CUES.search(sentence)
        if cue:
            return cue.group(0).lower()
        for hedge in _NUMBER_HEDGES.finditer(sentence):
            first = _first_mention_after(sentence, hedge.end())
            if first is not None and first.value == count and not first.is_percentage:
                return hedge.group(0).lower()
    return None


def count_is_estimated(count: Optional[int], quote: Optional[str]) -> bool:
    return estimate_cue_in_count_sentence(count, quote) is not None


# ---------------------------------------------------------------------------
# Skills and locations
# ---------------------------------------------------------------------------

def ground_skills(skills: List[str], source_text: str) -> Tuple[List[str], List[str]]:
    """Keep skills whose phrase appears in the text sent to the LLM."""
    normalized_source = normalize_for_match(source_text)
    kept: List[str] = []
    dropped: List[str] = []
    seen = set()

    for raw in skills or []:
        skill = str(raw).strip().strip(".,;:")
        if not skill:
            continue
        key = normalize_for_match(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in normalized_source:
            kept.append(skill)
        else:
            dropped.append(skill)
    return kept, dropped


# ---------------------------------------------------------------------------
# Job titles are not skills
# ---------------------------------------------------------------------------
# "engineer", "senior engineer", "software engineer" are roles, not skills. A skill string
# that is only a title is dropped; a title suffix on a technology is removed
# ("Python developer" -> "Python").
_TITLE_WORDS = {
    "engineer", "engineers", "developer", "developers", "manager", "managers",
    "analyst", "analysts", "intern", "interns", "associate", "associates",
    "lead", "leads", "director", "directors", "executive", "executives",
    "staff", "employee", "employees", "worker", "workers",
}
_SENIORITY_WORDS = {"senior", "junior", "sr", "jr", "principal", "chief", "head"}
# What is left after removing a title is not a skill either ("Data analyst" -> "Data").
_GENERIC_REMAINDERS = {
    "software", "data", "it", "tech", "technical", "engineering", "development", "team",
    "product", "project", "business", "web", "and", "&",
}


def _bare(token: str) -> str:
    return token.strip(" .,;:()[]\"'").lower()


def clean_job_title_skill(skill: str) -> Optional[str]:
    """Return the skill without job-title words, or None when nothing skill-like remains."""
    tokens = str(skill or "").split()
    if not tokens:
        return None
    # Leading seniority ("Senior Python developer" -> "Python developer").
    while len(tokens) > 1 and _bare(tokens[0]) in _SENIORITY_WORDS:
        tokens = tokens[1:]
    # Trailing title words ("Python developer" -> "Python").
    while tokens and _bare(tokens[-1]) in _TITLE_WORDS:
        tokens = tokens[:-1]
    # Trailing seniority/generic words left behind ("Python senior" is unlikely, "software").
    remaining = [t for t in tokens if _bare(t)]
    if not remaining:
        return None
    if all(_bare(t) in _SENIORITY_WORDS | _GENERIC_REMAINDERS | _TITLE_WORDS for t in remaining):
        return None
    return " ".join(remaining)


def clean_job_title_skills(skills: Iterable[str]) -> Tuple[List[str], int]:
    """Apply clean_job_title_skill to a list. Returns (cleaned skills, number dropped entirely)."""
    cleaned: List[str] = []
    dropped = 0
    for raw in skills or []:
        text = str(raw).strip()
        if not text:
            continue
        result = clean_job_title_skill(text)
        if result is None:
            dropped += 1
        else:
            cleaned.append(result)
    return cleaned, dropped


# ---------------------------------------------------------------------------
# Generic non-skills
# ---------------------------------------------------------------------------
# Business functions and umbrella words are not learnable skills for a student's gap
# analysis. Dropped (whole-string match, case-insensitive, optional trailing "skills"):
#   functions/departments: digital, finance, accounting, human resources, HR, procurement,
#       marketing, legal, sales, operations, customer service/support, administration,
#       supply chain, management
#   umbrella words: business, technology, tech, software, engineering, IT, data
#   vague AI umbrella terms: AI, AI models, AI tools, artificial intelligence (AI) --
#       "machine learning", "generative AI", "LLM", "computer vision" etc. are KEPT.
GENERIC_SKILLS = {
    "digital", "finance", "accounting", "human resources", "hr", "procurement", "marketing",
    "legal", "sales", "operations", "customer service", "customer support", "administration",
    "supply chain", "management", "business", "technology", "tech", "software", "engineering",
    "it", "data", "digital transformation",
    "ai", "ai models", "ai tools", "artificial intelligence", "artificial intelligence (ai)",
}


def is_generic_skill(skill: str) -> bool:
    key = re.sub(r"\s+", " ", str(skill or "").strip().strip(".,;:").lower())
    key = re.sub(r"\s+skills?$", "", key)
    return key in GENERIC_SKILLS


def drop_generic_skills(skills: Iterable[str]) -> Tuple[List[str], int]:
    """Remove generic non-skills. Returns (kept, number dropped)."""
    kept: List[str] = []
    dropped = 0
    for raw in skills or []:
        if is_generic_skill(raw):
            dropped += 1
        elif str(raw).strip():
            kept.append(raw)
    return kept, dropped


_COUNTRY_ALIASES = {
    "united states": {
        "united states", "united states of america", "usa", "us", "u.s.", "u.s.a.", "america"
    },
    "united kingdom": {"united kingdom", "uk", "u.k.", "britain", "great britain"},
    "united arab emirates": {"united arab emirates", "uae", "u.a.e."},
    "south korea": {"south korea", "korea", "republic of korea"},
    "india": {"india", "bharat"},
    "china": {"china", "prc"},
    "japan": {"japan"},
    "germany": {"germany"},
    "france": {"france"},
    "canada": {"canada"},
    "australia": {"australia"},
    "singapore": {"singapore"},
}


def _country_is_grounded(country: str, normalized_source: str) -> bool:
    key = normalize_for_match(country)
    if not key:
        return False
    if key in normalized_source:
        return True
    for canonical, aliases in _COUNTRY_ALIASES.items():
        if key == canonical or key in aliases:
            return any(normalize_for_match(alias) in normalized_source for alias in aliases)
    return False


def ground_location(
    city: Optional[str],
    country: Optional[str],
    source_text: str,
) -> Tuple[Optional[str], Optional[str], bool]:
    """Null out location parts that are not explicitly present in the text."""
    normalized_source = normalize_for_match(source_text)
    dropped = False

    clean_city = str(city).strip() if city else None
    clean_country = str(country).strip() if country else None

    if clean_city:
        city_key = normalize_for_match(clean_city)
        if not city_key or city_key not in normalized_source:
            clean_city = None
            dropped = True

    if clean_country:
        if not _country_is_grounded(clean_country, normalized_source):
            clean_country = None
            dropped = True

    return clean_city, clean_country, dropped


# ---------------------------------------------------------------------------
# Drop accounting
# ---------------------------------------------------------------------------

@dataclass
class DropCounters:
    invalid_item: int = 0
    invalid_action: int = 0
    invalid_timing: int = 0
    other_action: int = 0
    past_context: int = 0
    missing_company: int = 0
    low_confidence: int = 0
    invalid_confidence: int = 0
    evidence_quote: int = 0
    skill_not_grounded: int = 0
    job_title_skill: int = 0
    generic_skill: int = 0
    estimated_count: int = 0
    location_not_grounded: int = 0
    count_percentage: int = 0
    count_not_grounded: int = 0
    count_invalid: int = 0
    duplicate: int = 0
    unknown_company: int = 0
    verified: int = 0

    def add(self, field: str, amount: int = 1) -> None:
        setattr(self, field, getattr(self, field, 0) + amount)

    def merge(self, other: "DropCounters") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key, 0) + value)

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def clamp_confidence(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


