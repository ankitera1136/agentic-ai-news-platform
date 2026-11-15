"""Cheap title-based filters that run before any body fetch or LLM call.

Roundups, quizzes, newsletters, deal lists, trackers and market reports mention many
companies in passing. They are not about one company's workforce, so CareerRadar skips
them instead of paying for an LLM call that mostly produces noise.
"""
import re
from typing import Optional

_MARKET_WORDS = r"(?:stock|stocks|share|shares|sensex|nifty|market|markets|equities)"
_MOVE_WORDS = (
    r"(?:rally|rallies|rallying|surge|surges|slump|slumps|tumble|tumbles|"
    r"plunge|plunges|jump|jumps|gain|gains)"
)

# (reason, compiled pattern). Matched against the article title only.
_ROUNDUP_PATTERNS = [
    ("quiz", re.compile(r"\bquiz(?:zes)?\b", re.IGNORECASE)),
    ("newsletter", re.compile(r"\bnewsletters?\b", re.IGNORECASE)),
    ("roundup", re.compile(r"\bround-?ups?\b", re.IGNORECASE)),
    ("weekly", re.compile(r"\bweekly\b", re.IGNORECASE)),
    ("deals", re.compile(r"\bdeals\b", re.IGNORECASE)),
    ("tracker", re.compile(r"\btrackers?\b", re.IGNORECASE)),
    ("top_n_list", re.compile(r"\b(?:top|best)\s+\d+\b", re.IGNORECASE)),
    ("n_best_list", re.compile(r"\b\d+\s+(?:best|top)\b", re.IGNORECASE)),
    (
        "market_report",
        re.compile(rf"\b{_MARKET_WORDS}\b.{{0,60}}\b{_MOVE_WORDS}\b", re.IGNORECASE),
    ),
    ("market_report", re.compile(r"\brall(?:y|ies)\b", re.IGNORECASE)),
]


def roundup_reason(title: Optional[str]) -> Optional[str]:
    """Return a short reason when the title looks like a roundup-style article, else None."""
    text = (title or "").strip()
    if not text:
        return None
    for reason, pattern in _ROUNDUP_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def is_roundup_title(title: Optional[str]) -> bool:
    return roundup_reason(title) is not None
