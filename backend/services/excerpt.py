"""RSS excerpt handling.

* clean_excerpt(): turn an RSS excerpt into short plain text (used by the scraper for feeds
  with `max_summary_chars`, and at serve time for older rows).
* public_summary(): what public endpoints may return as `summary`. For sources whose terms
  restrict reproduction (RESTRICTED_EXCERPT_SOURCES) it never returns the raw excerpt.
"""
import html
import re
from typing import Optional

# Sources whose RSS excerpt must not be served as-is. Public endpoints return our own
# llm_summary (or a short cleaned excerpt when there is none).
RESTRICTED_EXCERPT_SOURCES = frozenset({"HRKatha"})
RESTRICTED_EXCERPT_MAX_CHARS = 300

_TAGS = re.compile(r"<[^>]+>")
_MORE_MARKER = re.compile(r"\[\s*(?:\.\.\.|\u2026)\s*\]")
_FOOTER = re.compile(r"\s*The post\b.*?\bappeared first on\b.*$", re.IGNORECASE | re.DOTALL)
_SPACES = re.compile(r"\s+")


def clean_excerpt(summary: Optional[str], max_chars: Optional[int] = None) -> str:
    """Plain text version of an RSS excerpt.

    Strips HTML, removes the "[...]" marker and the trailing "The post ... appeared first
    on ..." footer, then (when max_chars is given) cuts at a word boundary so the result is
    never longer than max_chars.
    """
    if not summary:
        return ""
    text = html.unescape(str(summary))
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)  # entities that were double-encoded
    text = _FOOTER.sub("", text)
    text = _MORE_MARKER.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if text[max_chars] != " ":  # we are inside a word: drop the partial word
        head, sep, _ = cut.rpartition(" ")
        cut = head if sep else cut
    return cut.rstrip(" ,;:-\u2013\u2014")


def is_restricted_source(source: Optional[str]) -> bool:
    return (source or "") in RESTRICTED_EXCERPT_SOURCES


def public_summary(source: Optional[str], summary: Optional[str], llm_summary: Optional[str]) -> str:
    """The `summary` value a public endpoint may return.

    Other sources: unchanged (`summary or ""`). Restricted sources: llm_summary, or a short
    cleaned excerpt when llm_summary is empty.
    """
    if not is_restricted_source(source):
        return summary or ""
    if llm_summary and llm_summary.strip():
        return llm_summary
    return clean_excerpt(summary, RESTRICTED_EXCERPT_MAX_CHARS)
