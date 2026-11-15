"""Deterministic text cleaning for user-supplied profile strings."""
import re
import unicodedata

# Whitespace-like control characters become a space so "a\tb" does not turn into "ab".
_SPACE_LIKE = {"\t", "\n", "\r", "\x0b", "\x0c", "\x85", "\u2028", "\u2029"}


def clean_text(value: str) -> str:
    """Strip control/format characters, collapse whitespace, trim.

    Removes every Unicode "C" category character (control, format such as zero-width
    and bidi marks, surrogate, private use, unassigned), except that tab/newline-style
    characters are turned into a single space first.
    """
    if not isinstance(value, str):
        raise TypeError("expected a string")
    out = []
    for ch in value:
        if ch in _SPACE_LIKE:
            out.append(" ")
        elif unicodedata.category(ch).startswith("C"):
            continue
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()
