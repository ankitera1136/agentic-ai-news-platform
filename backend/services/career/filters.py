"""Cheap title-based filters that run before any body fetch or LLM call.

Roundups, quizzes, newsletters, deal lists, trackers and market reports mention many
companies in passing. They are not about one company's workforce, so CareerRadar skips
them instead of paying for an LLM call that mostly produces noise.
"""
import html
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


# ---------------------------------------------------------------------------
# Career relevance pre-filter
# ---------------------------------------------------------------------------
# Decides from the title and RSS summary alone whether an article could contain a
# workforce event (layoffs, hiring, job cuts, new offices). Bare words such as
# "fired", "jobs", "recruit", "hires", "restructuring" and "employees" are NOT enough:
# they matched "fired Sam Altman", "technical recruiter", "software job titles" and
# "Nvidia hires law firm" in real data. Each pattern needs a workforce context.
#
# Bump FILTER_VERSION whenever the patterns change; it is part of the ledger content
# hash so rows skipped under an older version are re-evaluated.
FILTER_VERSION = "career-filter-v2"

_NUM = r"(?:\d[\d,.]*\s?(?:k|lakh|crore|million|%)?|one|two|three|half|hundreds|thousands)"
_PEOPLE = r"(?:jobs|roles|positions|staff|employees|workers|people|engineers|workforce)"
_JOBS = r"(?:jobs|roles|positions|staff|employees|workers)"

_RELEVANCE_PATTERNS = [
    ("layoff", re.compile(r"\blay-?offs?\b|\blay(?:s|ing)? off\b|\blaid off\b", re.IGNORECASE)),
    ("let_go", re.compile(rf"\blet(?:s|ting)? go\b[^.]{{0,25}}{_NUM}", re.IGNORECASE)),
    ("job_cuts", re.compile(r"\bjob (?:cuts?|losses|reductions?)\b|\bjobs? at risk\b", re.IGNORECASE)),
    (
        "cuts_people",
        re.compile(
            rf"\b(?:cuts?|cutting|sheds?|shedding|slash\w*|axe[sd]?|sacks?|sacked|eliminat\w*|"
            rf"fires?|firing|fired|retrench\w*|lose|lost)\b[^.]{{0,30}}{_NUM}[^.]{{0,15}}{_PEOPLE}",
            re.IGNORECASE,
        ),
    ),
    (
        "workforce_change",
        re.compile(
            r"\b(?:cut|cuts|cutting|reduc\w*|slash\w*|trim\w*|shed\w*|downsiz\w*|shrink\w*|"
            r"expand\w*|grow\w*|add\w*|restructur\w*)\b[^.]{0,40}\bworkforce\b"
            r"|\bworkforce\b[^.]{0,30}\b(?:cuts?|reductions?|reduc\w*|restructur\w*|expansion)\b",
            re.IGNORECASE,
        ),
    ),
    # v2: voluntary exit / buyout / separation programmes (workforce reduction without "layoff")
    (
        "voluntary_exit",
        re.compile(
            r"\bvoluntary\b[^.]{0,20}\b(?:exit|buy-?outs?|separation|severance|retirement|redundanc\w*|resignation)s?\b"
            r"|\b(?:offers?|offering|extends?|extended)\b[^.]{0,15}\bbuy-?outs?\b",
            re.IGNORECASE,
        ),
    ),
    # v2: "outsourcing deal affects 850 roles", "puts 300 jobs at risk"
    (
        "affects_roles",
        re.compile(
            rf"\b(?:affects?|affecting|impacts?|impacting|puts?|putting|threatens?)\b[^.]{{0,20}}{_NUM}[^.]{{0,10}}\b{_JOBS}\b",
            re.IGNORECASE,
        ),
    ),
    # v2: "850 roles impacted", "1,200 jobs at risk"
    (
        "roles_impacted",
        re.compile(
            rf"{_NUM}\s*(?:\w+\s+){{0,2}}{_JOBS}\s+(?:(?:are|were|could be|may be|will be|being)\s+)?"
            rf"(?:impacted|affected|at risk|axed|cut|eliminated|redundant|lost)\b",
            re.IGNORECASE,
        ),
    ),
    # v2: offshoring / outsourcing of roles
    (
        "offshoring_roles",
        re.compile(
            r"\b(?:offshor\w*|outsourc\w*)\b[^.]{0,40}\b(?:roles|jobs|positions)\b"
            r"|\b(?:roles|jobs|positions)\b[^.]{0,30}\b(?:offshor\w*|outsourc\w*)\b",
            re.IGNORECASE,
        ),
    ),
    ("headcount", re.compile(r"\bheadcount\b", re.IGNORECASE)),
    ("hiring", re.compile(r"\bhiring\b|\b(?:to|will|plans? to|set to|aims? to) hire\b|\bhires? \d", re.IGNORECASE)),
    ("add_jobs", re.compile(rf"\b(?:add|adds|adding|create|creates|creating|generate|generates)\b[^.]{{0,25}}\bjobs\b", re.IGNORECASE)),
    (
        "restructuring_staff",
        re.compile(
            r"\brestructur\w*\b[^.]{0,40}\b(?:workforce|jobs|staff|employees|roles)\b"
            r"|\b(?:workforce|staff|employees)\b[^.]{0,30}\brestructur\w*\b",
            re.IGNORECASE,
        ),
    ),
    ("redundancy", re.compile(r"\bredundanc(?:y|ies)\b|\bfurlough\w*\b|\bretrench\w*\b|\bhiring freeze\b|\battrition\b", re.IGNORECASE)),
    ("recruitment_drive", re.compile(r"\b(?:mass|campus) (?:recruitment|hiring|placements?)\b|\brecruitment drive\b", re.IGNORECASE)),
    (
        "new_office",
        re.compile(
            r"\bopen(?:s|ing|ed)? (?:an? |its )?(?:new )?(?:office|centre|center|campus|hub)\b"
            r"|\bnew (?:india )?(?:office|campus|(?:development|r&d|delivery|innovation) (?:centre|center))\b"
            r"|\bGCCs?\b|\bglobal capability cent(?:re|er)s?\b",
            re.IGNORECASE,
        ),
    ),
]


def career_relevance(title: Optional[str], summary: Optional[str] = None) -> Optional[str]:
    """Return a short reason when the title or RSS summary looks like career news, else None.

    HTML tags and entities in the summary are stripped first. The reason is the name
    of the first pattern that matched, prefixed with "summary:" when only the summary
    matched.
    """
    for label, text in (("", title), ("summary:", summary)):
        clean = _clean(text)
        if not clean:
            continue
        for reason, pattern in _RELEVANCE_PATTERNS:
            if pattern.search(clean):
                return f"{label}{reason}"
    return None


def _clean(value: Optional[str]) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()
