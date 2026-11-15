"""Full-article-body fetching and caching for CareerRadar.

Bodies are internal only: they are used to build extraction prompts and to audit
grounded evidence. Never return body_text through an API or UI.
"""

import html
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from backend.app.models import Article, ArticleBody

FETCH_TIMEOUT_SECONDS = 8.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # ~2 MB
MAX_BODY_CHARS = 12_000
MIN_BODY_CHARS = 400
MAX_LLM_CHARS = 6_000
MAX_FETCH_ATTEMPTS = 3  # initial attempt + at most 2 retries

USER_AGENT = (
    "CareerRadarBot/1.0 (internal research extraction; "
    "contact: admin@technewsdigest.local)"
)

_PAYWALL_MARKERS = re.compile(
    r"subscribe to (?:continue|read)|already a subscriber|sign in to continue|"
    r"this article is for subscribers|to continue reading|"
    r"create a (?:free )?account to continue|paywall",
    re.IGNORECASE,
)

_LAST_FETCH_AT = 0.0


@dataclass
class FetchResult:
    status: str  # ok | failed | paywalled | too_short
    body_text: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BodyContext:
    text: str
    text_source: str  # full_body | stored_summary | none
    body_length: int = 0
    cached: bool = False
    status: Optional[str] = None
    body_available: bool = False
    fetch_attempted: bool = False


def is_vercel_runtime() -> bool:
    """Reuse the same detection the app already uses."""
    return bool(os.getenv("VERCEL") or os.getenv("NOW_REGION"))


def _clean_html_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_stored_text(article: Article) -> str:
    """Title + RSS summary + LLM summary, used only when no body is available."""
    parts = [
        _clean_html_text(article.title),
        _clean_html_text(article.summary),
        _clean_html_text(article.llm_summary),
    ]
    return "\n\n".join(part for part in parts if part)


def text_for_llm(text: str) -> str:
    """The exact text sent to the LLM; grounding uses this same slice."""
    return (text or "")[:MAX_LLM_CHARS]


def _respect_fetch_delay(delay_seconds: float) -> None:
    global _LAST_FETCH_AT
    if delay_seconds <= 0:
        return
    elapsed = time.monotonic() - _LAST_FETCH_AT
    if elapsed < delay_seconds:
        time.sleep(delay_seconds - elapsed)
    _LAST_FETCH_AT = time.monotonic()


def _is_fetchable_url(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def fetch_article_body(
    url: str,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_chars: int = MAX_BODY_CHARS,
    min_chars: int = MIN_BODY_CHARS,
    session: Optional[requests.Session] = None,
) -> FetchResult:
    """Fetch one article page and extract <p> text only."""
    if not _is_fetchable_url(url):
        return FetchResult("failed", error="invalid_url")

    client = session or requests
    try:
        response = client.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        return FetchResult("failed", error="timeout")
    except requests.exceptions.ConnectionError:
        return FetchResult("failed", error="connection_error")
    except requests.exceptions.RequestException as exc:
        return FetchResult("failed", error=f"request_error:{type(exc).__name__}")
    except Exception as exc:
        return FetchResult("failed", error=f"unexpected:{type(exc).__name__}")

    http_status = response.status_code
    try:
        if http_status in (401, 402):
            return FetchResult("paywalled", http_status=http_status, error="auth_required")
        if http_status < 200 or http_status >= 300:
            return FetchResult("failed", http_status=http_status, error=f"http_{http_status}")

        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not any(t in content_type for t in ("text/html", "application/xhtml")):
            return FetchResult("failed", http_status=http_status, error="not_html")

        declared_length = response.headers.get("Content-Length")
        if declared_length:
            try:
                if int(declared_length) > max_bytes:
                    return FetchResult("failed", http_status=http_status, error="response_too_large")
            except ValueError:
                pass

        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                return FetchResult("failed", http_status=http_status, error="response_too_large")
            chunks.append(chunk)
        raw = b"".join(chunks)
    except requests.exceptions.Timeout:
        return FetchResult("failed", http_status=http_status, error="read_timeout")
    except Exception as exc:
        return FetchResult("failed", http_status=http_status, error=f"read_error:{type(exc).__name__}")
    finally:
        try:
            response.close()
        except Exception:
            pass

    encoding = response.encoding or "utf-8"
    try:
        page = raw.decode(encoding, errors="replace")
    except (LookupError, TypeError):
        page = raw.decode("utf-8", errors="replace")

    soup = BeautifulSoup(page, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]):
        tag.decompose()

    paragraphs = []
    for paragraph in soup.find_all("p"):
        value = re.sub(r"\s+", " ", paragraph.get_text(" ", strip=True))
        if len(value) >= 30:
            paragraphs.append(value)
    body_text = "\n\n".join(paragraphs).strip()

    if _PAYWALL_MARKERS.search(page) and len(body_text) < min_chars * 2:
        return FetchResult("paywalled", http_status=http_status, error="paywall_marker")

    if len(body_text) < min_chars:
        return FetchResult("too_short", http_status=http_status, body_text=body_text[:max_chars], error="body_too_short")

    return FetchResult("ok", http_status=http_status, body_text=body_text[:max_chars])


def _fallback_context(article: Article, status: Optional[str], cached: bool, body_length: int) -> BodyContext:
    return BodyContext(
        text=build_stored_text(article),
        text_source="stored_summary",
        body_length=body_length,
        cached=cached,
        status=status,
        body_available=False,
    )


def ensure_article_body(
    db: Session,
    article: Article,
    allow_fetch: bool = True,
    delay_seconds: float = 0.5,
    fetch_session: Optional[requests.Session] = None,
) -> BodyContext:
    """Return the best available text, fetching and caching a full body when allowed."""
    cached = db.query(ArticleBody).filter(ArticleBody.article_id == article.id).first()

    if cached and cached.status == "ok" and cached.body_text:
        body_text = cached.body_text
        return BodyContext(
            text=f"{article.title}\n\n{body_text}",
            text_source="full_body",
            body_length=len(body_text),
            cached=True,
            status="ok",
            body_available=True,
        )

    if not allow_fetch:
        # Vercel cron path: never fetch. Only a cached successful body is eligible.
        return BodyContext(
            text="",
            text_source="none",
            body_length=len(cached.body_text or "") if cached else 0,
            cached=bool(cached),
            status=cached.status if cached else None,
            body_available=False,
        )

    if cached and cached.status in ("paywalled", "too_short"):
        return _fallback_context(article, cached.status, True, len(cached.body_text or ""))

    failed_terminal = bool(
        cached and cached.status == "failed" and (cached.attempts or 0) >= MAX_FETCH_ATTEMPTS
    )
    if failed_terminal:
        return _fallback_context(article, "failed", True, len(cached.body_text or ""))

    _respect_fetch_delay(delay_seconds)
    result = fetch_article_body(article.link or "", session=fetch_session)

    previous_attempts = (cached.attempts or 0) if cached else 0
    attempts = previous_attempts + 1
    if cached is None:
        cached = ArticleBody(article_id=article.id)
        db.add(cached)
    cached.body_text = result.body_text
    cached.fetched_at = datetime.utcnow()
    cached.status = result.status
    cached.http_status = result.http_status
    cached.error = result.error
    cached.attempts = attempts
    db.commit()

    if result.status == "ok" and result.body_text:
        return BodyContext(
            text=f"{article.title}\n\n{result.body_text}",
            text_source="full_body",
            body_length=len(result.body_text),
            cached=False,
            status="ok",
            body_available=True,
            fetch_attempted=True,
        )

    return _fallback_context(article, result.status, False, len(result.body_text or ""))
