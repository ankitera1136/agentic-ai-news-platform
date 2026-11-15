import requests

from backend.app.models import ArticleBody
from backend.services.career import body_fetcher
from backend.services.career.body_fetcher import (
    MAX_BODY_CHARS,
    ensure_article_body,
    fetch_article_body,
)


class FakeResponse:
    def __init__(self, status_code=200, text="", headers=None, encoding="utf-8", chunks=None):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = encoding
        self._chunks = chunks if chunks is not None else [text.encode("utf-8")]
        self.closed = False

    def iter_content(self, chunk_size=65536):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.response


def _long_html(prefix="Body"):
    paragraphs = "".join(
        f"<p>{prefix} paragraph {i}: " + ("robotics hiring engineers in Bengaluru " * 4) + "</p>"
        for i in range(8)
    )
    return (
        "<html><body>"
        "<nav>NAVTEXT</nav><script>SCRIPTTEXT</script><header>HEADERTEXT</header>"
        f"{paragraphs}"
        "<footer>FOOTERTEXT</footer></body></html>"
    )


def test_fetch_success_extracts_only_paragraphs():
    session = FakeSession(FakeResponse(text=_long_html()))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "ok"
    assert result.http_status == 200
    assert "Body paragraph 0" in result.body_text
    for unwanted in ("NAVTEXT", "SCRIPTTEXT", "HEADERTEXT", "FOOTERTEXT"):
        assert unwanted not in result.body_text


def test_fetch_invalid_url_does_not_call_http():
    session = FakeSession(FakeResponse(text=_long_html()))
    result = fetch_article_body("ftp://example.com/a", session=session)

    assert result.status == "failed"
    assert result.error == "invalid_url"
    assert session.calls == 0


def test_fetch_timeout_is_failed():
    session = FakeSession(exc=requests.exceptions.Timeout())
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "failed"
    assert result.error == "timeout"


def test_fetch_paywall_status():
    session = FakeSession(FakeResponse(status_code=402, text=""))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "paywalled"
    assert result.http_status == 402


def test_fetch_paywall_marker():
    html = "<html><body><p>Short intro.</p><p>Subscribe to continue reading.</p></body></html>"
    session = FakeSession(FakeResponse(text=html))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "paywalled"
    assert result.error == "paywall_marker"


def test_fetch_too_short():
    html = "<html><body><p>Only a short teaser paragraph is available here.</p></body></html>"
    session = FakeSession(FakeResponse(text=html))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "too_short"
    assert result.error == "body_too_short"


def test_fetch_rejects_non_html():
    session = FakeSession(FakeResponse(headers={"Content-Type": "application/json"}, text="{}"))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "failed"
    assert result.error == "not_html"


def test_fetch_rejects_declared_oversize_response():
    headers = {"Content-Type": "text/html", "Content-Length": str(3 * 1024 * 1024)}
    session = FakeSession(FakeResponse(headers=headers, text=""))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "failed"
    assert result.error == "response_too_large"


def test_fetch_rejects_streamed_oversize_response():
    session = FakeSession(FakeResponse(chunks=[b"a" * (1024 * 1024), b"b" * (1024 * 1024) + b"c"]))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "failed"
    assert result.error == "response_too_large"


def test_fetch_truncates_stored_body():
    huge = "".join(f"<p>{'x' * 5000} paragraph {i}</p>" for i in range(5))
    session = FakeSession(FakeResponse(text=f"<html><body>{huge}</body></html>"))
    result = fetch_article_body("https://example.com/a", session=session)

    assert result.status == "ok"
    assert len(result.body_text) == MAX_BODY_CHARS


def test_cached_body_is_reused_without_http(career_db, make_article, monkeypatch):
    article = make_article()
    calls = {"count": 0}

    def fake_fetch(url, **kwargs):
        calls["count"] += 1
        return body_fetcher.FetchResult(
            "ok", body_text="Full body with plenty of detail. " * 30, http_status=200
        )

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)

    first = ensure_article_body(career_db, article, allow_fetch=True, delay_seconds=0)
    second = ensure_article_body(career_db, article, allow_fetch=True, delay_seconds=0)

    assert first.text_source == "full_body"
    assert second.text_source == "full_body"
    assert second.cached is True
    assert calls["count"] == 1


def test_failed_fetch_is_retried_at_most_twice(career_db, make_article, monkeypatch):
    article = make_article()
    calls = {"count": 0}

    def fake_fetch(url, **kwargs):
        calls["count"] += 1
        return body_fetcher.FetchResult("failed", error="timeout", http_status=None)

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)

    for _ in range(4):
        context = ensure_article_body(career_db, article, allow_fetch=True, delay_seconds=0)
        assert context.text_source == "stored_summary"

    row = career_db.query(ArticleBody).filter_by(article_id=article.id).one()
    assert calls["count"] == 3
    assert row.status == "failed"
    assert row.attempts == 3


def test_paywalled_body_is_terminal_and_not_retried(career_db, make_article, monkeypatch):
    article = make_article()
    calls = {"count": 0}

    def fake_fetch(url, **kwargs):
        calls["count"] += 1
        return body_fetcher.FetchResult("paywalled", http_status=402, error="auth_required")

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)

    first = ensure_article_body(career_db, article, allow_fetch=True, delay_seconds=0)
    second = ensure_article_body(career_db, article, allow_fetch=True, delay_seconds=0)

    assert first.text_source == "stored_summary"
    assert second.text_source == "stored_summary"
    assert calls["count"] == 1


def test_cache_only_mode_skips_paywalled_cached_body(career_db, make_article, monkeypatch):
    article = make_article()
    career_db.add(
        ArticleBody(
            article_id=article.id,
            body_text=None,
            status="paywalled",
            attempts=1,
            http_status=402,
        )
    )
    career_db.commit()
    calls = {"count": 0}

    def fake_fetch(url, **kwargs):
        calls["count"] += 1
        return body_fetcher.FetchResult("ok", body_text="body", http_status=200)

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)

    context = ensure_article_body(career_db, article, allow_fetch=False)

    assert context.text_source == "none"
    assert calls["count"] == 0


def test_cache_only_mode_never_fetches(career_db, make_article, monkeypatch):
    article = make_article()
    calls = {"count": 0}

    def fake_fetch(url, **kwargs):
        calls["count"] += 1
        return body_fetcher.FetchResult("ok", body_text="body", http_status=200)

    monkeypatch.setattr(body_fetcher, "fetch_article_body", fake_fetch)

    context = ensure_article_body(career_db, article, allow_fetch=False)

    assert context.text_source == "none"
    assert context.body_available is False
    assert calls["count"] == 0
