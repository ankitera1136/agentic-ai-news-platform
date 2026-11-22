"""Per-feed scraper settings (career_only, max_entries). No network, no LLM, no DB."""
import json
import random
import string
import time
from email.utils import formatdate
from types import SimpleNamespace

import feedparser
import pytest

from backend.scrapers import techcrunch as scraper

# Random-looking, seeded titles keep the headline dedup (Jaccard / SequenceMatcher)
# from treating different fake stories as duplicates of each other.
_RNG = random.Random(1234)


def _unique_title(prefix):
    words = ["".join(_RNG.choice(string.ascii_lowercase) for _ in range(8)) for _ in range(5)]
    return " ".join(words)


def _entry(title, summary="A short neutral summary.", hours_ago=2):
    stamp = time.time() - hours_ago * 3600
    return feedparser.FeedParserDict(
        title=title,
        link=f"https://news.example.com/{abs(hash(title))}",
        summary=summary,
        published=formatdate(stamp, usegmt=True),
        published_parsed=time.gmtime(stamp),
    )


class Harness:
    """Stubs requests.get, feedparser.parse, call_groq, sleep and the DB for one scrape."""

    def __init__(self, monkeypatch, tmp_path, entries_by_url):
        self.entries_by_url = entries_by_url
        self.llm_calls = []
        self.image_requests = []
        self.stored = []
        self.saved_files = {}
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setattr(scraper.time, "sleep", lambda *_: None)

        def fake_get(url, **kwargs):
            if url in entries_by_url:
                return SimpleNamespace(status_code=200, content=url.encode())
            self.image_requests.append(url)
            return SimpleNamespace(status_code=404, content=b"")

        def fake_parse(content):
            url = content.decode() if isinstance(content, bytes) else content
            return SimpleNamespace(entries=list(entries_by_url[url]))

        def fake_call_groq(api_key, payload, timeout=25, preferred_model=None):
            self.llm_calls.append(payload["messages"][0]["content"])
            return "STUB SUMMARY", "stub-model"

        monkeypatch.setattr(scraper.requests, "get", fake_get)
        monkeypatch.setattr(scraper.feedparser, "parse", fake_parse)
        monkeypatch.setattr(scraper, "call_groq", fake_call_groq)
        monkeypatch.setattr(
            scraper, "_save_article_to_db",
            lambda article_id, data, published: self.stored.append((article_id, data)),
        )
        # Keep the dedup title loader off the real database.
        import backend.app.database as database

        def _no_db():
            raise RuntimeError("no database in tests")

        monkeypatch.setattr(database, "SessionLocal", _no_db)

    def run(self, feeds):
        scraper_feeds = scraper.FEEDS
        scraper.FEEDS = feeds
        try:
            scraper.fetch_and_save_all_sources()
        finally:
            scraper.FEEDS = scraper_feeds
        return self.stored

    @property
    def stored_ids(self):
        return [article_id for article_id, _ in self.stored]


def _snapshot(stored):
    return sorted(
        (article_id, data["source"], data["title"], data["llm_summary"], data["link"])
        for article_id, data in stored
    )


# ---------------------------------------------------------------------------
# Regression: the existing FEEDS list behaves exactly as before
# ---------------------------------------------------------------------------

ORIGINAL_FEED_NAMES = [
    "TechCrunch", "Ars Technica", "MIT Tech Review", "Wired", "Inc42", "YourStory",
    "Economic Times Tech", "MediaNama", "The Hindu Tech", "GitHub Blog", "Stack Overflow Blog",
    "freeCodeCamp", "Towards Data Science", "Hacker News",
]
CAREER_FEED_NAMES = ["HRKatha", "Mint", "Entrackr", "Indian Express", "Analytics India Magazine"]


def _original_feeds():
    return [dict(feed) for feed in scraper.FEEDS if feed["name"] in ORIGINAL_FEED_NAMES]


def _regression_inputs():
    original = _original_feeds()
    entries = {}
    for feed in original:
        # 20 entries per feed; only the first 15 may be processed.
        entries[feed["url"]] = [_entry(_unique_title(feed["name"])) for _ in range(20)]
    return original, entries


def test_existing_14_feeds_are_unchanged_and_have_no_new_keys():
    original = _original_feeds()
    assert [f["name"] for f in original] == ORIGINAL_FEED_NAMES  # same order as before
    for feed in original:
        assert set(feed) == {"name", "url"}, feed


def test_five_career_feeds_precede_economic_times_and_are_gated():
    names = [f["name"] for f in scraper.FEEDS]
    assert len(names) == 19
    et_index = names.index("Economic Times Tech")
    for name in CAREER_FEED_NAMES:
        feed = scraper.FEEDS[names.index(name)]
        assert names.index(name) < et_index, name
        assert feed["career_only"] is True, name
        assert feed["url"].startswith("https://"), name
    expected_urls = {
        "HRKatha": "https://www.hrkatha.com/feed/",
        "Mint": "https://www.livemint.com/rss/companies",
        "Entrackr": "https://entrackr.com/rss",
        "Indian Express": "https://indianexpress.com/section/technology/feed/",
        "Analytics India Magazine": "https://analyticsindiamag.com/feed/",
    }
    for name, url in expected_urls.items():
        assert scraper.FEEDS[names.index(name)]["url"] == url


def test_feed_names_are_unique():
    names = [f["name"] for f in scraper.FEEDS]
    assert len(names) == len(set(names))


def test_existing_feeds_store_exactly_the_first_15_entries_each(monkeypatch, tmp_path):
    feeds, entries = _regression_inputs()
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run(feeds)

    expected = []
    for feed in feeds:
        for entry in entries[feed["url"]][:15]:
            date_str = time.strftime("%Y-%m-%d", entry.published_parsed)
            article_id = f"{date_str}-{scraper.slugify(feed['name'])}-{scraper.slugify(entry.title)[:45]}"
            expected.append((article_id, feed["name"], entry.title, "STUB SUMMARY", entry.link))

    assert _snapshot(stored) == sorted(expected)
    assert len(stored) == 14 * 15
    assert len(harness.llm_calls) == 14 * 15  # one summary call per stored article
    # JSON files are written in the same format (keys unchanged).
    sample_id, sample = stored[0]
    saved = json.loads((tmp_path / "data" / "summaries" / f"{sample_id}.json").read_text(encoding="utf-8"))
    assert set(saved) == {"title", "link", "summary", "published", "image_url", "llm_summary", "source"}
    assert saved["llm_summary"] == "STUB SUMMARY"


def test_feed_dict_with_only_name_and_url_matches_explicit_defaults(monkeypatch, tmp_path):
    entries = {"https://feed.example/a": [_entry(_unique_title("A")) for _ in range(20)]}
    plain = [{"name": "Plain", "url": "https://feed.example/a"}]
    explicit = [{"name": "Plain", "url": "https://feed.example/a", "career_only": False, "max_entries": 15}]

    first = Harness(monkeypatch, tmp_path / "one", entries) if (tmp_path / "one").mkdir() is None else None
    stored_plain = first.run(plain)
    second = Harness(monkeypatch, tmp_path / "two", entries) if (tmp_path / "two").mkdir() is None else None
    stored_explicit = second.run(explicit)

    assert _snapshot(stored_plain) == _snapshot(stored_explicit)
    assert len(stored_plain) == 15


# ---------------------------------------------------------------------------
# career_only
# ---------------------------------------------------------------------------

def test_career_only_drops_non_career_before_llm_and_image_request(monkeypatch, tmp_path):
    url = "https://feed.example/career"
    entries = {url: [
        _entry("The OpenAI board member who fired Sam Altman says we still can't trust AI CEOs"),
        _entry("A family tree of software job titles"),
        _entry("New phone launches with a bigger battery"),
    ]}
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run([{"name": "Gated", "url": url, "career_only": True}])
    assert stored == []
    assert harness.llm_calls == []
    assert harness.image_requests == []


def test_career_only_keeps_a_real_layoff_title(monkeypatch, tmp_path):
    url = "https://feed.example/career"
    entries = {url: [
        _entry("Qualcomm layoffs raise questions over severance support for affected employees in India"),
        _entry("New phone launches with a bigger battery"),
        _entry("Quarterly update", summary="<p>The company plans to <b>lay off</b> 300 people.</p>"),
    ]}
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run([{"name": "Gated", "url": url, "career_only": True}])
    titles = [data["title"] for _, data in stored]
    assert titles == [
        "Qualcomm layoffs raise questions over severance support for affected employees in India",
        "Quarterly update",
    ]
    assert len(harness.llm_calls) == 2  # only kept entries reach the LLM
    assert len(harness.image_requests) == 2  # og:image only for kept entries


def test_career_only_runs_after_skip_keywords(monkeypatch, tmp_path):
    url = "https://feed.example/career"
    entries = {url: [_entry("Best deals: company lays off 500 staff and holds a sale")]}
    harness = Harness(monkeypatch, tmp_path, entries)
    assert harness.run([{"name": "Gated", "url": url, "career_only": True}]) == []


def test_career_only_runs_before_dedup(monkeypatch, tmp_path):
    """A gated-out entry must not be compared with, or added to, the dedup title list."""
    url = "https://feed.example/career"
    seen = []
    real = scraper.is_duplicate_headline

    def spy(new_title, existing_titles, *args, **kwargs):
        seen.append(new_title)
        return real(new_title, existing_titles, *args, **kwargs)

    monkeypatch.setattr(scraper, "is_duplicate_headline", spy)
    entries = {url: [_entry("New phone launches with a bigger battery"), _entry("Acme lays off 300 staff")]}
    harness = Harness(monkeypatch, tmp_path, entries)
    harness.run([{"name": "Gated", "url": url, "career_only": True}])
    assert seen == ["Acme lays off 300 staff"]


def test_career_only_false_stores_non_career_entries(monkeypatch, tmp_path):
    url = "https://feed.example/open"
    entries = {url: [_entry("New phone launches with a bigger battery")]}
    harness = Harness(monkeypatch, tmp_path, entries)
    assert len(harness.run([{"name": "Open", "url": url}])) == 1


def test_feed_log_line_reports_fetched_skipped_stored(monkeypatch, tmp_path, capsys):
    url = "https://feed.example/career"
    entries = {url: [_entry("New phone launches with a bigger battery"), _entry("Acme lays off 300 staff")]}
    harness = Harness(monkeypatch, tmp_path, entries)
    harness.run([{"name": "Gated", "url": url, "career_only": True}])
    assert "[FEED] Gated: fetched=2 skipped_by_career_gate=1 stored=1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# max_entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 15), ("missing", 15), (3, 3), ("7", 7), (1, 1), (100, 100),
        (0, 1), (-5, 1), (500, 100), (2.9, 2),
        ("abc", 15), ("", 15), (None, 15), ([], 15), (True, 15), (False, 15),
    ],
)
def test_feed_max_entries_values(value, expected):
    feed = {"name": "x", "url": "u"} if value == "missing" else {"name": "x", "url": "u", "max_entries": value}
    assert scraper.feed_max_entries(feed) == expected


def test_max_entries_limits_processing(monkeypatch, tmp_path):
    url = "https://feed.example/small"
    entries = {url: [_entry(_unique_title("S")) for _ in range(10)]}
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run([{"name": "Small", "url": url, "max_entries": 4}])
    assert len(stored) == 4
    assert len(harness.llm_calls) == 4


def test_max_entries_above_15_is_honoured_and_clamped_at_100(monkeypatch, tmp_path):
    url = "https://feed.example/big"
    entries = {url: [_entry(_unique_title("B")) for _ in range(120)]}
    harness = Harness(monkeypatch, tmp_path, entries)
    assert len(harness.run([{"name": "Big", "url": url, "max_entries": 40}])) == 40
    harness = Harness(monkeypatch, tmp_path / "x", entries) if (tmp_path / "x").mkdir() is None else None
    assert len(harness.run([{"name": "Big", "url": url, "max_entries": 1000}])) == 100


def test_invalid_max_entries_falls_back_to_15(monkeypatch, tmp_path):
    url = "https://feed.example/bad"
    entries = {url: [_entry(_unique_title("D")) for _ in range(20)]}
    harness = Harness(monkeypatch, tmp_path, entries)
    assert len(harness.run([{"name": "Bad", "url": url, "max_entries": "lots"}])) == 15


# ---------------------------------------------------------------------------
# max_summary_chars
# ---------------------------------------------------------------------------

LONG_HTML_EXCERPT = (
    "<p>" + ("Company news about staffing and plans continues here. " * 8)
    + "Later in the text the company will <b>lay off</b> 300 people. [...]</p>\n"
    '<p>The post <a href="https://x.example/p">Neutral headline</a> '
    'appeared first on <a href="https://x.example">Some Site</a>.</p>'
)


def test_only_hrkatha_has_max_summary_chars_and_it_is_300():
    for feed in scraper.FEEDS:
        if feed["name"] == "HRKatha":
            assert feed["max_summary_chars"] == 300 and feed["career_only"] is True
        else:
            assert "max_summary_chars" not in feed, feed["name"]


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), (300, 300), ("120", 120), (0, None), (-4, None), ("abc", None), (True, None), (2.9, 2)],
)
def test_feed_max_summary_chars_values(value, expected):
    feed = {"name": "x", "url": "u"} if value is None else {"name": "x", "url": "u", "max_summary_chars": value}
    assert scraper.feed_max_summary_chars(feed) == expected


def test_other_feeds_store_the_raw_summary_unchanged(monkeypatch, tmp_path):
    url = "https://feed.example/raw"
    entries = {url: [_entry("Neutral headline one", summary=LONG_HTML_EXCERPT)]}
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run([{"name": "Raw", "url": url}])
    assert stored[0][1]["summary"] == LONG_HTML_EXCERPT


def test_max_summary_chars_stores_plain_text_without_footer_and_within_limit(monkeypatch, tmp_path):
    url = "https://feed.example/limited"
    entries = {url: [_entry("Neutral headline two", summary=LONG_HTML_EXCERPT)]}
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run([{"name": "Limited", "url": url, "max_summary_chars": 300}])
    summary = stored[0][1]["summary"]
    assert 0 < len(summary) <= 300
    assert "<" not in summary and ">" not in summary
    assert "appeared first on" not in summary and "The post" not in summary
    assert "[...]" not in summary
    assert not summary.endswith(" ")
    saved = json.loads((tmp_path / "data" / "summaries" / f"{stored[0][0]}.json").read_text(encoding="utf-8"))
    assert saved["summary"] == summary  # the JSON file matches what is stored


def test_gate_and_llm_prompt_use_the_full_excerpt_before_truncation(monkeypatch, tmp_path):
    url = "https://feed.example/limited"
    entries = {url: [_entry("Neutral headline three", summary=LONG_HTML_EXCERPT)]}
    harness = Harness(monkeypatch, tmp_path, entries)
    stored = harness.run([{"name": "Limited", "url": url, "career_only": True, "max_summary_chars": 300}])
    # "lay off 300 people" only appears after char 300, so the stored text cannot contain it,
    # yet the career gate (full excerpt) kept the entry...
    assert len(stored) == 1
    assert "lay off" not in stored[0][1]["summary"]
    # ...and the LLM summary prompt received the full, untruncated excerpt.
    assert len(harness.llm_calls) == 1
    assert LONG_HTML_EXCERPT in harness.llm_calls[0]


def test_only_indian_express_has_max_entries_50():
    for feed in scraper.FEEDS:
        if feed["name"] == "Indian Express":
            assert feed["max_entries"] == 50 and feed["career_only"] is True
            assert scraper.feed_max_entries(feed) == 50
        else:
            assert "max_entries" not in feed, feed["name"]
