# DATA_PLAN: real hiring/layoff data and LLM capacity for CareerRadar

Date of checks: 2026-11-15 (feeds fetched about 17:00 to 18:30 UTC). Planning only. Nothing in the repo, `.env` or the DB was changed. The only file written is this one.

How I checked: I read PROJECT_STATE.md, then the current code (`backend/scrapers/techcrunch.py`, `backend/app/groq_client.py`, `backend/app/webapp.py`, `backend/services/career/*`, `backend/services/brief/*`, `dedup/*`, `scripts/backfill_career_signals.py`). I queried `tech_news.db` read-only. I fetched candidate feeds with throwaway scripts kept in `/tmp` (not in the repo). Each script fetched the feed, parsed it with `feedparser`, matched titles against the hiring/layoff keywords, fetched 3 article pages with the repo's own `fetch_article_body()` (the same code path CareerRadar uses), and read `robots.txt`. I did not call the Groq API (that needs your key), so live Groq headers for your account are **unverified**. For `.env` I only listed variable names with values masked, plus three non-secret CareerRadar values.

## 0. Findings in short

1. **The data is missing because of the sources.** In the last 14 days the DB has 236 articles from 14 sources. Zero of them are real hiring or layoff news. The 3 title matches for your keyword list are false positives ("fired Sam Altman", "recruited AI exec to spy", "technical recruiter" Q&A). The big India story of the week (Qualcomm India layoffs, covered by HRKatha, Mint, News18, ET, HT, TOI) is not in the DB at all.
2. **Only a few feeds are both relevant and readable.** I tested 72 feed URLs. Most failed one of: stale, not a feed, paywalled body, body returns 403 to our bot, robots.txt forbids. The usable set is small: HRKatha (best), Mint Companies, Entrackr, Indian Express Tech, Analytics India Magazine, all with a keyword gate. Expect roughly **7 to 9 relevant articles per week** (about 5 to 7 distinct events after de-duplication). This is real but thin. It is a lower bound because I only counted keyword-matching titles.
3. **The LLM bottleneck is not the new feeds. It is extraction on every article and the 8K tokens-per-minute cap.** One extraction call is about 3.3K tokens. The free-tier table in Groq's docs shows 8K tokens per minute per model, so two calls a minute is the ceiling. CareerRadar's default spacing is 1 second. Your 41 stored extractions already show the spill: 26 of 41 were served by a fallback model (15 gpt-oss-120b, 15 qwen, 11 gpt-oss-20b).
4. **A keyword pre-filter removes about 98% of extraction calls on today's data** (6 of 308 articles pass), and a `career_only` gate on the new feeds keeps them to about 4 extra articles a day instead of about 54.
5. **A single Groq key is enough for the filtered plan.** A second key from another account is not acceptable: Groq's Acceptable Use Policy names that exact pattern (Part C, C9).
6. **Side findings you will want to know about:** 22 of 308 stored `llm_summary` values (7%) are leaked model reasoning ("The user wants a summary of..."). `ARTICLE_RETENTION_DAYS` defaults to 30, which silently deletes career signals after 30 days (they cascade with the article) while the API accepts `days` up to 90 and the trend defaults to 8 weeks. The scraper's cross-publisher dedup keeps the first outlet seen, so a paywalled copy can hide a readable one.

---

## Part A. The current pull

### A1. How the scraper works (`backend/scrapers/techcrunch.py`)

| Question | Answer (from code) |
|---|---|
| Feed entry structure | A plain dict in the list `FEEDS`: `{'name': 'Inc42', 'url': 'https://inc42.com/feed/'}`. Only `name` and `url` are read. There are **no per-feed settings**. |
| Feed count | 14 (TechCrunch, Ars Technica, MIT Tech Review, Wired, Inc42, YourStory, Economic Times Tech, MediaNama, The Hindu Tech, GitHub Blog, Stack Overflow Blog, freeCodeCamp, Towards Data Science, Hacker News). |
| Entries per feed per run | `feed.entries[:15]`, hard-coded. Feed fetch timeout 10 s, one HTTP GET per feed. |
| Other gates (global, in order) | Skip if older than 60 days (`published_parsed`). Skip if the JSON file for `date-source-title` already exists. Skip if the title contains a word from `SKIP_KEYWORDS` (review, deals, sale, sports, etc.). Then the dedup check. |
| Dedup | `is_duplicate_headline()` against titles of all DB articles with `created_at` in the last 72 h plus the 150 newest JSON files. Duplicate if (token Jaccard >= 0.48 and >= 3 shared tokens) or `SequenceMatcher` ratio >= 0.62. A duplicate is **dropped**, not clustered. First outlet seen wins, in the order of `FEEDS`. |
| LLM summary | One `call_groq` per new article that survives the gates (`get_llm_summary_groq`, `max_tokens=120`, `temperature=0.7`, timeout 12 s), then `time.sleep(0.5)`. On failure it falls back to the cleaned RSS summary (first 200 chars). It reads `GROQ_API_KEY` only. Each new article without an image in the feed also costs one extra HTTP GET (og:image, 2 s timeout). |
| Storage | Writes `data/summaries/{date}-{slug(source)}-{slug(title)[:45]}.json`, then inserts the row (`Article.id` = that filename base) with keyword categories. `created_at` is the parsed **publication date** (RFC 822 only; if parsing fails it becomes "now"). Timezone offsets are dropped, not converted. |
| `source` name | `feed_info['name']`, copied to the JSON `source` and to `Article.source`. If missing, the model default is `'TechNews'`. |
| Run frequency | `_run_scrape_sync()` runs from APScheduler **hourly** (`interval hours=1`) and once at every server startup (`run_startup_background_tasks`). Pipeline order: scrape, sync files to DB, dedup/cluster, CareerRadar (only if `CAREER_RADAR_ENABLED`), retention cleanup. |
| Vercel / Render cron | The scheduler is disabled on Vercel. `GET /api/cron/scrape` is public (no auth). `vercel.json` has **no `crons` entry** and proxies `/api/*` to the Render service, so I cannot tell what calls it: **unverified**. |
| Retention | `ARTICLE_RETENTION_DAYS`, default **30** (not set in your `.env`). `cleanup_old_articles(days, min_keep=50)` deletes by `created_at` (publication date), oldest first, never below 50 rows. `career_signals`, `career_extractions` and `article_bodies` reference `articles.id` with `ON DELETE CASCADE` (I read the SQLite DDL), so **signals die with their article**. |
| Mismatch | The freshness gate is 60 days but retention is 30, so an item aged 31 to 60 days is inserted, summarised (one LLM call) and deleted by cleanup in the same run. |

### A2. Groq usage per cycle and per day

Everything goes through `call_groq()` except two paths: the dedup verifier (Groq SDK, `llama-3.1-8b-instant`, `max_tokens=5`) and admin "re-summarize" (a direct HTTP call to `gpt-oss-120b`).

How `call_groq` behaves: it builds a model list (`preferred_model`, `GROQ_MODEL`, live `/models` list sorted by a priority list that starts with `openai/gpt-oss-120b`, then the static chain). For each model it tries up to 2 times. On HTTP 429 it sleeps 1 s and retries, then moves to the next model. One logical call can therefore turn into many HTTP requests when rate limited. It does not read `retry-after` or the `x-ratelimit-*` headers.

| Workload | Calls per trigger | Calls per day (estimate) | Typical tokens per call | Model bucket |
|---|---|---|---|---|
| Scraper summary | 1 per new article | **about 60** (55, 63 and 60 new articles on 13, 14 and 15 Nov; steady state **unverified**, only 3 days observed) | about 0.4K (RSS summaries average 74 to 463 chars) | gpt-oss-120b first |
| Dedup verifier (L3) | 1 per borderline pair, cached in `pair_cache` | **at most about 8** (DB history holds 23 L3 events and 22 cached verdicts) | about 0.3K | `llama-3.1-8b-instant` (separate bucket; its limits are not in the docs table I fetched: **unverified**) |
| CareerRadar extraction | 1 per eligible article, +1 if the JSON is invalid | cap = 5 per run x 24 runs = **120**. Demand today is about 60 x 0.77 (full body) x 0.85 (not roundup) = **about 40**, plus a backlog of 255 never-processed articles (about 150 to 200 LLM calls) | **about 3.3K** (system prompt 4.5K chars + body up to 6K chars is about 2.7K in, up to 1K out) | gpt-oss-120b first, or `CAREER_LLM_MODEL` |
| Chat (`POST /chat`) | 1 per message (10 per minute per IP) | unknown | about 2.5K (6 articles, 8 history turns, up to 900 out) | `AI_SETTINGS` model, default gpt-oss-120b |
| Fact-check (`/cross-check`) | 1 per cluster member (target + cluster, or up to 4 related) | per click; cached 1 h per process | about 0.6K each | gpt-oss-120b first |
| Weekly brief | 1 writer call, +1 retry, per user per ISO week; regenerate limited to 5 per hour per user | 2 per brief | about 2 to 3K | `CAREER_LLM_MODEL` or default |
| "Today in tech history" | 1 per date, cached in `data/history` | at most 1 per instance per day (Render's filesystem is ephemeral, so a restart can repeat it: **unverified**) | about 1K | default |
| Admin re-summarize | 1 per click, direct HTTP, no fallback | rare | about 0.5K | gpt-oss-120b |

Totals at the hourly schedule:
- **Now (CareerRadar off in your `.env`):** about 70 calls per day plus interactive use.
- **CareerRadar on, `CAREER_EXTRACTION_BATCH_LIMIT=5`, no pre-filter:** about 110 typical and up to 190 per day plus interactive use. At the 120-per-day cap, extraction alone is about 400K tokens a day.

Limits from Groq's docs table ([rate-limits page](https://console.groq.com/docs/rate-limits), fetched today). Limits apply per organization and per model. The page calls these the "base" limits and the wording about which plan they belong to is ambiguous, so treat the free-tier mapping as **unverified**:

| Model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| openai/gpt-oss-120b | 30 | 1K | 8K | 200K |
| openai/gpt-oss-20b | 30 | 1K | 8K | 200K |
| qwen/qwen3.8-27b | 30 | 1K | 8K | 200K |

What this means for you:
- **Tokens per minute is the binding limit.** At 3.3K tokens per extraction call, the 120b model allows about 2 calls per minute. `CAREER_LLM_CALL_DELAY_SECONDS` defaults to 1.0, so a batch of 5 sends about 16K tokens in 10 to 20 seconds and 429s on the first model. It only "works" because `call_groq` spills to the next model. Your ledger shows it: `model_used` is 15 x gpt-oss-120b, 15 x qwen3.8-27b, 11 x gpt-oss-20b, 12 x none (no LLM call).
- **Summaries burst too.** The loop makes about 40 calls per minute (0.5 s sleep plus latency). That exceeds 30 RPM and 8K TPM when more than about 20 new articles arrive in one run (server restart, new feed, first run).
- **All features share the same first-choice bucket** (gpt-oss-120b, one organization): scraper summaries, extraction, chat, fact-check, brief, admin re-summarize. A 5-call extraction batch uses up the minute for chat. Chat then falls to the next model after 2 s of sleeping per rejected model.
- **Summary quality bug (verified in the DB):** 22 of 308 `llm_summary` values begin with "The user wants..." or "We need to produce...". `max_tokens=120` on a reasoning model is consumed by reasoning, `content` comes back empty, and `call_groq` returns the `reasoning` field instead. Groq's [reasoning docs](https://console.groq.com/docs/reasoning) list `include_reasoning=false` and `reasoning_effort` (low, medium, high) for the gpt-oss models.

### A3. DB check (read-only), last 14 days

The window covers the whole DB (oldest row 2026-10-23). 236 of 308 articles are from the last 14 days, and 178 of those arrived on 13 to 15 Nov, because scraping only recently started running regularly. "Kw titles (literal)" uses your list as substrings: hiring, layoff, laid off, jobs, recruit, headcount, workforce, hires, fired, job cuts, restructuring. "Kw titles (regex)" is a word-boundary regex that adds singular forms (job, hire, lays off and so on).

| Source | In feeds today | Articles (14 d) | Kw titles (literal) | Kw titles (regex) | Layoff titles | Real hiring/layoff news |
|---|---|---|---|---|---|---|
| Hacker News | yes | 30 | 0 | 1 | 0 | 0 |
| Wired | yes | 27 | 0 | 0 | 0 | 0 |
| TechCrunch | yes | 25 | 0 | 0 | 0 | 0 |
| Ars Technica | yes | 24 | 0 | 0 | 0 | 0 |
| YourStory | yes | 23 | 0 | 0 | 0 | 0 |
| The Verge | **no** (not in `FEEDS` now, but JSON files for it exist up to 14 Nov; why it left the list is **unverified**) | 18 | 2 | 2 | 0 | 0 |
| Economic Times Tech | yes | 15 | 0 | 1 | 0 | 0 |
| The Hindu Tech | yes | 14 | 0 | 0 | 0 | 0 |
| Inc42 | yes | 13 | 0 | 0 | 0 | 0 |
| MediaNama | yes | 10 | 0 | 0 | 0 | 0 |
| MIT Tech Review | yes | 10 | 0 | 0 | 0 | 0 |
| Stack Overflow Blog | yes | 9 | 1 | 1 | 0 | 0 |
| freeCodeCamp | yes | 9 | 0 | 0 | 0 | 0 |
| GitHub Blog | yes | 8 | 0 | 0 | 0 | 0 |
| The Ken | **no** | 1 | 0 | 0 | 0 | 0 |
| **Total** | | **236** | **3** | **5** | **0** | **0** |

The matches are all noise: "...who fired Sam Altman...", "China reportedly recruited AI exec...", "How do you hire a human? Q&A with ... technical recruiter", "Nvidia ... take over Jensen Huang's job", "A family tree of software job titles".

Other DB facts:
- `career_signals` has **4** rows, not 3 as in PROJECT_STATE: Bitdeer NEW_OFFICE (Hacker News), Ember FUNDING and EXPANSION (Inc42), and a Google HIRING row (id 4, quote "Google will hire engineers.", 0.9 confidence) attached to an Ars Technica article about crows. Row 4 looks like your inserted test row; I could not confirm that (**unverified**).
- `career_extractions`: 41 full_body, 7 stored_summary (headline only), 5 skipped_roundup. 255 non-copy articles have no ledger row.
- Weekly funnel per source works with plain SQL (tested). It is the basis for the monitoring step in Part D:
  ```sql
  SELECT strftime('%Y-W%W', a.created_at) AS week, a.source,
         COUNT(*) AS articles,
         SUM(e.article_id IS NOT NULL) AS extracted,
         SUM(e.text_source='full_body') AS full_body,
         SUM(e.text_source='stored_summary') AS headline_only,
         COALESCE(SUM(e.signals_count),0) AS signals
  FROM articles a LEFT JOIN career_extractions e ON e.article_id=a.id
  WHERE a.is_copy_of IS NULL GROUP BY 1,2 ORDER BY 1 DESC, 3 DESC;
  ```
  (SQLite syntax; PostgreSQL needs `to_char(a.created_at,'IYYY-"W"IW')` and `COUNT(*) FILTER (WHERE ...)`.)
- A candidate pre-filter regex (hiring, hire(s/d), layoff(s), lays off, laid off, let go, job cuts, jobs, recruit*, headcount, workforce, fired, restructuring, retrench*, redundanc*, furlough*, attrition, freshers, employees, "cuts N jobs/roles", "opens new office/centre", GCC) matched **6 of 308** articles on title or RSS summary, and **1 of the 255** unprocessed ones. That is the saving from a pre-filter.

---

## Part B. Candidate sources, verified

### B4/B5. Fifteen shortlisted candidates, fetched today

Columns: **HTTP/valid** is the feed fetch. **Items/wk** is items in the last 14 days divided by 2 when the feed reaches back 14 days or more, otherwise items divided by the feed's date span times 7 (so short windows are rough). **Kw 14d** is titles matching the strict keyword regex among items in the last 14 days. **Summary** is the median RSS summary length in characters. **Bodies** is 3 article pages fetched with `fetch_article_body()` (declared UA `CareerRadarBot/1.0`). **robots** is the result for `CareerRadarBot` on the sampled article URLs (and `*` for the feed).

| # | Candidate | Feed URL | HTTP / valid | Entries | Newest | Items/wk | Kw 14d (relevant events, my reading) | Summary | Bodies ok | robots / terms | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HRKatha (all) | `https://www.hrkatha.com/feed/` | 200 / yes | 50 | 15 Nov | about 38 | 11 (6 company events in 9 days) | 471 | 3/3 | robots allow. **Terms of Service say "No part of the HRKatha content may be reproduced, distributed, or transmitted in any form"** | **Use**, gated. See question 9. |
| 1b | HRKatha search "layoffs" | `https://www.hrkatha.com/?s=layoffs&feed=rss2` | 200 / yes | 50 | 11 Nov | about 5.5 | 9 (7 company events in 14 days) | 445 | 3/3 | same | Optional. It reaches back to 24 Aug, so it is the only way to seed history. |
| 2 | Mint Companies | `https://www.livemint.com/rss/companies` | 200 / yes | 35 | 15 Nov (today) | about 190 | 3 (1 real: Qualcomm layoffs; the others are "CEO's job" and a bio) | 216 | 3/3 | robots allow; terms not read (**unverified**) | **Use**, gated. Feed spans only 1.3 days, so its true weekly rate is **unverified**. |
| 3 | Entrackr | `https://entrackr.com/rss` (`/feed` returns 404) | 200 / yes | 50 | 14 Nov | about 38 | 1 (PhysicsWallah job cuts) | 2292 | 3/3 | robots allow; terms not read | **Use**, gated. |
| 4 | Indian Express Technology | `https://indianexpress.com/section/technology/feed/` | 200 / yes | 200 | 15 Nov | about 42 | 3 (2 real: Canva layoffs, Australian job cuts and offshoring) | **0** (no description in feed) | 3/3 | robots allow; terms not read | **Use**, gated. No feed summary means the stored summary and LLM summary come from the title only. |
| 5 | Analytics India Magazine | `https://analyticsindiamag.com/feed/` | 200 / yes | 50 | 14 Nov | about 70 | 1 (Qantas 850 roles, India angle) | 151 | 3/3 | robots allow; terms not read | **Use**, gated. Its `/tag/layoffs/feed/` ignores the tag and returns the same all-site feed. |
| 6 | TechCrunch tag "layoffs" | `https://techcrunch.com/tag/layoffs/feed/` | 200 / yes | 20 | 4 Nov | about 1 | 2 (Canva, Squarespace) | 134 | 3/3 | robots allow | Optional. Good precision, but 20 items span 6 months. The existing TechCrunch main feed already carries new layoffs posts. |
| 7 | Economic Times IT | `.../tech/information-tech/rssfeeds/78570530.cms` | 200 / yes | 50 | 13 Nov | about 6.5 | 3 (2 hiring-forecast reports on GCCs) | 277 | **0/3 paywalled** | robots allow | **Reject for CareerRadar.** A paywalled body means headline-only, which produces no signal. |
| 8 | Economic Times Company | `.../news/company/rssfeeds/2143429.cms` | 200 / yes | 49 | 14 Nov | about 18 | 0 in 14 days | 416 | 3/3 | robots allow | Hold. Readable, but 0 hits in the sample. |
| 9 | News18 Business | `https://www.news18.com/commonfeeds/v1/eng/rss/business.xml` | 200 / yes | 200 | 15 Nov | about 97 | 6 (5 real: Qualcomm x2, Boeing, Canva, Disney) | 155 | **0/3 (HTTP 403 to our bot UA)** | robots allow | **Reject.** Best headline coverage of Indian layoffs, but we cannot read the body without spoofing a browser UA. I do not recommend that. |
| 10 | Business Standard Companies | `https://www.business-standard.com/rss/companies-101.rss` | 200 / yes | 35 | 15 Nov | about 117 | 0 | 991 | **0/3 (403)** | robots allow | Reject. |
| 11 | BusinessLine Companies | `https://www.thehindubusinessline.com/companies/feeder/default.rss` | 200 / yes | 60 | 15 Nov | about 180 | 1 (not a workforce event) | 122 | 3/3 | robots.txt fetch timed out (**unverified**) | Hold. Only a "restructuring charges" hit. |
| 12 | YourStory tag "layoffs" | `https://yourstory.com/tag/layoffs/feed` | 200 / yes | 20 | **27 Oct** | 0 in 14 days | 0 | 159 | 3/3 | robots allow | **Reject: stale** (19.6 days old). |
| 13 | Inc42 layoffs | `https://inc42.com/tag/layoffs/feed/` (HTML, not a feed); search feed `https://inc42.com/?s=layoffs&feed=rss2` | tag: 200 / **no**; search: 200 / yes | 0; 24 | 12 Nov | about 1.5 | 0 | 118 | 3/3 | robots allow (the search path is disallowed for `*`) | **Reject.** |
| 14 | Hacker News "jobs" (YC job posts via hnrss) | `https://hnrss.org/jobs` | 200 / yes | 20 | 13 Nov | about 4.5 | 9, all job ads ("X (YC F25) is hiring ...") | 142 | 2/3 | robots allow | Optional. Real HIRING signals with skills, but US/YC only and they are ads, not news. See question 10. |
| 15 | Google News RSS search "layoffs India tech" | `https://news.google.com/rss/search?q=layoffs+India+tech+when:7d&hl=en-IN&gl=IN&ceid=IN:en` | 200 / yes | 31 | 15 Nov | about 42 | 30 of 31 | 96 (title only) | **0/3** (interstitial page, 0 chars) | **`news.google.com/robots.txt` is `User-agent: * / Disallow: /`** with only a few allowed paths | **Reject** (see B7). |

### Other candidates tested and rejected

| Candidate | URL (abbreviated) | Result |
|---|---|---|
| ET Startups, ET Tech (current feed) | `.../tech/startups/...78570540.cms`, `.../tech/rssfeeds/13357270.cms` | Fresh, 0 and 1 keyword titles, bodies paywalled 3/3. |
| ET HR-Leadership | `.../small-biz/hr-leadership/rssfeeds/47280670.cms` | Stale (22.5 days). |
| Moneycontrol Tech / Business | `moneycontrol.com/rss/technology.xml`, `/business.xml` | First run: valid but newest entry is April 2024. Second run: HTTP 403. Bodies 403. |
| Business Standard Tech | `business-standard.com/rss/technology-108.rss` | Bodies 403 to our bot. |
| Mint Technology / Industry / News | `livemint.com/rss/technology`, `/industry`, `/news` | Bodies fine, but 0, 0 and 1 keyword titles (the 1 is an anecdote). |
| BusinessLine Info-Tech, The Hindu Business, Hindu topic feed | `.../info-tech/feeder/default.rss`, `thehindu.com/business/feeder/default.rss`, `.../topic/layoffs/feeder/default.rss` | Info-Tech timed out 3 times (45 s); Hindu Business valid with 1 macro hit; topic feed HTTP 522. |
| NDTV Profit | `ndtvprofit.com/rss` | HTTP 403. |
| Times of India Tech / Business | `timesofindia.indiatimes.com/rssfeeds/66949542.cms`, `.../1898055.cms` | Tech is stale (39 days). Business: all 3 sampled bodies are the same 426 characters, so the page is not returning the article. |
| Financial Express Jobs / Tech | `financialexpress.com/jobs-career/feed/`, `/life/technology/feed/` | Return HTML, not a feed. |
| Indian Express Jobs / topic "layoffs" / Business | `indianexpress.com/section/jobs/feed/`, `/about/layoffs/feed/`, `/section/business/feed/` | Jobs is stale (March 2025) and is government recruitment ads. Topic feed returns HTML. Business has 0 hits. |
| Hindustan Times Tech / Business, India Today Tech | `hindustantimes.com/feeds/rss/...`, `indiatoday.in/rss/1206550` | Only 7, 2 and 20 items, 0 keyword titles. |
| People Matters | `peoplematters.in/rss`, `/feed.rss` | HTTP 404 (the homepage advertises `/feed.rss`, which also 404s). |
| Naukri blog, foundit insights | `naukri.com/blog/feed/`, `foundit.in/career-advice/feed/` | Naukri stale (37 days); both bodies 403. foundit posts are "Insights Tracker" titles that the roundup filter would skip. |
| Indeed Hiring Lab | `hiringlab.org/feed/` | Fresh, bodies ok, but macro research with no company-level events. Not useful for extraction. |
| NASSCOM, NASSCOM Community | `nasscom.in/rss.xml`, `community.nasscom.in/rss.xml` | Valid RSS with 0 items. |
| PIB releases | `pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3` | No dates, 0 keyword titles, bodies 403. |
| The Verge (all, layoffs tag), Ars business, Ars layoffs tag, Wired layoffs tag | `theverge.com/rss/...`, `arstechnica.com/business/feed/`, `.../tag/layoffs/feed/`, `wired.com/feed/tag/layoffs/latest/rss` | Verge all: 0 hits; Verge tag, Ars business: 0 entries; Ars tag stale and 405; Wired tag 404. |
| HN search "layoffs" | `hnrss.org/newest?q=layoffs&points=20` | 1 item in 14 days. |
| Guardian business, BBC Business, CNBC Technology and Careers, Computerworld | various | Hits are UK jobs policy or macro stories (0 company events), or 0 hits in a one-day window. |
| GeekWire, HR Dive, News18 Tech | various | Bodies 403. |
| The Register | `theregister.com/headlines.atom` | 0 keyword titles. Python's `robotparser` says its `robots.txt` blocks `CareerRadarBot` on the sampled paths, yet the fetcher still downloaded them (see step 17). |
| Layoffs.fyi | `layoffs.fyi/feed` | Returns a 2020 blog feed. No API or RSS for the tracker. |
| Google News "hiring freshers IT India", "tech layoffs global" | same pattern as #15 | Same robots and body problems. |

### B6. Ranking and recommended final list

Ranking by relevant items per week among feeds whose bodies we can read (my reading of keyword-matching titles in the last 14 days; a lower bound, because non-keyword events are missed):

| Rank | Feed | Relevant per week (est.) | Notes |
|---|---|---|---|
| 1 | HRKatha | about 4 | 6 company events in a 9.3-day window; the search feed shows 7 in 14 days |
| 2 | Mint Companies | 1 to 2 (**unverified**, 1.3-day window) | Qualcomm layoffs found |
| 3 | Indian Express Technology | about 1 | Canva and Australian offshoring |
| 4 | Analytics India Magazine | about 1 (**unverified**, 5-day window) | India angle on global cuts |
| 5 | Entrackr | about 0.7 | startup job cuts |
| 6 | TechCrunch layoffs tag | about 1 | already covered by the main feed |
| (7) | HN jobs | about 4.5 job ads | not news |
| not usable | News18 Business (about 2.5 events/week), ET IT (about 1/week), Google News (about 40 headlines/week) | | body 403, paywalled, robots |

**Recommendation (sized for the capacity in Part C):** add 5 feeds, all with a `career_only` gate: HRKatha, Mint Companies, Entrackr, Indian Express Technology, Analytics India Magazine. Add the HRKatha search feed only for a one-time history seed. Hold the TechCrunch tag and HN jobs until you answer questions 9 and 10. Expected result: about 4 extra stored articles per day (range 3 to 6) and about 7 to 9 relevant articles per week.

Honest limit: this will not make the heatmap look dense on its own. Hiring news with skill names is rarer than layoff news (layoff stories carry no skills). See the ATS option in B7.

### B7. Non-RSS approaches

| Option | What I found | Pros | Cons | Verdict |
|---|---|---|---|---|
| Google News RSS search queries | Works over HTTP, 30 of 31 titles match, but `news.google.com/robots.txt` disallows `*` except a few paths, and the article links are redirect pages (body 0 characters, 3 of 3). | Best keyword precision, India and global. | Robots disallow. No readable body, so extraction would be headline-only. | **No.** Do not bypass. |
| GDELT DOC 2.0 API | `api.gdeltproject.org/api/v2/doc/doc?query=layoffs sourcecountry:IN&mode=artlist&format=json` returned 32 articles for 14 days. 17 were Mint, many only mention "layoffs" in passing. | Public JSON, no key, publisher URLs. | Low precision, small volume, rate-limit guidance **unverified**. | Not worth it now. |
| Hacker News Algolia API | `hn.algolia.com/api/v1/search_by_date?query=layoffs` works. | Public JSON, no key. | Same content as the hnrss search (1 item in 14 days). | No. |
| Public ATS job-board APIs (Greenhouse, Lever) | Live checks: Lever `cred` 15 postings, Lever `paytm` 202, Greenhouse `groww` 9; `phonepe`, `razorpay`, `postman` tokens returned 404 on Greenhouse. [Greenhouse documents](https://developers.greenhouse.io/job-board.html) the Job Board API as public JSON. | Real, structured, India-located HIRING data with skills from titles and descriptions, **no LLM needed**, daily snapshots give a true trend. | New ingestion path and a new table, not news. Needs a company list. Lever terms **unverified**. Must respect each company's board. | **Best candidate for phase 2.** Needs your decision (question 11). |
| Indeed Hiring Lab data (GitHub CSV) | README confirms daily job-postings index by country and sector. | Official, citable macro series. | A national index, not company events. India coverage and license **unverified**. | Context only. |
| Sitemaps | Not tested per site. News sitemaps list the last two days only. | Works where RSS is missing. | Same content as RSS, per-site parsing. | No. |
| Newsletters (Layoffs.fyi, TLDR) | Layoffs.fyi has no feed or API. | Curated. | Needs email ingestion, terms and parsing, fragile. | No. |
| News aggregator APIs (NewsAPI-style, Guardian Open Platform) | Not tested (need keys). Free tiers are usually development or non-commercial: **unverified**. | JSON. | Key, quota and terms to check; Guardian is UK/global. | Skip for now. |

---

## Part C. LLM capacity and API keys

### C8. Extra calls per day and what is cheaper

Assumes hourly runs, `CAREER_EXTRACTION_BATCH_LIMIT=5`, 3.3K tokens per extraction call, 0.4K per summary call.

| Scenario | New articles/day | Extra summary calls | Extra extraction calls | Extra tokens/day |
|---|---|---|---|---|
| 5 recommended feeds with `career_only` gate | about 4 (3 to 6) | +4 | +4 (+10% retries) | about +15K |
| Same 5 feeds without a gate (for comparison) | about 54 (HRKatha 5.4, Mint 27, Entrackr 5.4, IE 6, AIM 10) | +54 | +about 42 (full bodies, not roundups) | about +160K (would exceed the 200K tokens/day on one model together with today's load) |
| Pre-filter on the existing 14 feeds | | | extraction demand drops from about 40/day to about 1 to 3/day (6 of 308 pass) | about -130K |
| **Plan total (gate + pre-filter)** | about 64/day | about 64 | **about 4 to 9** | about 60K to 90K/day all workloads, plus chat and fact-check |

**Can the single key handle it?** Yes for the filtered plan: about 80 to 110 calls per day against 1K per day, and about 60K to 150K tokens per day against 200K per model per day (the high end assumes heavy chat on the same model). It does not hold for the unfiltered version, and it only holds if extraction is spaced out. Live headers for your organization are **unverified**. To read them yourself with a 1-token request: `curl -si https://api.groq.com/openai/v1/chat/completions -H "Authorization: Bearer $GROQ_API_KEY" -H "Content-Type: application/json" -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'` and read the `x-ratelimit-*` headers.

Cheaper steps, ordered by payoff:

| Step | Effect | Risk |
|---|---|---|
| Keyword pre-filter before body fetch and LLM (title + RSS summary) | Removes about 95% of extraction calls and most body fetches | Drops FUNDING-only stories. FUNDING is not in `MARKET_ACTIONS`, so heatmap and gap views do not use it (per `market.py` docstring). Medium: changes extraction behavior. |
| `career_only` gate on the new feeds | Keeps ingestion to about 4 per day | Low (default off) |
| `CAREER_LLM_CALL_DELAY_SECONDS=25` to 30 | Stays under 8K tokens per minute on one model | Longer run: 5 calls take about 2.5 minutes inside the hourly job. Env only. |
| `CAREER_LLM_MODEL=openai/gpt-oss-20b` for extraction | Half the price, and a different per-model bucket, so extraction stops competing with chat and fact-check on gpt-oss-120b. Quality is **unverified**: compare on the 41 stored full-body extractions first. | Medium until compared |
| `include_reasoning=false` and `reasoning_effort="low"` in the summary payload | Stops the 7% leaked-reasoning summaries and cuts output tokens. Parameters are in Groq's reasoning docs; behavior on your account is **unverified**. | Low, but it changes existing scraper output |
| Skip the LLM summary for some feeds | Not worth it: at about 4 per day the saving is small | n/a |
| Prompt caching | Groq's [blog](https://groq.com/blog/gpt-oss-improvements-prompt-caching-and-lower-pricing) describes a 50% discount on a cached identical prefix, and the 1.1K-token system prompt is an identical prefix. Whether it applies on the free tier is **unverified**. | None |
| Honor `retry-after` in `call_groq` for background jobs | Fewer wasted 429 requests | Low-medium (shared function) |

### C9. A second Groq key

What the documents say:
- Groq's [Acceptable Use & Responsible AI Policy](https://console.groq.com/docs/legal/ai-policy) (effective 15 Oct 2025) says customers agree not to use the services "beyond published parameters, rate limits, or use limitations, including by registering multiple accounts or orchestrating usage between multiple organizations."
- The [Services Agreement](https://console.groq.com/docs/legal/services-agreement) says Groq "has no obligation to provide multiple accounts" (3.2), bars use that violates the AUP (6.3), and lets Groq suspend access without prior notice for a breach (section 7).
- The [rate-limits page](https://console.groq.com/docs/rate-limits) says limits apply "at the organization level, not individual users." A second key in the **same** organization therefore adds no capacity.
- [Spend limits](https://console.groq.com/docs/spend-limits) are organization-wide across all keys and paid-plan only.

| Option | Capacity gain | Pros | Cons / risk | My view |
|---|---|---|---|---|
| New key from a second free account of yours | Doubles limits | Fast | This is the pattern the AUP names. Risk of suspension of both accounts, and it would sit in a project you hand in. | **Do not.** |
| Second key in the same organization (a Groq "project" or another key) | None | Revocation and per-workload attribution. The limits page says limits "can be customized per project", so a project can be capped lower to protect chat (availability on free plan **unverified**). | No extra capacity. | Allowed and useful for isolation only. |
| Teammate's key, legitimately theirs | Only if it replaces yours | Fine if the teammate owns the deployment and the key is the single production key | Using it **in parallel** with yours to get more throughput is "orchestrating usage between multiple organizations". Also their account is billed and suspended for your usage. | Only as a swap, with the teammate's agreement. |
| Paid Developer tier | Higher limits, Batch and Flex processing | Cleanest. Cost is small: third-party price pages citing Groq list gpt-oss-120b at $0.15 in / $0.60 out per 1M tokens and gpt-oss-20b at $0.075 / $0.30 (**unverified** against the live pricing page). Plan volume of about 100K tokens per day is about $0.03 per day on the 120b model. Spend limit available. | Needs a card; student budget | **Best if capacity ever becomes a problem.** |
| Different provider for some workloads | Independent limits | Legitimate; the code already has `TOGETHER_API_KEY` in `.env` (valid or not is **unverified**). Many providers expose OpenAI-compatible chat endpoints, so `call_groq` could take a base URL per workload. | Each provider's free-tier limits and terms need checking (not done). Extraction quality must be re-tested. | Good second step, for example CareerRadar elsewhere and chat on Groq. |
| Reduce calls | n/a | Plan above | Needs the code steps | **Do this first.** |
| Local model for extraction (for example Ollama on your Mac) | Unlimited locally | No rate limit | Not available on Render; extraction quality **unverified** | Option for a local-only demo. |

If a second key is approved (same organization, or a swap), the selection would work like this (not implemented):
1. Add to `backend/app/groq_client.py` a function `get_api_key(workload: str | None) -> str`. It returns `os.getenv(f"GROQ_API_KEY_{workload.upper()}")` when that variable is set and non-empty, otherwise `os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or ""`.
2. `CareerExtractor.__init__` calls `get_api_key("career")` instead of reading the two names itself. `backend/services/brief/generator.py::_api_key()` does the same with `"career"` (it already shares `CAREER_LLM_MODEL`). Chat, fact-check, summaries and history keep calling the default.
3. With `GROQ_API_KEY_CAREER` unset, every path resolves to exactly today's key, so behavior is unchanged. Tests that pass `api_key` explicitly or monkeypatch `call_groq` are unaffected.
4. `get_live_groq_models()` caches the model list globally for 15 minutes, regardless of key. Key the cache by a hash of the key if the keys can belong to different organizations.
5. No automatic key rotation or fail-over on HTTP 429. That would be the circumvention pattern.
6. Add `GROQ_API_KEY_CAREER=` (blank) to `.env.example`.

---

## Part D. Plan

Risk labels: **Existing** = changes behavior of the scraper or another feature. **Low / Med / High** = chance of breaking something. Every step has a test and a rollback. Steps are ordered so that each one can stop the plan safely.

### Phase 0. Decisions (no code)
0. Answer the questions at the end, especially the paid tier, HRKatha's terms (9), HN jobs (10), the ATS option (11), deployment target (6), and history depth (3).

### Phase 1. Measure and protect
1. **Monitoring script** `scripts/career_funnel_report.py` (new file, read-only). Prints per week and per source: articles, pre-filter passes, extracted, full_body, headline_only, signals, using the SQL in A3 (via SQLAlchemy so it runs on Neon too). *Risk: none. Test:* run on the local DB and compare with the A3 numbers (2026-W45 Hacker News 30 / 5 / 5 / 0 / 1). *Rollback:* delete the file.
2. **Env only:** `CAREER_LLM_CALL_DELAY_SECONDS=25`. *Existing (CareerRadar batches take about 2 minutes). Risk: Low. Test:* run the backfill on 3 articles with `--limit 3` and watch that no 429 appears in the log. *Rollback:* remove the variable.
3. **Env only:** `ARTICLE_RETENTION_DAYS=60` (or 90, question 3). This fixes the 30-day cliff for signals, the 60-day freshness-gate mismatch and the 8-week trend. *Existing (all articles). Risk: Low. DB growth about 60 articles per day, about 3.6K rows at 60 days; Neon storage limit **unverified**.* *Test:* `GET /api/career/trend?weeks=8` returns more than 4 weeks after a few weeks of data. *Rollback:* unset.
4. **Back up first:** copy `tech_news.db` (and, on Neon, take a branch or snapshot; availability on your plan **unverified**).

### Phase 2. Pre-filter before extraction
5. **Pure function** `career_relevance(title, summary) -> Optional[str]` in `backend/services/career/filters.py` (new function; existing roundup filter untouched), plus unit tests. *Risk: Low. Test:* on the local DB it must pass 6 of 308 articles (the script I used is in A3's finding); every pass and a sample of non-passes are reviewed by hand.
6. **Wire it into** `CareerExtractor._process_article_obj`, after the roundup check and before `ensure_article_body`. Behind a new env flag `CAREER_PREFILTER_ENABLED`, default **false**. A skipped article gets a ledger row with `text_source="skipped_not_career"` (same pattern as `skipped_roundup`), so it is not re-evaluated. The content hash must include a filter version, otherwise changing the regex will not re-open old rows. *Existing (CareerRadar). Risk: Med.* *Test:* new tests in `tests/test_career_extractor.py` (skip with flag on, unchanged with flag off); then a `--dry-run` backfill. *Rollback:* flag off; delete rows where `text_source='skipped_not_career'`.
7. **Roundup filter conflict:** `filters.py` skips titles containing "weekly" and "tracker" (for example "Layoffs tracker 2026: ..."). Keep that for now, and review skipped real layoff stories from the funnel report after a week.

### Phase 3. Scraper per-feed settings
8. **Add optional per-feed keys** (defaults replicate today): `career_only` (bool, default false), `max_entries` (int, default 15), and optionally `llm_summary` (bool, default true). Implement `feed_info.get(...)` in `fetch_and_save_all_sources`. The `career_only` gate runs right after the `SKIP_KEYWORDS` check and before dedup, og:image and the LLM summary, using the same function as step 5 on title plus RSS summary. *Existing (scraper, all feeds). Risk: Med because the function is shared by all feeds, but with no keys set the code path is identical.* *Test:* unit test with a fake feed and a stubbed `call_groq`; assert the 14 existing feeds produce the same articles as before.
9. **Summary payload fix** (optional, separate commit): add `include_reasoning=False` and `reasoning_effort="low"` to the scraper summary payload. *Existing. Risk: Low, but check one live call first.*
10. **Dedup order:** put the new career feeds **before** `Economic Times Tech` in `FEEDS`. Today the first outlet seen wins, so a paywalled ET copy of the Qualcomm story would hide HRKatha's readable copy. The market view already merges signals by company, action and ISO week, so keeping all outlets would be safe, but that changes dedup behavior and is out of scope. *Risk: Low.*

### Phase 4. Feeds (one per commit, in this order)
Exact entries to add (new optional keys from step 8; the first four lines are the same structure as today):
```python
    # ── Career / workforce news (gated) ──────────────────────────────
    {'name': 'HRKatha',                 'url': 'https://www.hrkatha.com/feed/',                              'career_only': True},
    {'name': 'Mint',                    'url': 'https://www.livemint.com/rss/companies',                     'career_only': True},
    {'name': 'Entrackr',                'url': 'https://entrackr.com/rss',                                   'career_only': True},
    {'name': 'Indian Express',          'url': 'https://indianexpress.com/section/technology/feed/',         'career_only': True},
    {'name': 'Analytics India Magazine','url': 'https://analyticsindiamag.com/feed/',                        'career_only': True},
```
11. Add HRKatha only. Run one scrape locally (`python -m backend.scrapers.techcrunch`). Expected: about 0 to 5 stored articles on the first run (11 of 50 match). *Existing (scraper). Risk: Low.* *Test:* funnel report shows source `HRKatha` with article count and no ledger rows before extraction. *Rollback:* remove the line; delete rows of that source (articles, then `article_categories`, which has no cascade; the admin delete endpoint does both).
12. Repeat for Mint, Entrackr, Indian Express, Analytics India Magazine, each after reviewing the previous one's matches by hand (precision check).
13. **One-time history seed** (optional): temporarily set `max_entries` to 50 for HRKatha, and add the search feed `https://www.hrkatha.com/?s=layoffs&feed=rss2` for one run. The feeds only reach back about 9 days (HRKatha), 9 (Entrackr), 5 (AIM), 36 (Indian Express) and 1.3 (Mint), so the earliest history you can get is limited. Remove the temporary entry afterwards.

### Phase 5. Enable extraction and backfill (relevant first)
14. Set `CAREER_PREFILTER_ENABLED=true`. Run the existing backfill CLI as a dry run with title keywords, for example: `python scripts/backfill_career_signals.py --unprocessed --title-keyword layoff --title-keyword "laid off" --title-keyword hiring --title-keyword "job cuts" --title-keyword headcount --title-keyword workforce --limit 20 --dry-run`. The CLI already supports this relevant-first selection, so no new code is needed. Do not use `--keyword-filter` together with `--unprocessed`: in the current script `--unprocessed` takes the title-keyword branch and ignores `--keyword-filter`, and the built-in `--keyword-filter` list includes funding words ("funding", "raises", "expands", "office"). Then run it for real without `--dry-run`. The old unprocessed articles that do not match are skipped by the pre-filter without LLM calls (about 1 of 255 passes). *Existing (CareerRadar). Risk: Low-Med.* *Test:* funnel report; `GET /api/career/company-board` shows only real events. *Rollback:* `CAREER_RADAR_ENABLED=false`.
15. Set `CAREER_RADAR_ENABLED=true` for the hourly hook only after step 14 looks right. Keep `CAREER_EXTRACTION_BATCH_LIMIT=5`.
16. **Optional:** `CAREER_LLM_MODEL=openai/gpt-oss-20b` after comparing outputs on the 41 stored extractions. *Risk: Med until compared.*

### Phase 6. Hygiene needed before scale (small, independent)
17. `body_fetcher.py` does not read `robots.txt` (The Register's rules disallow our bot, yet the fetch succeeded) and its User-Agent contact is `admin@technewsdigest.local`, which does not resolve. Add a cached robots check and a real contact. *Existing (CareerRadar). Risk: Low-Med.*
18. Protect `GET /api/cron/scrape` with a secret before extraction runs on a public URL (PROJECT_STATE already lists it as open). *Existing. Risk: Low, but the caller (Vercel cron or pinger) must send the secret: whatever calls it is **unverified**.*

### Phase 7. Only if approved
19. Per-workload key (`GROQ_API_KEY_CAREER`), exactly as in C9, only for the same-organization or swap cases.
20. Phase 2 data source: public ATS job-board snapshots (Greenhouse, Lever) for a list of Indian companies, stored as structured HIRING counts with no LLM. Needs question 11.

### Monitoring
- `scripts/career_funnel_report.py` weekly, by source: articles, pre-filter passes, extracted, headline-only, signals.
- `GET /api/admin/stats` already exists; I would add per-model call counters from `call_groq` (model used, 429 count) there. Optional and **Existing**.
- Weekly "real events" review: 10 random signals, check each quote against the article by hand.

### Rollback summary
Each step above lists its own. In all cases: set the env flags back (`CAREER_RADAR_ENABLED=false`, `CAREER_PREFILTER_ENABLED=false`), remove the feed line, delete the source's rows (articles, then `article_categories`, then the cascade handles `career_*` and `article_bodies`), or restore the DB copy from step 4.

---

## D11. Risks to other features and how the plan avoids them

| Risk | Where it comes from | Mitigation in the plan |
|---|---|---|
| Rate limits starve chat, fact-check or the brief | All use gpt-oss-120b first, one organization, 8K tokens per minute; an extraction batch of 5 uses the whole minute | Pre-filter (about 95% fewer calls), 25 s spacing, extraction on gpt-oss-20b (separate bucket) after a quality check, batch limit stays 5 |
| 429 amplification | `call_groq` retries each model twice and walks the whole chain, with no `retry-after` | Fewer calls; optional `retry-after` handling |
| DB growth | 54 per day if the new feeds were ungated | `career_only` gate keeps it at about 4 per day; retention 60 is about 3.6K rows |
| Retention deletes signals | 30-day default; `career_signals` cascades on article delete; API allows 90 days and trend 8 weeks | `ARTICLE_RETENTION_DAYS=60` (or 90). A separate retention for articles that have signals would need a change in `cleanup_old_articles` (not in this plan unless you want it) |
| Vercel timeouts / no body fetch | On Vercel, CareerRadar uses cached bodies only and batch 3; a 25 s delay times 3 calls would not fit a serverless limit (exact limit **unverified**). `vercel.json` sends `/api/*` to Render anyway | Run the scheduler on Render or locally (question 6) |
| Paywalled copy hides a readable one | Cross-publisher dedup drops later outlets | Put career feeds before ET in `FEEDS` |
| Publisher terms | HRKatha forbids reproduction | Store only headline, link and our own summary; do not show RSS text; `article_bodies` stays internal (already the rule); your call (question 9) |
| Leaked-reasoning summaries (22 of 308) | `call_groq` falls back to the `reasoning` field when `content` is empty | Step 9 |
| `/api/cron/scrape` is public | Anyone can trigger LLM spend | Step 18 |

---

## D12. What I could NOT verify

- Your organization's live Groq limits and `x-ratelimit-*` headers (no API call made). Which plan the docs table describes. Limits of `llama-3.1-8b-instant`. Whether rejected 429 requests count toward RPM and RPD. Whether TPM counts `max_tokens`.
- Groq prices (taken from third-party pages that cite Groq's pricing page) and whether prompt caching applies on the free tier.
- Whether gpt-oss-20b matches gpt-oss-120b for extraction. Whether HRKatha's short bodies (1.7K to 2K chars) produce grounded quotes.
- Steady-state ingestion rate (3 days observed). Weekly rates for Mint Companies and Analytics India Magazine (feed windows of 1.3 and 5 days).
- Relevance counts only cover keyword-matching titles, so they are lower bounds. I did not review non-matching titles.
- Terms of use for every publisher except HRKatha. `robots.txt` for BusinessLine and The Hindu (timed out). BusinessLine Info-Tech feed (timed out three times; may be slow from here rather than dead).
- What calls `/api/cron/scrape` in production, whether Render sleeps, the runtime Python version, Neon storage limits and branching on your plan.
- Whether production already runs CareerRadar. Whether signal row 4 (Google HIRING) is your test row.
- India coverage and license of the Indeed Hiring Lab dataset. Lever's terms. GDELT rate limits.
- Groq "project" limits on the free plan.

---

## Questions for you

1. Is this a solo project or a team project, and who owns the Groq account that will run in production?
2. Is a paid Groq tier acceptable (about cents per day at this volume, with a monthly spend limit), or must it stay on the free tier?
3. How many days of history should the demo show (30, 60 or 90)? That decides `ARTICLE_RETENTION_DAYS`. Most feeds only reach back 1 to 9 days, so a 30-day history needs 4 to 5 weeks of collection from now.
4. India only, or India plus global? The plan above is India-first with some global stories carried by HRKatha and Indian Express.
5. Are FUNDING signals needed? The pre-filter drops funding-only stories. The market views do not use FUNDING.
6. Where will the demo run: Render (scheduler on), your laptop, or Vercel only? Vercel alone cannot fetch bodies or run extraction (batch 3, cached bodies only).
7. Is headline-only data acceptable for some Indian outlets (News18, Business Standard, ET)? Today those produce no signals. I recommend not spoofing a browser User-Agent to get bodies.
8. Do you want the 7% leaked-reasoning summaries fixed (step 9), even though it changes existing scraper output?
9. HRKatha's Terms of Service forbid reproduction of its content. Do you accept storing only headline, link and our own summary (and not showing its RSS excerpt), or should HRKatha be dropped?
10. Do you want Hacker News "jobs" (YC job ads) as a source of HIRING signals with skills? They are global, not India, and they are ads, not news.
11. Do you want a phase 2 that reads public ATS job-board APIs (Greenhouse, Lever) for a list of Indian companies? It would give real hiring counts and skills without LLM calls, but it is a new feature and needs a company list from you.
12. If capacity does become a problem, which do you prefer: paid tier, moving chat or CareerRadar to another provider, or only reducing calls? I do not recommend a second free account.
