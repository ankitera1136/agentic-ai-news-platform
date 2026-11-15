# PROJECT_STATE: agentic-ai-news-platform

Snapshot date: 2026-11-15. Built from the files, git state and local SQLite DB on that date.
Items I could not confirm are marked **unverified**. No secret values appear here.

## 1. Stack

| Item | Value |
|---|---|
| Backend | FastAPI (`fastapi>=0.115`), uvicorn, SQLAlchemy 2.x (sync), Pydantic v2, APScheduler (AsyncIOScheduler), slowapi (rate limits), cachetools (TTLCache) |
| Python | Local venv runs 3.15.2. Vercel/Render runtime version is **unverified** (no `runtime.txt`, no `.python-version`). |
| Frontend | Server-rendered Jinja2 templates with inline vanilla JS in each template. No React/Vue, no build step, no `package.json`. |
| Frontend serving | FastAPI serves the templates (`Jinja2Templates`) and mounts `/static` from `frontend/static`. |
| Styling | Hand-written `frontend/static/css/style.css` (991 lines, dark glassmorphic theme) plus per-template `<style>`. Font Awesome 6.4 and Google Fonts from CDN. three.js on the home page, `marked` on the chat page. The README says Tailwind CDN, but no template references it. |
| LLM | Groq HTTP API via `requests` (`backend/app/groq_client.py`). It discovers live models, caches them 15 min, and falls back down a chain (`openai/gpt-oss-120b` first). The `groq` SDK is in requirements but unused (**unverified**). |
| Dedup/NLP | sentence-transformers, spaCy `en_core_web_sm`, datasketch (MinHash), numpy/scipy |
| Auth libs | python-jose (JWT HS256), bcrypt, passlib. `fastapi-users` is installed but not used (**unverified**). |
| DB drivers | sqlite (stdlib), psycopg2-binary (used), pg8000 and pymysql (listed, unused) |
| Scraping | requests, feedparser, beautifulsoup4 |
| Build tools | None. Run with `python main.py` or `uvicorn backend.app.webapp:app`. |
| Heavy deps | sentence-transformers and spaCy are in the single `requirements.txt`. This is a size risk for Vercel serverless (**unverified**). |

## 2. Structure

```
agentic-ai-news-platform/
├── main.py                    uvicorn entry (PORT env, reload=True); imports backend.app.webapp:app
├── api/index.py               Vercel serverless entry; re-exports the FastAPI app
├── app/webapp.py              back-compat shim (app.webapp:app) for hosts that expect it
├── vercel.json                builds api/index.py; proxies /api/*, /auth/*, /ping to the Render backend; rest to api/index.py
├── .vercelignore              excludes .venv, .env, tech_news.db, *.md, etc.
├── requirements.txt           single dependency list
├── .env / .env.example        config (see section 9)
├── backend/
│   ├── app/
│   │   ├── webapp.py          1119 lines: ALL routes, scheduler, startup, admin logic, chat RAG, fact-check route
│   │   ├── models.py          SQLAlchemy models (section 3)
│   │   ├── schemas.py         Pydantic request/response models
│   │   ├── auth.py            JWT, bcrypt, get_current_* dependencies, create_user
│   │   ├── database.py        engine, SessionLocal, get_db, init_database() (create_all + migrations + seed categories)
│   │   ├── groq_client.py     call_groq() with live model discovery and fallback
│   │   └── config.py          legacy (TOGETHER_API_KEY, 2 source URLs); effectively unused
│   ├── scrapers/techcrunch.py 14 RSS feeds -> JSON files + DB rows (name is historical)
│   └── services/
│       ├── categorization_service.py  keyword categorizer, file->DB sync, dedup hook, retention cleanup
│       ├── claim_extractor.py         LLM claim extraction with quote grounding (used by fact-check)
│       ├── claim_normalizer.py        deterministic number/unit/currency comparison
│       ├── contradiction_engine.py    cross-source agreement index and discrepancies
│       ├── digest_service.py          picks articles per user interests (no email sending found)
│       ├── history_service.py         "today in tech history" (cached JSON in data/history + Groq)
│       └── career/                    CareerRadar (section 8)
├── career/migrate.py          idempotent CareerRadar table migration (SQLAlchemy and raw-SQLite paths)
├── dedup/                     3-layer story clustering: copy_detector (MinHash), clusterer (embeddings+NER), verifier (Groq), pipeline, migrate, backfill, config
├── eval/                      clustering benchmark / labeling tools + ground_truth_pairs.json
├── scripts/
│   ├── backfill_career_signals.py   CareerRadar CLI
│   └── migrate_to_postgres.py       SQLite -> PostgreSQL copy tool
├── frontend/
│   ├── templates/             index.html (2318 lines), dashboard.html (521), admin.html (998), chat.html (326)
│   └── static/                css/style.css, js/app.js (card glow, scroll animations)
├── data/
│   ├── summaries/             one JSON per scraped article (463+ files, tracked in git)
│   └── history/               cached "today in tech" JSON per date
├── tests/                     pytest suite (section 10)
├── tech_news.db (+ -shm/-wal) local SQLite DB (git-ignored)
└── PROJECT_STATE.md           this file
```

## 3. Database

**Dev vs prod.** Dev defaults to `sqlite:///./tech_news.db`, in WAL mode with foreign keys on. On Vercel it defaults to `sqlite:////tmp/tech_news.db`, which is ephemeral. Production is intended to be Neon PostgreSQL via `DATABASE_URL`. `postgres://` and `postgresql://` URLs are rewritten to `postgresql+psycopg2://`. The README and `vercel.json` point at a Render backend (`agentic-ai-news-platform.onrender.com`). Whether production is actually running on Postgres is **unverified**.

**Migrations.** There is no Alembic. `init_database()` runs on startup:
1. `Base.metadata.create_all`.
2. `dedup.migrate.upgrade()`.
3. `career.migrate.upgrade()`.
4. Ad-hoc `ALTER TABLE ... ADD COLUMN` in try/except (`articles.source`, `users.is_admin`).
5. It then seeds or overwrites the default categories on every start (keywords and descriptions are reset).

Any new column on an existing table needs its own explicit migration, because `create_all` does not alter tables.

**Local row counts:** users 1, categories 19, articles 299, article_categories 912, clusters 547, cluster_log 4856, pair_cache 22, user_subscriptions 0, user_interests 0, digest_logs 0, notification_logs 0, article_bodies 57, career_extractions 53, career_signals 3.

| Table (model) | Columns | Relationships |
|---|---|---|
| `users` (User) | id (String UUID PK), email (unique, indexed), hashed_password, full_name, is_active, is_verified, is_admin, created_at, daily_digest_enabled, weekly_digest_enabled, instant_notifications, digest_time ("HH:MM"), time_zone | interests -> categories via `user_interests`; subscriptions -> `user_subscriptions` |
| `user_interests` | user_id (FK users.id), category_id (FK categories.id). No PK. | many-to-many |
| `categories` (Category) | id, name (unique), description, keywords (JSON) | users, articles |
| `articles` (Article) | id (String PK, derived from filename), title, link, summary, llm_summary, published (String, not datetime), image_url, source, created_at, cluster_id (FK clusters), is_copy_of (FK articles), embedding (bytes), entities (JSON), minhash (bytes), text_basis, clustered_at | categories, cluster |
| `article_categories` | id, article_id, category_id, relevance_score (1-10) | article <-> category |
| `user_subscriptions` | id, user_id, subscription_type, category_id (nullable), keywords (JSON), is_active, created_at | user |
| `digest_logs` | id, user_id, digest_type, sent_at, article_count, email_status | none declared |
| `notification_logs` | id, user_id, article_id, notification_type, sent_at, status | none declared |
| `clusters` (Cluster) | id, primary_article_id (plain String, no FK), created_at, updated_at, size | articles |
| `pair_cache` | a_id + b_id (composite PK), verdict (SAME/DIFFERENT), created_at | none |
| `cluster_log` | id, article_id, decision, best_cluster_id, best_score, layer, created_at | none |
| `article_bodies` | article_id (PK, FK articles ON DELETE CASCADE), body_text, fetched_at, status (ok/failed/paywalled/too_short), http_status, error, attempts | internal only, never expose |
| `career_extractions` | article_id (PK, FK CASCADE), content_hash, text_source (full_body/stored_summary), status (succeeded/failed), attempts, model_used, signals_count, error, extracted_at | idempotency ledger |
| `career_signals` | id, source_article_id (FK CASCADE, indexed), company (indexed), company_raw, action (indexed), timing (current/planned/past_context), count, location_city, location_country, skills_mentioned (JSON), confidence, evidence_quote, extracted_at | per-article signals |

Notes:
- `CareerSignal` has no relationship attributes and no unique constraint. Re-extraction deletes and re-inserts the article's rows.
- Local SQLite has no ON DELETE behaviour issues, but the foreign-key pragma is only enabled by the app engine.

## 4. API

Auth legend: **none** = public; **user** = Bearer JWT of an active user; **admin** = Bearer JWT of an admin.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET, HEAD | `/ping`, `/health`, `/healthz` | none | keep-alive, returns "OK" |
| GET | `/` | none | HTML home; renders the article feed server-side |
| GET | `/chat` | none | chat page HTML |
| POST | `/chat` | none (10/min per IP) | RAG answer from the latest 150 articles via Groq |
| GET | `/dashboard` | none (page; JS checks the token) | user dashboard HTML |
| GET | `/admin` | none (page; JS checks the token) | admin console HTML |
| POST | `/auth/register` | none (5/min per IP) | create account (JSON `UserCreate`) |
| POST | `/auth/token` | none | login; accepts form or JSON (`username` or `email`, `password`); returns JWT |
| POST | `/auth/google` | none | exchange a Google ID token for a JWT; auto-creates the user |
| GET | `/auth/me` | user | current user; auto-elevates `ADMIN_EMAIL` |
| GET | `/categories` | none | all categories |
| GET | `/preferences` | user | digest preferences plus interested category ids |
| PUT | `/preferences` | user | update digest preferences and interests |
| POST | `/subscriptions` | user | create a subscription |
| GET | `/subscriptions` | user | list own subscriptions |
| DELETE | `/subscriptions/{subscription_id}` | user | delete own subscription |
| GET | `/digest/preview` | user | preview a digest |
| POST | `/digest/send` | user | generate a digest for self (no SMTP code found, see section 11) |
| GET | `/digest/history` | user | last 20 digest logs |
| GET | `/articles` | none | articles, optional category filter |
| GET | `/articles/personalized` | user | articles matching the user's interests |
| GET | `/api/articles/feed` | none | paginated feed: `page`, `limit=40`, `category`, `search`; cluster-aware |
| GET | `/api/articles/{article_id}/cross-check` | none | fact-check and contradiction audit (1h TTL cache, max 512) |
| GET | `/api/clusters/{cluster_id}/articles` | none | member articles of a cluster |
| GET | `/api/tech-history/today` | none | "today in tech history" |
| GET | `/api/cron/scrape` | **none** | runs the full scrape pipeline (Vercel cron / external pinger) |
| GET | `/api/admin/stats` | admin | system stats |
| GET | `/api/admin/articles` | admin | searchable article list |
| POST | `/api/admin/articles/{article_id}/resummarize` | admin | re-run the LLM summary |
| DELETE | `/api/admin/articles/{article_id}` | admin | delete an article |
| GET | `/api/admin/users` | admin | user directory |
| PUT | `/api/admin/users/{user_id}/role` | admin | toggle admin flag (see bug in section 11) |
| GET | `/api/admin/logs` | admin | in-memory scraper and admin log (last 100) |
| GET, POST | `/api/admin/ai-settings` | admin | read or update Groq model, temperature, max_tokens (persisted to `data/ai_settings.json`) |
| POST | `/api/admin/trigger-scrape` | admin | manual scrape in a worker thread |

**There is no CareerRadar or student-profile route yet.**

Cross-cutting: CORS is `allow_origins=["*"]` with credentials disabled. Rate limits (slowapi, per IP) apply only to `/auth/register` and `/chat`.

## 5. Auth and users

- **Login:** `POST /auth/token` checks the email (case-insensitive) and bcrypt password. It issues a JWT (HS256, `sub` = email, 30-minute expiry, no refresh). Google sign-in is `POST /auth/google` using Google's `tokeninfo` endpoint. A first Google login creates a user with a random password.
- **Client side:** the token is kept in `localStorage['authToken']` and sent as `Authorization: Bearer <token>`. `OAuth2PasswordBearer(tokenUrl="auth/token")` extracts it.
- **Current user:** `get_current_user` decodes the JWT and looks the user up by email. `get_current_active_user` adds an `is_active` check. `get_current_admin_user` requires `is_admin`, and it auto-promotes the `ADMIN_EMAIL` user. Use these dependencies on new routes.
- **Admin vs normal:** `User.is_admin`. Admin pages call `/api/admin/*`. The `/admin` and `/dashboard` pages are public HTML; the guard is in client JS plus the API.
- **Existing profile and preferences fields:** `full_name`, the digest fields (daily/weekly/instant, `digest_time`, `time_zone`), and `interests` (category many-to-many). There is **no** student, course, skills, location or career-goal field anywhere.
- **Local data:** 1 user, an admin.

## 6. Frontend

| Template | Route | Role |
|---|---|---|
| `index.html` | `/` | main feed: Explore, category filters, search, infinite scroll, cluster "covered by N outlets", cross-check modal, login/register/Google modal, bookmarks, tech-history widget |
| `chat.html` | `/chat` | AI assistant chat (anonymous, no JWT sent apart from a token read) |
| `dashboard.html` | `/dashboard` | tabs: Preferences, Saved stories, Account |
| `admin.html` | `/admin` | admin console: stats, articles, users, logs, AI settings, manual scrape |

- **API calls:** plain `fetch()` inside each template's inline `<script>`. Index uses `/api/articles/feed`, `/api/articles/{id}/cross-check`, `/api/tech-history/today`, `/auth/*`. Admin uses `/api/admin/*`. Chat uses `POST /chat`.
- **Dashboard gap:** the dashboard loads only `/auth/me`. `saveCategoryPreferences()` shows a success message but makes **no** API call, and the Account tab is not wired to any save endpoint. Saved stories use `localStorage['techcore_bookmarks']`.
- **Navigation:** top navbar in `index.html` (Explore `/`, AI Assistant `/chat`, Dashboard `/dashboard` shown after login, Admin link injected for admins). Each page repeats its own navbar markup; there is no shared base template.
- **Where new pages fit:**
  - A **CareerRadar page** is a new template `frontend/templates/career.html` plus a `GET /career` page route in `webapp.py`, and a nav link added in `index.html` (and the other templates). It would call a new `/api/career/*` JSON endpoint.
  - A **profile form** fits as a new tab in `dashboard.html` (next to Preferences, Saved, Account). It needs real endpoints (for example `GET/PUT /api/profile`) and a new table or columns. The existing tab buttons use the `dash-nav-btn` and `dash-panel` pattern.

## 7. Pipeline

All scheduled work runs through `_run_scrape_sync()` in `webapp.py`:

1. **Scrape** (`fetch_and_save_all_sources`): 14 RSS feeds (TechCrunch, Ars Technica, MIT Tech Review, Wired, Inc42, YourStory, Economic Times Tech, MediaNama, The Hindu Tech, GitHub Blog, Stack Overflow Blog, freeCodeCamp, Towards Data Science, Hacker News). It takes the top 15 entries per feed, does a headline-similarity dedup against the last 72h and local files, makes an LLM summary via Groq, and writes `data/summaries/*.json`. The Verge appears in the DB, but is not in the current feed list (**unverified** where those rows came from).
2. **Sync** (`categorizer.sync_articles_from_files`): JSON files -> `articles` rows, keyword categorization (`article_categories`), and `dedup` pipeline per new article.
3. **Dedup/cluster** (`categorizer.deduplicate_existing_database`): `dedup.pipeline` runs 3 layers: MinHash copy detection (sets `is_copy_of`), sentence-embedding + NER clustering with time decay (`cluster_id`), and a Groq pair verifier (cached in `pair_cache`). `DEDUP_MODE` selects the mode.
4. **CareerRadar** (only if `CAREER_RADAR_ENABLED` is true; off locally): `career_extractor.process_pending_articles()` handles a batch.
5. **Retention** (`cleanup_old_articles`): deletes articles older than `ARTICLE_RETENTION_DAYS` (default 30), keeping a minimum.

**How it is triggered:**
- Local/Render: an APScheduler job runs hourly, and one run starts at server startup after a full recategorization.
- Vercel: the scheduler is disabled when `VERCEL` or `NOW_REGION` is set; scraping must be triggered through `GET /api/cron/scrape`. The cron config is not in `vercel.json`, but `vercel.json` proxies `/api/*` to Render, so the cron hits Render (**unverified** what actually calls it).
- Manual: `POST /api/admin/trigger-scrape` (admin).

**Fact-check and contradiction engine:** on demand only. `GET /api/articles/{id}/cross-check` gathers cluster members (or related articles within 48h), runs `GroundedClaimExtractor` (Groq, quote-grounded), `claim_normalizer` (deterministic comparisons), and `ContradictionEngine.analyze_story_cluster`. The result is cached in a `TTLCache`. It is not part of the scheduled job.

**Chat:** `/chat` does keyword retrieval over the latest 150 articles and calls Groq with the active AI settings.

## 8. CareerRadar status

**Built:**
- `backend/services/career/`: `body_fetcher.py` (fetch and cache full bodies; paywall/short/failed handling), `filters.py` (title-based roundup filter), `extractor.py` (`CareerExtractor`, `career_extractor` singleton, batch + single-article flow, validation), `grounding.py` (quote, count, location and skill grounding; `DropCounters`), `normalizer.py` (company alias canonicalization), `prompt.py`, `repository.py`, `schemas.py`.
- Tables: `article_bodies`, `career_extractions`, `career_signals` (section 3), migrated by `career/migrate.py` and called from `init_database()`.
- Scheduler hook in `_run_scrape_sync()`, gated by `CAREER_RADAR_ENABLED`. **It is `false` in the local `.env` and in `.env.example`.**
- CLI `scripts/backfill_career_signals.py`. Options: `--limit`, `--article-id`, `--max-per-source`, `--source`, `--keyword-filter`, `--title-keyword`, `--unprocessed`, `--dry-run`, `--force`, `--reextract-all`, `--no-fetch`, `--no-table`, `--model`.

**Extraction rules (current: `PROMPT_VERSION = "v4"`, `EXTRACTION_VERSION = "career-v2"`; both feed the content hash):**
1. **Roundup filter (code, before any fetch or LLM call):** titles matching quiz, newsletter, roundup, weekly, deals, tracker, "top N"/"N best" lists, or stock/market rally reports. The ledger row is `succeeded` with `signals_count = 0` and `text_source = "skipped_roundup"`. Counted as `skipped_roundup` in batch stats.
2. **Headline-only:** when the only text is the stored summary (`text_source = "stored_summary"`, e.g. paywalled pages), no LLM call is made and no signals are stored. The ledger row stays, `signals_count = 0`. Counted as `headline_only`. The old 0.7 gate and 0.6 confidence cap were removed.
3. **No OTHER:** the prompt offers six actions (HIRING, LAYOFF, HIRING_FREEZE, EXPANSION, FUNDING, NEW_OFFICE). A returned `OTHER` is dropped and counted as `other_action`. `OTHER` stays in `ACTIONS` and the table so old rows remain valid.
4. **Timing:** `past_context` signals are dropped and counted.
5. **Counts:** only from announced hiring/layoffs (prompt rule); percentages and unsupported numbers are nulled by grounding.
6. **Rate limit:** `CAREER_LLM_CALL_DELAY_SECONDS` (default 1.0) spaces Groq calls within one extractor instance. A rejected configured model is warned about once and dropped.

**Data state (local DB, after the v4 reset and re-extraction):**
- `career_extractions` 53, all `succeeded`: 41 full_body, 7 stored_summary (headline-only), 5 skipped_roundup.
- `career_signals` 3: Bitdeer NEW_OFFICE, Ember FUNDING, Ember EXPANSION.
- `article_bodies` 57 (ok 47, paywalled 7, failed 2, too_short 1).
- 246 of 299 non-copy articles have no ledger row.
- Only 2 articles in the DB have layoff/hiring-type words in the title; both produced 0 signals.

**Known quality problems:**
- 3 signals from 53 articles, none of them hiring or layoffs. The data set has little hiring/layoff news.
- The Bitdeer signal is wrong: a data-center opening in an article about Lutnick, not a workforce event.
- The Ember EXPANSION signal is labelled `current` but the quote is a plan.
- Economic Times Tech articles are mostly paywalled, so Indian coverage is thin (7 of 10 sampled were headline-only).
- Groq free-tier rate limits still cause failed attempts in large batches; failed rows are retried on later runs (3 attempts per content hash).

**Not done:**
- No API endpoint, no UI, and no student-profile matching.
- No aggregation queries or pagination.
- 246 articles have not been processed.
- The scheduler hook has not been exercised on the new code against live scraping (**unverified**).

## 9. Config

**Environment variable names** (values not shown):

| Name | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL. SQLite by default, Neon Postgres in prod. |
| `PG_DATABASE_URL` | used by the Postgres migration script (**unverified** other uses) |
| `DATABASE_PATH` | SQLite file path override used by the dedup and career raw-SQLite migration fallbacks |
| `SECRET_KEY` | JWT signing key. A hard-coded default exists if unset. |
| `ADMIN_EMAIL` | address that is auto-granted admin |
| `GROQ_API_KEY` / `GROK_API_KEY` | Groq API key (both names accepted) |
| `GROQ_MODEL` | preferred Groq model override |
| `XAI_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY` | listed in `.env.example`. Only `TOGETHER_API_KEY` appears in code (legacy `config.py`). The rest are **unverified**. |
| `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `FROM_EMAIL`, `FROM_NAME` | in `.env.example`; no code reading them was found |
| `DEDUP_MODE` | clustering mode (`cluster` default) |
| `ARTICLE_RETENTION_DAYS` | retention window (default 30) |
| `PORT` | `main.py` listen port (default 8000) |
| `VERCEL`, `NOW_REGION` | set by Vercel; switch to /tmp SQLite and disable the scheduler |
| `CAREER_RADAR_ENABLED` | feature flag for the scrape-job hook (default false; false in local `.env`) |
| `CAREER_EXTRACTION_BATCH_LIMIT` | articles per run locally (default 25) |
| `CAREER_EXTRACTION_BATCH_LIMIT_VERCEL` | batch size on Vercel (default 3); cache-only bodies there |
| `CAREER_LLM_MODEL` | optional Groq model override for CareerRadar (blank now) |
| `CAREER_FETCH_DELAY_SECONDS` | delay between body fetches (default 0.5) |
| `CAREER_LLM_CALL_DELAY_SECONDS` | minimum delay between Groq calls in CareerRadar batches (default 1.0) |

The Google OAuth client ID is hard-coded in `index.html` and `admin.html`; it is not an env var.

**Deployment:**
- **Vercel:** frontend and API entry `api/index.py`, with `/api/*`, `/auth/*`, `/ping` proxied to a Render service.
- **Render:** runs the persistent backend (scheduler on).
- **Neon:** PostgreSQL.
- **Local:** `python main.py`.

The actual production state is **unverified**.

## 10. Tests and git

**Layout:** `tests/` with `conftest.py` (a `--slow` option; `career_db` temp-SQLite fixture; `make_article` factory).

| File | Tests |
|---|---|
| `test_career_extractor.py` | 53 |
| `test_career_grounding.py` | 38 |
| `test_career_body_fetcher.py` | 15 |
| `test_career_migration.py` | 5 |
| `test_groq_client_fallback.py` | 1 |
| `test_event_clusterer.py` | 20 |
| `test_copy_detector.py` | 6 |
| `test_verifier.py` | 5 |
| `test_legacy.py` | 3 |
| `test_migrate.py` | 1 |

**Run:** `.venv/bin/python -m pytest tests/ -q` (add `--slow` for model-download tests).
**Latest result: 146 passed, 1 skipped, 0 failed.** The skip is `tests/test_event_clusterer.py:246`, which needs `--slow` and a model download.
There are no API/route tests and no frontend tests. CareerRadar tests mock the LLM and the body fetch.

**Git:**
- Branch `main`, remote `origin` = `https://github.com/ankitera1136/agentic-ai-news-platform.git`. `phase1` (`a31230e`) is pushed.
- CareerRadar step 1 is committed locally as "CareerRadar step 1: career signal extractor" and **not pushed**. See `git log` for the hash.
- Not committed by design: `.env`, `tech_news.db` and its `-shm`/`-wal` files (git-ignored), and new `data/summaries/*.json` files from scraping (untracked).
- Previous commits (newest first): `phase1`; Update README with platform architecture, features, and setup guide; Clean up repository: untrack academic report files and redundant scraper markdown; Enhance LLM claim extraction with robust JSON parsing; Implement 3-layer story clustering, deduplication pipeline, and Neon PostgreSQL migration; Proxy auth and login endpoints to Render backend; Proxy API calls to Render backend; Configure Vercel serverless deployment and update categorization service; feat(ai): time-aware RAG retrieval, executive prompt engineering, and clean output formatting; fix(ai, categorization): dynamic Groq live discovery and enhanced 100% article categorization taxonomy.

## 11. Conventions and risks

**Conventions to follow:**
- Routes live in the single `backend/app/webapp.py`; request/response models in `schemas.py`; ORM models in `models.py`. Business logic sits in `backend/services/*` as module-level singletons (`categorizer`, `digest_service`, `career_extractor`).
- Always use `Depends(get_db)` for sessions and `get_current_active_user` / `get_current_admin_user` for auth.
- Imports are absolute (`backend.app...`, `backend.services...`). Some modules keep a `try/except ImportError` fallback with relative imports (those fallbacks are broken: `from ...career.migrate`).
- Errors: `HTTPException` in routes; services catch broad exceptions and `print("[TAG] ...")`. CareerRadar uses the `logging` logger `career_radar`.
- LLM calls always go through `call_groq()`. CareerRadar validates every LLM item with Pydantic and drops anything not grounded in the article text, counting each drop.
- Config via `os.getenv` read at call time or import time. Templates carry their own inline JS and CSS.

**Known bugs and risks:**
- `PUT /api/admin/users/{user_id}/role` declares `user_id: int`, but `User.id` is a string UUID. FastAPI would reject a UUID path value with a 422 (**unverified at runtime**). The admin UI likely cannot toggle roles.
- `/api/cron/scrape` is unauthenticated and can be hit by anyone to start a full scrape (Groq cost, load).
- `/auth/google` does not check the token's `aud` against your client ID or `email_verified`. Any valid Google ID token is accepted.
- `SECRET_KEY` falls back to a hard-coded default; JWTs are 30 minutes with no refresh; tokens are stored in `localStorage`.
- CORS is open to all origins.
- The dashboard Preferences "Save" is a stub (no API call). No user preference is persisted from the UI.
- `/digest/send` and the `SMTP_*` settings: no code that sends email was found (**unverified**; `digest_service` appears to only select articles).
- `init_database()` overwrites category keywords on every startup. `categories` holds 19 rows although the seed defines 11 (extra rows are unexplained).
- Category and `user_interests` use string user ids; `user_interests` has no primary key or cascade.
- `articles.published` is a string, so date logic relies on parsing helpers in `webapp.py`.
- `sync_articles_from_files` and the scraper use relative paths (`data/summaries/`), so they depend on the current working directory.
- JSON summaries are committed to git (`data/summaries/`), so every scrape dirties the repo.
- On Vercel the SQLite DB is in `/tmp` (ephemeral), and heavy NLP dependencies may exceed serverless limits.
- Deprecated APIs: `datetime.utcnow()` and `@app.on_event("startup")` (many warnings in tests).
- Groq free-tier rate limits are hit constantly during CareerRadar batches. Failures are retried on later runs (3 attempts per content hash).
- `README` clone URL and the Tailwind claim are out of date.
- Some earlier guidance in chat used wrong admin paths. The real ones are in section 4.

**Risks specific to adding a student profile + API + UI:**
- Schema changes need hand-written migrations (no Alembic). The ad-hoc `ALTER` pattern in `init_database()` and `career/migrate.py` (SQLite and Postgres paths) is the model to copy. Test on both dialects.
- `User.id` is a UUID string, so any new foreign key to users must use `String`.
- Profile data is personal. The current stack has no email verification (`is_verified` is never enforced), no encryption at rest, and CORS `*`. Fix the Google audience check and the open cron route before storing personal profiles.
- All routes are in one 1100-line file; adding more there raises merge conflicts. A new `APIRouter` module is a safer fit, but this is a new pattern for the repo.
- The templates duplicate navbars, CSS and token handling, so every page change must be repeated in up to 4 files.
- CareerRadar matching to a student needs queries over `career_signals` (company, action, location, skills), and `skills_mentioned` is a JSON column. Filtering on it differs between SQLite and Postgres.
- Signal quality is not yet reliable enough to show to users (3 signals from 53 articles, 2 of them questionable; see section 8).
- `article_bodies.body_text` must never be returned by any API.
