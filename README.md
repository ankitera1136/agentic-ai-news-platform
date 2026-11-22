# Agentic AI News Platform

A tech news intelligence platform. It scrapes RSS feeds, groups duplicate stories, cross-checks claims across outlets, and extracts hiring and layoff signals for a student-focused career view (CareerRadar).

Built with FastAPI, SQLAlchemy (SQLite locally, PostgreSQL/Neon in production), Groq LLMs, and server-rendered Jinja2 pages.

---

## Features

- **Multi-source scraping.** 14 RSS feeds (TechCrunch, Ars Technica, MIT Tech Review, Wired, Inc42, YourStory, Economic Times Tech, The Hindu Tech, Hacker News and others). Each article gets a Groq summary and keyword categories. An hourly scheduler runs the job.
- **3-layer story clustering.**
  1. MinHash LSH finds copied or syndicated articles.
  2. `all-MiniLM-L6-v2` embeddings plus named-entity matching and a 72h time decay group articles about the same event.
  3. A Groq verifier decides borderline pairs (results cached in `pair_cache`).
- **Cross-check and contradiction audit.** Extracts claims with verbatim quote grounding, compares numbers deterministically, and reports an agreement index across outlets (`GET /api/articles/{id}/cross-check`).
- **CareerRadar.** Extracts hiring, layoff, expansion, funding and new-office signals from article bodies, with grounding checks that drop unsupported output. Provides skill demand, a skill-gap view against the student's profile, a company board, a heatmap, a trend view, and a personalized weekly brief.
- **Student profile.** Per-user profile (`/api/profile`) used for skill-gap matching and the weekly brief.
- **AI chat.** `/chat` answers questions from the latest 150 articles.
- **Accounts.** Email/password and Google sign-in (JWT), user dashboard, admin console.

## Tech stack

| Area | Tools |
|---|---|
| Backend | FastAPI, uvicorn, SQLAlchemy 2, Pydantic v2, APScheduler, slowapi |
| NLP | sentence-transformers, spaCy `en_core_web_sm`, datasketch |
| LLM | Groq HTTP API with live model discovery and fallback |
| Database | SQLite (default), PostgreSQL/Neon via `DATABASE_URL` |
| Frontend | Jinja2 templates, inline vanilla JS, custom CSS. There is no build step. |

---

## Quick start

### Prerequisites

- Python 3.11 or newer (the project's venv currently runs 3.15)
- A free [Groq API key](https://console.groq.com/keys)
- About 1 GB of disk for the NLP dependencies

### 1. Clone and install

```bash
git clone https://github.com/ankitera1136/agentic-ai-news-platform.git
cd agentic-ai-news-platform

python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` already installs the spaCy model `en_core_web_sm`. If the entity step later complains the model is missing, run `python -m spacy download en_core_web_sm`.

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and set at least:

| Variable | Notes |
|---|---|
| `GROQ_API_KEY` | Required for summaries, chat, the cluster verifier, cross-check and CareerRadar. |
| `SECRET_KEY` | JWT signing key. Use a long random value (`python -c "import secrets; print(secrets.token_hex(32))"`). |
| `ADMIN_EMAIL` | The account with this email becomes admin on login. |
| `DATABASE_URL` | Leave as `sqlite:///./tech_news.db` for local work. Use a `postgresql://...` URL for Neon. |

Optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `CAREER_RADAR_ENABLED` | `false` | Run CareerRadar extraction at the end of each scrape job. |
| `CAREER_PREFILTER_ENABLED` | `false` | Skip articles that do not look career-related before any fetch or LLM call. |
| `CAREER_EXTRACTION_BATCH_LIMIT` | `25` | Articles processed per scrape run. |
| `CAREER_LLM_MODEL` | empty | Override the Groq model for CareerRadar only. |
| `CAREER_FETCH_DELAY_SECONDS` | `0.5` | Delay between article body fetches. |
| `CAREER_LLM_CALL_DELAY_SECONDS` | `1.0` | Delay between Groq calls in batches (protects the free-tier rate limit). |
| `GROQ_MODEL` | auto | Preferred Groq model. |
| `DEDUP_MODE` | `cluster` | Clustering mode. |
| `ARTICLE_RETENTION_DAYS` | `30` | Older articles are deleted after each scrape. |
| `PORT` | `8000` | Listen port for `python main.py`. |

The SMTP and extra API key entries in `.env.example` are placeholders. Nothing in the code sends email yet.

### 3. Start the server

```bash
python main.py
```

Open <http://localhost:8000>. On first start the app creates the database, runs the dedup and CareerRadar migrations, and seeds the categories. It then runs one full scrape in the background, which can take several minutes on first run because the embedding model downloads once. After that, the scraper runs hourly.

Equivalent command without auto-reload settings from `main.py`:

```bash
uvicorn backend.app.webapp:app --port 8000
```

Run from the repo root. The scraper reads and writes `data/summaries/` with relative paths.

### 4. Create your account

1. Go to <http://localhost:8000> and register. Use the same email as `ADMIN_EMAIL` to get admin access.
2. Fill in your student profile in the dashboard (`/dashboard`). CareerRadar uses it for the skill-gap view and weekly brief.

---

## Pages

| Path | What it shows |
|---|---|
| `/` | News feed with category filters, search, clusters ("covered by N outlets") and cross-check |
| `/career` | CareerRadar: skills in demand, your skill gap, companies, heatmap, trend, weekly brief |
| `/chat` | AI assistant |
| `/dashboard` | Preferences, saved stories, account |
| `/admin` | Admin console: stats, articles, users, logs, AI settings, manual scrape |

Interactive API docs are at `/docs`.

### Main API groups

| Prefix | Purpose |
|---|---|
| `/auth/*` | Register, login, Google sign-in, current user |
| `/api/articles/feed`, `/api/clusters/*` | Feed and clusters |
| `/api/articles/{id}/cross-check` | Contradiction audit |
| `/api/profile` | Student profile (`GET`, `PUT`, `DELETE`) |
| `/api/career/*` | `skill-demand`, `my-gap`, `company-board`, `heatmap`, `trend` |
| `/api/career/brief` | Weekly brief (`GET`, `POST /regenerate`, `GET /history`) |
| `/api/admin/*` | Admin only |
| `/api/cron/scrape` | Triggers a scrape (unauthenticated, see Deployment) |

---

## CareerRadar workflow

Extraction is off in the scheduled job by default. To fill the career tables:

```bash
# Preview what would be processed, without calling the LLM
python scripts/backfill_career_signals.py --unprocessed --limit 25 --dry-run

# Extract signals from articles that have not been processed yet
python scripts/backfill_career_signals.py --unprocessed --limit 25

# Other useful options
#   --article-id ID      one article (repeatable)
#   --source NAME        limit to a source (repeatable)
#   --force              redo already-processed articles
#   --no-fetch           use cached bodies only
```

Run `python scripts/backfill_career_signals.py --help` for the full list.

Other scripts:

```bash
# Weekly briefs (one user, or everyone)
python scripts/generate_weekly_briefs.py --user-email you@example.com
python scripts/generate_weekly_briefs.py --all --force

# Per-week, per-source extraction funnel (read-only)
python scripts/career_funnel_report.py --weeks 4

# Copy a SQLite database into PostgreSQL
python scripts/migrate_to_postgres.py
```

To extract automatically after every scrape, set `CAREER_RADAR_ENABLED=true`.

Groq free-tier limits are easy to hit in large batches. Failed articles are retried on later runs (3 attempts per content hash).

---

## Tests and evaluation

```bash
python -m pytest tests/ -q            # add --slow to include model-download tests
python eval/evaluate.py               # clustering evaluation
python eval/benchmark.py              # clustering benchmark
```

Tests mock the LLM and network, so no API key is needed. There are no frontend tests.

---

## Project layout

```
main.py                    local entry point
api/index.py               Vercel entry point
backend/app/               FastAPI app (webapp.py), models, auth, database, Groq client
backend/app/routers/       profile, career and brief routers
backend/scrapers/          RSS scraper
backend/services/          categorization, claims, contradiction engine, career/, brief/, profile/
career/, student_profile/  database migrations for CareerRadar and profile tables
dedup/                     3-layer clustering pipeline
eval/                      clustering evaluation tools
scripts/                   CLI jobs (backfill, briefs, reports, Postgres migration)
frontend/                  Jinja2 templates and static files
data/summaries/            one JSON per scraped article
tests/                     pytest suite
PROJECT_STATE.md           detailed engineering notes (may lag behind the code)
```

---

## Deployment notes

- **Local:** `python main.py`. Scheduler on, SQLite file in the repo root.
- **Render + Neon:** run the persistent backend with `DATABASE_URL` set to the Neon URL. The scheduler runs there.
- **Vercel:** `vercel.json` serves pages from `api/index.py` and proxies `/api/*`, `/auth/*` and `/ping` to the Render backend. On Vercel the scheduler is disabled and SQLite lives in `/tmp` (ephemeral).
- `GET /api/cron/scrape` has no authentication. Put it behind a secret or remove it before exposing the app publicly.
- There is no Alembic. Schema changes to existing tables need an explicit migration (see `career/migrate.py`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Empty feed after start | Wait for the first scrape to finish, or use **Trigger scrape** in `/admin`. Check `GROQ_API_KEY`. |
| `ModuleNotFoundError: backend` | Run commands from the repo root with the venv active. |
| spaCy model not found | `python -m spacy download en_core_web_sm` |
| Many Groq 429 errors | Raise `CAREER_LLM_CALL_DELAY_SECONDS` or lower `--limit`. |
| Cannot reach `/admin` features | Log in with the `ADMIN_EMAIL` account. |
| Port in use | Set `PORT=8001` in `.env`. |
