import feedparser
import os
import sys
from datetime import datetime
import re
import requests
import json
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# Fix Windows terminal encoding so emoji in print() don't crash the process
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def extract_image_url(entry):
    # 1. Try media_content
    if 'media_content' in entry and entry.media_content:
        url = entry.media_content[0].get('url')
        if url:
            return url
    # 2. Try media_thumbnail
    if 'media_thumbnail' in entry and entry.media_thumbnail:
        url = entry.media_thumbnail[0].get('url')
        if url:
            return url
    # 3. Try enclosures
    if 'enclosures' in entry and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get('type', '').startswith('image/') and enc.get('href'):
                return enc['href']
    # 4. Try to extract from summary or content
    html_sources = []
    if hasattr(entry, 'summary'):
        html_sources.append(entry.summary)
    if hasattr(entry, 'content') and entry.content:
        html_sources.append(entry.content[0].value)
    for html in html_sources:
        match = re.search(r'<img[^>]+src="([^"]+)"', html)
        if match:
            return match.group(1)
    # 5. Fast scrape article page for og:image (2s max timeout)
    try:
        article_url = entry.link
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(article_url, headers=headers, timeout=2)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return og_image['content']
    except Exception:
        pass
    return ''

from backend.app.groq_client import call_groq
from backend.services.career.filters import career_relevance
from backend.services.excerpt import clean_excerpt

def get_llm_summary_groq(title, summary, audience="general tech audience"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Skipping LLM summarization.")
        return summary[:200] if summary else title

    prompt = (
        f"Article title: {title}\n"
        f"Article summary: {summary}\n\n"
        f"Summarize the above for a {audience} in 2-3 sentences."
    )
    payload = {
        "max_tokens": 120,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        text, model_used = call_groq(api_key, payload, timeout=12)
        return text
    except Exception as e:
        # Final fallback: strip HTML from raw summary
        print(f"[GROQ SUMMARY FALLBACK] {e}")
        clean_summary = re.sub(r'<[^>]+>', '', summary).strip()
        return clean_summary[:200] + "..." if len(clean_summary) > 200 else clean_summary

from difflib import SequenceMatcher

STOPWORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'in',
    'with', 'by', 'as', 'from', 'that', 'this', 'it', 'are', 'be', 'was', 'were', 'has',
    'had', 'have', 'but', 'not', 'if', 'then', 'so', 'do', 'does', 'did', 'can', 'will',
    'just', 'about', 'into', 'over', 'after', 'before', 'more', 'less', 'than', 'up',
    'out', 'off', 'no', 'yes', 'you', 'new', 'says', 'how', 'why', 'what', 'first',
    'here', 'now', 'its', 'itself', 'all', 'any', 'both', 'each', 'few', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'too', 'very', 'announced', 'announces', 'launches', 'unveils'
}

def clean_title_tokens(title: str) -> set:
    """Extract clean content keywords from headline"""
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', title.lower())
    return set(w for w in words if w not in STOPWORDS)

def is_duplicate_headline(new_title: str, existing_titles: list, jaccard_thresh=0.48, seq_thresh=0.62):
    """
    Check if new_title is a duplicate of any title in existing_titles.
    Uses Token Jaccard Overlap and Levenshtein Sequence Ratio.
    """
    new_tokens = clean_title_tokens(new_title)
    if not new_tokens:
        return False, None
        
    for existing_title in existing_titles:
        existing_tokens = clean_title_tokens(existing_title)
        if not existing_tokens:
            continue
            
        intersection = len(new_tokens & existing_tokens)
        union = len(new_tokens | existing_tokens)
        jaccard = intersection / union if union > 0 else 0
        seq_ratio = SequenceMatcher(None, new_title.lower(), existing_title.lower()).ratio()
        
        # Match if either strong token overlap or high sequence ratio
        if (jaccard >= jaccard_thresh and intersection >= 3) or seq_ratio >= seq_thresh:
            return True, existing_title
            
    return False, None

FEEDS = [
    # ── Global Tech News ───────────────────────────────────────────────
    {'name': 'TechCrunch',         'url': 'https://techcrunch.com/feed/'},
    {'name': 'Ars Technica',       'url': 'https://feeds.arstechnica.com/arstechnica/index'},
    {'name': 'MIT Tech Review',    'url': 'https://www.technologyreview.com/feed/'},
    {'name': 'Wired',              'url': 'https://www.wired.com/feed/rss'},

    # ── Indian IT & Startup Ecosystem ──────────────────────────────────
    {'name': 'Inc42',              'url': 'https://inc42.com/feed/'},
    {'name': 'YourStory',          'url': 'https://yourstory.com/feed'},
    # ── Career / workforce news (gated: only entries that look like hiring, layoffs, etc.) ──
    # Placed BEFORE Economic Times Tech on purpose: headline dedup keeps the first outlet seen,
    # and ET bodies are paywalled (headline-only, no signals), so readable copies must win.
    {'name': 'HRKatha',            'url': 'https://www.hrkatha.com/feed/',                      'career_only': True, 'max_summary_chars': 300},
    {'name': 'Mint',               'url': 'https://www.livemint.com/rss/companies',             'career_only': True},
    {'name': 'Entrackr',           'url': 'https://entrackr.com/rss',                           'career_only': True},
    {'name': 'Indian Express',     'url': 'https://indianexpress.com/section/technology/feed/', 'career_only': True, 'max_entries': 50},
    {'name': 'Analytics India Magazine', 'url': 'https://analyticsindiamag.com/feed/',          'career_only': True},

    {'name': 'Economic Times Tech','url': 'https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms'},
    {'name': 'MediaNama',          'url': 'https://www.medianama.com/feed/'},
    {'name': 'The Hindu Tech',     'url': 'https://www.thehindu.com/sci-tech/technology/feeder/default.rss'},

    # ── Developer & CS Learning ────────────────────────────────────────
    {'name': 'GitHub Blog',        'url': 'https://github.blog/feed/'},
    {'name': 'Stack Overflow Blog','url': 'https://stackoverflow.blog/feed/'},
    {'name': 'freeCodeCamp',       'url': 'https://www.freecodecamp.org/news/rss/'},
    {'name': 'Towards Data Science','url': 'https://towardsdatascience.com/feed'},

    # ── Community Curated ──────────────────────────────────────────────
    {'name': 'Hacker News',        'url': 'https://hnrss.org/frontpage'},
]

DEFAULT_MAX_ENTRIES = 15
MAX_ENTRIES_LIMIT = 100


def feed_max_entries(feed_info) -> int:
    """Per-feed `max_entries`: default 15, clamped to 1..100, invalid values fall back to 15."""
    value = feed_info.get('max_entries', DEFAULT_MAX_ENTRIES)
    if isinstance(value, bool):
        return DEFAULT_MAX_ENTRIES
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_ENTRIES
    return max(1, min(MAX_ENTRIES_LIMIT, value))


def feed_max_summary_chars(feed_info):
    """Per-feed `max_summary_chars`: None (default) keeps the raw RSS summary unchanged.

    A positive integer stores the summary as plain text cut at a word boundary at that
    length. Invalid values (0, negative, bool, non-numeric) fall back to None.
    """
    value = feed_info.get('max_summary_chars')
    if value is None or isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def feed_career_only(feed_info) -> bool:
    """Per-feed `career_only`: when true, only entries that look like career news are kept."""
    return bool(feed_info.get('career_only', False))


def fetch_and_save_all_sources():
    SAVE_DIR = 'data/summaries/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    total_new_articles = 0
    total_duplicates_skipped = 0

    # Also load existing titles from DB for deduplication on production
    existing_titles = []
    try:
        from backend.app.database import SessionLocal as _SL
        from backend.app.models import Article as _Art
        _db = _SL()
        try:
            from datetime import timedelta
            _cutoff = datetime.utcnow() - timedelta(hours=72)
            existing_titles = [r[0] for r in _db.query(_Art.title).filter(_Art.created_at >= _cutoff).all() if r[0]]
        finally:
            _db.close()
    except Exception:
        pass

    # Also load recent titles from disk (local dev fallback)
    if os.path.exists(SAVE_DIR):
        for fname in sorted(os.listdir(SAVE_DIR), reverse=True)[:150]:
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(SAVE_DIR, fname), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get('title') and data['title'] not in existing_titles:
                            existing_titles.append(data['title'])
                except Exception:
                    pass

    for feed_info in FEEDS:
        source_name = feed_info['name']
        feed_url = feed_info['url']
        max_entries = feed_max_entries(feed_info)
        career_only = feed_career_only(feed_info)
        max_summary_chars = feed_max_summary_chars(feed_info)
        feed_fetched = 0
        feed_gate_skipped = 0
        feed_stored = 0

        try:
            response = requests.get(feed_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"[{source_name}] Failed to fetch feed: {response.status_code}")
                continue

            feed = feedparser.parse(response.content)
            print(f"Fetched {len(feed.entries)} entries from {source_name} RSS feed.")

            feed_entries = feed.entries[:max_entries]  # default: top 15 entries per source
            feed_fetched = len(feed_entries)
            for entry in feed_entries:
                title = entry.title
                link = entry.link
                summary = entry.summary if hasattr(entry, 'summary') else ''
                published = entry.published if hasattr(entry, 'published') else (entry.updated if hasattr(entry, 'updated') else '')

                date_str = ''
                article_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        article_date = datetime(*entry.published_parsed[:6])
                        date_str = article_date.strftime('%Y-%m-%d')
                    except Exception:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')

                # FRESHNESS GATE: skip articles older than 60 days — prevents old reviews
                # and stale content from polluting the student feed.
                if article_date:
                    age_days = (datetime.now() - article_date).days
                    if age_days > 60:
                        continue

                filename_base = f"{date_str}-{slugify(source_name)}-{slugify(title)[:45]}"
                json_filepath = os.path.join(SAVE_DIR, f"{filename_base}.json")

                # Check exact file match (local dev)
                if os.path.exists(json_filepath):
                    continue

                # RELEVANCE GATE: skip off-topic content not useful for CS students
                SKIP_KEYWORDS = [
                    'review: ', 'hands-on:', 'best deals', 'sale', 'discount', 'coupon',
                    'black friday', 'gift guide', 'recipe', 'food', 'fashion', 'beauty',
                    'pickleball', 'sports', 'nfl', 'nba', 'cricket', 'football', 'rei ',
                    'litter box', 'pet ', 'hiking', 'camping', 'outdoor gear', 'headphone review',
                    'tv review', 'mattress', 'pillow', 'coffee maker', 'vacuum',
                    'best earbuds', 'best laptop deals', 'best phone deals',
                ]
                title_lower = title.lower()
                if any(kw in title_lower for kw in SKIP_KEYWORDS):
                    continue

                # CAREER GATE (opt-in per feed): runs before dedup, og:image and the LLM summary
                if career_only and not career_relevance(title, summary):
                    feed_gate_skipped += 1
                    continue

                # Check cross-publisher duplicate headline
                is_dup, matched_title = is_duplicate_headline(title, existing_titles)
                if is_dup:
                    print(f"[{source_name}] Skipped redundant story: '{title[:45]}...'")
                    total_duplicates_skipped += 1
                    continue

                image_url = extract_image_url(entry)
                llm_summary = get_llm_summary_groq(title, summary)
                time.sleep(0.5)

                # The gate and the LLM summary above used the FULL excerpt; only what we
                # store is shortened (plain text), and only for feeds that ask for it.
                stored_summary = (
                    clean_excerpt(summary, max_summary_chars) if max_summary_chars else summary
                )

                article_data = {
                    'title': title,
                    'link': link,
                    'summary': stored_summary,
                    'published': published,
                    'image_url': image_url,
                    'llm_summary': llm_summary,
                    'source': source_name
                }

                # Save JSON file (local dev backup)
                try:
                    with open(json_filepath, 'w', encoding='utf-8') as jf:
                        json.dump(article_data, jf, ensure_ascii=False, indent=2)
                except Exception as fe:
                    print(f"[{source_name}] File save warning: {fe}")

                # Write DIRECTLY to DB — works on Render (ephemeral FS) and locally
                _save_article_to_db(filename_base, article_data, published)

                existing_titles.append(title)
                total_new_articles += 1
                feed_stored += 1
        except Exception as e:
            print(f"Error scraping {source_name}: {e}")
        print(
            f"[FEED] {source_name}: fetched={feed_fetched} "
            f"skipped_by_career_gate={feed_gate_skipped} stored={feed_stored}"
        )

    print(f"Multi-source scraping complete. Added {total_new_articles} unique stories (Skipped {total_duplicates_skipped} redundant duplicates).")


def _save_article_to_db(article_id, article_data, published_str):
    """Insert a single scraped article directly into DB with automatic categorization tags."""
    try:
        from backend.app.database import SessionLocal
        from backend.app.models import Article, ArticleCategory
        from backend.services.categorization_service import categorizer

        db = SessionLocal()
        try:
            # Skip if already exists
            if db.query(Article.id).filter(Article.id == article_id).first():
                return

            # Parse publication date
            created_at_dt = datetime.now()
            if published_str:
                try:
                    from email.utils import parsedate_to_datetime
                    created_at_dt = parsedate_to_datetime(published_str).replace(tzinfo=None)
                except Exception:
                    pass

            article = Article(
                id=article_id,
                title=article_data.get('title', ''),
                link=article_data.get('link', ''),
                summary=article_data.get('summary', ''),
                llm_summary=article_data.get('llm_summary', ''),
                published=published_str,
                image_url=article_data.get('image_url', ''),
                source=article_data.get('source', 'TechNews'),
                created_at=created_at_dt
            )
            db.add(article)
            db.flush()

            # Automatically categorize the new article
            categorizations = categorizer.categorize_article(article_data, db)
            for category_id, relevance_score in categorizations:
                db.add(ArticleCategory(
                    article_id=article_id,
                    category_id=category_id,
                    relevance_score=relevance_score
                ))

            db.commit()
            print(f"[DB] Saved & Categorized ({len(categorizations)} tags): {article_data.get('title', '')[:60]}")
        except Exception as e:
            db.rollback()
            # Unique constraint = already exists, not a real error
            if 'unique' not in str(e).lower() and 'duplicate' not in str(e).lower():
                print(f"[DB] Insert warning: {e}")
        finally:
            db.close()
    except Exception as e:
        print(f"[DB] Could not connect: {e}")


def fetch_and_save_techcrunch_articles():
    fetch_and_save_all_sources()

if __name__ == "__main__":
    fetch_and_save_all_sources()