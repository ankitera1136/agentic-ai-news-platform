"""CareerRadar extraction service.

Flow per article:
1. Resolve text (cached full body, fetch+cache, or stored summary fallback).
2. Compute a content hash from the exact text sent to the LLM.
3. Skip when the same hash was already extracted successfully.
4. Call Groq once; retry once only when the response is invalid JSON.
5. Validate, ground, normalize, deduplicate, then persist atomically.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.groq_client import call_groq
from backend.app.models import Article, CareerSignal
from backend.services.career.body_fetcher import (
    BodyContext,
    ensure_article_body,
    is_vercel_runtime,
    text_for_llm,
)
from backend.services.career.grounding import (
    DropCounters,
    clamp_confidence,
    clean_job_title_skills,
    count_is_estimated,
    drop_generic_skills,
    ground_count,
    ground_location,
    ground_skills,
    verify_quote,
)
from backend.services.career.filters import FILTER_VERSION, career_relevance, roundup_reason
from backend.services.career.normalizer import normalize_company
from backend.services.career.prompt import (
    CAREER_RETRY_PROMPT,
    CAREER_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_user_prompt,
)
from backend.services.career.repository import (
    get_article,
    get_extraction,
    pending_articles,
    replace_signals,
    upsert_extraction,
)
from backend.services.career.schemas import (
    ACTIONS,
    TIMINGS,
    CareerSignalLLM,
    ValidatedCareerSignal,
)

logger = logging.getLogger("career_radar")

MIN_CONFIDENCE = 0.5
DEFAULT_LLM_CALL_DELAY_SECONDS = 1.0
# Ledger text_source used for articles skipped by the title filter (no body, no LLM call).
ROUNDUP_TEXT_SOURCE = "skipped_roundup"
# Ledger text_source for articles the career pre-filter rejected (no body, no LLM call).
NOT_CAREER_TEXT_SOURCE = "skipped_not_career"
HEADLINE_ONLY_REASON = "headline_only"
MIN_TEXT_CHARS = 30
MAX_LLM_ATTEMPTS = 3
MAX_ERROR_CHARS = 1_000
DEFAULT_BATCH_LIMIT = 25
DEFAULT_VERCEL_BATCH_LIMIT = 3

# Bump when prompt or validation semantics change so previously extracted articles
# are re-processed instead of being treated as unchanged.
EXTRACTION_VERSION = "career-v2"


@dataclass
class ArticleOutcome:
    article_id: str
    title: str
    status: str  # succeeded | failed | skipped | skipped_roundup | skipped_not_career | no_body | not_found
    signals_count: int = 0
    text_source: Optional[str] = None
    body_length: int = 0
    error: Optional[str] = None
    unknown_companies: List[str] = field(default_factory=list)
    dropped: DropCounters = field(default_factory=DropCounters)


@dataclass
class BatchStats:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    zero_signal: int = 0
    skipped: int = 0
    skipped_roundup: int = 0
    skipped_not_career: int = 0
    headline_only: int = 0
    no_body: int = 0
    not_found: int = 0
    dropped: DropCounters = field(default_factory=DropCounters)
    unknown_companies: List[str] = field(default_factory=list)
    outcomes: List[ArticleOutcome] = field(default_factory=list)

    def record(self, outcome: ArticleOutcome) -> None:
        if outcome.status == "succeeded":
            self.processed += 1
            self.succeeded += 1
            if outcome.signals_count == 0:
                self.zero_signal += 1
            if outcome.error == HEADLINE_ONLY_REASON:
                self.headline_only += 1
        elif outcome.status == "failed":
            self.processed += 1
            self.failed += 1
        elif outcome.status == "skipped":
            self.skipped += 1
        elif outcome.status == "skipped_roundup":
            self.skipped_roundup += 1
        elif outcome.status == "skipped_not_career":
            self.skipped_not_career += 1
        elif outcome.status == "no_body":
            self.no_body += 1
        elif outcome.status == "not_found":
            self.not_found += 1

        self.dropped.merge(outcome.dropped)
        for company in outcome.unknown_companies:
            if company not in self.unknown_companies:
                self.unknown_companies.append(company)
        self.outcomes.append(outcome)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "zero_signal": self.zero_signal,
            "skipped": self.skipped,
            "skipped_roundup": self.skipped_roundup,
            "skipped_not_career": self.skipped_not_career,
            "headline_only": self.headline_only,
            "no_body": self.no_body,
            "not_found": self.not_found,
            "dropped": self.dropped.as_dict(),
            "unknown_companies": sorted(set(self.unknown_companies)),
        }


def prefilter_enabled() -> bool:
    """CAREER_PREFILTER_ENABLED, read at call time. Default: off."""
    return os.getenv("CAREER_PREFILTER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def default_batch_limit() -> int:
    """Vercel gets a deliberately small batch; the scheduler keeps the larger one."""
    if is_vercel_runtime():
        return _int_env("CAREER_EXTRACTION_BATCH_LIMIT_VERCEL", DEFAULT_VERCEL_BATCH_LIMIT)
    return _int_env("CAREER_EXTRACTION_BATCH_LIMIT", DEFAULT_BATCH_LIMIT)


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return max(1, value)
    except (TypeError, ValueError):
        return default


def compute_content_hash(text_source: str, llm_text: str) -> str:
    payload = f"{EXTRACTION_VERSION}\n{PROMPT_VERSION}\n{text_source}\n{llm_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strip_think(text: str) -> str:
    return re.sub(r" thinking.*?</think>", "", text or "", flags=re.DOTALL)


def parse_json_array(raw: Optional[str]) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Recover a JSON list using the same layered approach as claim_extractor."""
    if raw is None:
        return None, "empty_response"
    text = _strip_think(str(raw)).strip()
    if not text:
        return None, "empty_response"

    candidates: List[str] = [text]

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidates.append(array_match.group(0))

    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        coerced = _coerce_signal_list(parsed)
        if coerced is not None:
            return coerced, None

    return None, "invalid_json"


def _coerce_signal_list(parsed: Any) -> Optional[List[Any]]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("signals", "career_signals", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        if any(key in parsed for key in ("company", "action")):
            return [parsed]
    return None


class CareerExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fetch_delay_seconds: Optional[float] = None,
        llm_call_delay_seconds: Optional[float] = None,
    ):
        env_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or ""
        self.api_key = api_key if api_key is not None else env_key
        self.model = model or os.getenv("CAREER_LLM_MODEL") or None
        if fetch_delay_seconds is None:
            try:
                fetch_delay_seconds = float(os.getenv("CAREER_FETCH_DELAY_SECONDS", "0.5"))
            except ValueError:
                fetch_delay_seconds = 0.5
        self.fetch_delay_seconds = max(0.0, fetch_delay_seconds)
        if llm_call_delay_seconds is None:
            try:
                llm_call_delay_seconds = float(
                    os.getenv("CAREER_LLM_CALL_DELAY_SECONDS", str(DEFAULT_LLM_CALL_DELAY_SECONDS))
                )
            except ValueError:
                llm_call_delay_seconds = DEFAULT_LLM_CALL_DELAY_SECONDS
        self.llm_call_delay_seconds = max(0.0, llm_call_delay_seconds)
        self._last_llm_call: Optional[float] = None
        self._warned_models: set = set()

    # -- public API ---------------------------------------------------------

    def process_pending_articles(
        self,
        db: Optional[Session] = None,
        limit: Optional[int] = None,
        allow_fetch: Optional[bool] = None,
        fetch_session=None,
    ) -> BatchStats:
        owns_session = db is None
        session = db or SessionLocal()
        if allow_fetch is None:
            allow_fetch = not is_vercel_runtime()
        batch_limit = limit or default_batch_limit()
        cached_only = not allow_fetch

        stats = BatchStats()
        offset = 0
        window = max(batch_limit * 5, 25)
        processed_attempts = 0

        try:
            while processed_attempts < batch_limit:
                candidates = pending_articles(session, window, cached_only=cached_only, offset=offset)
                if not candidates:
                    break

                for article in candidates:
                    outcome = self._process_article_obj(
                        session,
                        article,
                        allow_fetch=allow_fetch,
                        fetch_session=fetch_session,
                    )
                    stats.record(outcome)
                    if outcome.status in ("succeeded", "failed"):
                        processed_attempts += 1
                    if processed_attempts >= batch_limit:
                        break

                if len(candidates) < window:
                    break
                offset += window

            logger.info(
                "CareerRadar batch complete: processed=%s succeeded=%s failed=%s "
                "zero_signal=%s skipped=%s no_body=%s",
                stats.processed, stats.succeeded, stats.failed,
                stats.zero_signal, stats.skipped, stats.no_body,
            )
            return stats
        finally:
            if owns_session:
                session.close()

    def process_article(
        self,
        article_id: str,
        db: Optional[Session] = None,
        allow_fetch: Optional[bool] = None,
        fetch_session=None,
    ) -> ArticleOutcome:
        owns_session = db is None
        session = db or SessionLocal()
        if allow_fetch is None:
            allow_fetch = not is_vercel_runtime()
        try:
            article = get_article(session, article_id)
            if article is None:
                return ArticleOutcome(article_id, "", "not_found", error="article_not_found")
            return self._process_article_obj(
                session,
                article,
                allow_fetch=allow_fetch,
                fetch_session=fetch_session,
            )
        finally:
            if owns_session:
                session.close()

    # -- per-article flow ---------------------------------------------------

    def _process_article_obj(
        self,
        db: Session,
        article: Article,
        allow_fetch: bool,
        fetch_session=None,
    ) -> ArticleOutcome:
        title = article.title or ""
        if article.is_copy_of:
            return ArticleOutcome(article.id, title, "skipped", error="copy_article")

        # Title filter: runs before any body fetch or LLM call.
        reason = roundup_reason(title)
        if reason:
            return self._skip_roundup(db, article, reason)

        # Career pre-filter (opt-in): also runs before any body fetch or LLM call.
        if prefilter_enabled():
            skipped = self._prefilter_skip(db, article)
            if skipped is not None:
                return skipped

        context = ensure_article_body(
            db,
            article,
            allow_fetch=allow_fetch,
            delay_seconds=self.fetch_delay_seconds,
            fetch_session=fetch_session,
        )
        if context.text_source == "none" or not (context.text or "").strip():
            return ArticleOutcome(
                article.id, title, "no_body", text_source=context.text_source,
                body_length=context.body_length, error="body_not_cached",
            )

        llm_text = text_for_llm(context.text)
        content_hash = compute_content_hash(context.text_source, llm_text)

        existing = get_extraction(db, article.id)
        if existing and existing.status == "succeeded" and existing.content_hash == content_hash:
            return ArticleOutcome(
                article.id, title, "skipped", signals_count=existing.signals_count or 0,
                text_source=context.text_source, body_length=context.body_length,
                error="unchanged",
            )
        if (
            existing
            and existing.status == "failed"
            and existing.content_hash == content_hash
            and (existing.attempts or 0) >= MAX_LLM_ATTEMPTS
        ):
            return ArticleOutcome(
                article.id, title, "skipped", text_source=context.text_source,
                body_length=context.body_length, error="retry_limit_reached",
            )

        prior_attempts = 0
        if existing and existing.content_hash == content_hash:
            prior_attempts = existing.attempts or 0

        if context.text_source == "stored_summary":
            # A headline/summary is too thin to ground a workforce claim. Keep a ledger
            # row (so the article is not retried) but emit nothing and make no LLM call.
            return self._persist_zero_signal(
                db, article, content_hash, context, prior_attempts + 1,
                reason=HEADLINE_ONLY_REASON,
            )

        if len(llm_text.strip()) < MIN_TEXT_CHARS:
            return self._persist_zero_signal(
                db, article, content_hash, context, prior_attempts + 1,
                reason="text_too_short",
            )

        if not self.api_key:
            return self._mark_failed(
                db, article, content_hash, context, prior_attempts + 1,
                "missing_api_key", None,
            )

        messages = [
            {"role": "system", "content": CAREER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    title=article.title or "",
                    source=article.source or "",
                    published=str(article.published or ""),
                    text=llm_text,
                ),
            },
        ]

        raw, model_used, error = self._call_llm(messages)
        if error:
            return self._mark_failed(
                db, article, content_hash, context, prior_attempts + 1, error, model_used,
            )

        parsed, parse_error = parse_json_array(raw)
        if parsed is None:
            retry_messages = list(messages)
            if raw:
                retry_messages.append({"role": "assistant", "content": str(raw)[:2_000]})
            retry_messages.append({"role": "user", "content": CAREER_RETRY_PROMPT})
            raw_retry, retry_model, retry_error = self._call_llm(retry_messages)
            model_used = retry_model or model_used
            if retry_error:
                return self._mark_failed(
                    db, article, content_hash, context, prior_attempts + 1, retry_error, model_used,
                )
            parsed, parse_error = parse_json_array(raw_retry)

        if parsed is None:
            return self._mark_failed(
                db, article, content_hash, context, prior_attempts + 1,
                f"invalid_json_after_retry:{parse_error}", model_used,
            )

        counters = DropCounters()
        unknown_companies: List[str] = []
        validated = self._validate_signals(
            parsed, llm_text, context.text_source, counters, unknown_companies,
        )

        if unknown_companies:
            logger.warning(
                "CareerRadar unknown companies in article %s: %s",
                article.id, ", ".join(sorted(set(unknown_companies))),
            )
        if any(value for key, value in counters.as_dict().items() if key != "verified"):
            logger.info(
                "CareerRadar grounding drops for article %s: %s",
                article.id, {k: v for k, v in counters.as_dict().items() if v},
            )

        rows = [
            CareerSignal(
                source_article_id=article.id,
                company=signal.company,
                company_raw=signal.company_raw,
                action=signal.action,
                timing=signal.timing,
                count=signal.count,
                location_city=signal.location_city,
                location_country=signal.location_country,
                skills_mentioned=signal.skills_mentioned or [],
                confidence=signal.confidence,
                evidence_quote=signal.evidence_quote,
            )
            for signal in validated
        ]

        try:
            replace_signals(db, article.id, rows)
            upsert_extraction(
                db,
                article.id,
                content_hash=content_hash,
                text_source=context.text_source,
                status="succeeded",
                attempts=prior_attempts + 1,
                model_used=model_used,
                signals_count=len(rows),
                error=None,
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("CareerRadar persistence failed for article %s", article.id)
            return ArticleOutcome(
                article.id, title, "failed", text_source=context.text_source,
                body_length=context.body_length,
                error=f"persistence_error:{type(exc).__name__}",
                unknown_companies=sorted(set(unknown_companies)),
                dropped=counters,
            )

        return ArticleOutcome(
            article.id,
            title,
            "succeeded",
            signals_count=len(rows),
            text_source=context.text_source,
            body_length=context.body_length,
            unknown_companies=sorted(set(unknown_companies)),
            dropped=counters,
        )

    # -- LLM ----------------------------------------------------------------

    def _wait_between_llm_calls(self) -> None:
        """Space out Groq calls so batches stay under the free-tier rate limit."""
        if self._last_llm_call is not None and self.llm_call_delay_seconds > 0:
            remaining = self.llm_call_delay_seconds - (time.monotonic() - self._last_llm_call)
            if remaining > 0:
                time.sleep(remaining)

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        self._wait_between_llm_calls()
        self._last_llm_call = time.monotonic()
        payload = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1_000,
        }
        try:
            result = call_groq(
                self.api_key,
                payload,
                timeout=30,
                preferred_model=self.model,
            )
            if isinstance(result, tuple):
                text = result[0] if len(result) > 0 else None
                model_used = result[1] if len(result) > 1 else None
            else:
                text = result
                model_used = None
            self._note_model_fallback(model_used)
            if not text:
                return None, model_used, "empty_llm_response"
            return text, model_used, None
        except Exception as exc:
            logger.exception("CareerRadar LLM call failed")
            return None, None, f"llm_error:{type(exc).__name__}: {str(exc)[:300]}"

    def _note_model_fallback(self, model_used: Optional[str]) -> None:
        """Warn once when the configured model was rejected, then stop requesting it.

        call_groq() already skips a rejected model and moves down the fallback chain.
        Dropping the configured name afterwards means later calls in this run go
        straight to the model that works instead of probing the bad one every time.
        """
        if not self.model or not model_used or model_used == self.model:
            return
        if self.model not in self._warned_models:
            self._warned_models.add(self.model)
            logger.warning(
                "CareerRadar: configured model %r was rejected by Groq; using %r instead. "
                "Update or blank CAREER_LLM_MODEL.",
                self.model, model_used,
            )
        self.model = None

    # -- validation ---------------------------------------------------------

    def _validate_signals(
        self,
        items: List[Any],
        source_text: str,
        text_source: str,
        counters: DropCounters,
        unknown_companies: List[str],
    ) -> List[ValidatedCareerSignal]:
        validated: Dict[tuple, ValidatedCareerSignal] = {}

        for item in items:
            if not isinstance(item, dict):
                counters.add("invalid_item")
                continue
            try:
                raw = CareerSignalLLM.model_validate(item)
            except ValidationError:
                counters.add("invalid_item")
                continue

            action = (raw.action or "").strip().upper().replace(" ", "_").replace("-", "_")
            if action not in ACTIONS:
                counters.add("invalid_action")
                continue

            timing = (raw.timing or "current").strip().lower()
            if timing not in TIMINGS:
                counters.add("invalid_timing")
                continue
            if timing == "past_context":
                # Historical background, biography, anecdotes, and opinion context
                # are not career news. Count them so prompt tuning is measurable.
                counters.add("past_context")
                continue

            if action == "OTHER":
                # Kept in ACTIONS so old rows stay valid, but never stored from new runs.
                counters.add("other_action")
                continue

            company = normalize_company(raw.company)
            if not company.canonical:
                counters.add("missing_company")
                continue
            if not company.matched_alias:
                counters.add("unknown_company")
                unknown_companies.append(company.canonical)

            confidence = clamp_confidence(raw.confidence)
            if confidence is None:
                counters.add("invalid_confidence")
                continue
            if confidence < MIN_CONFIDENCE:
                counters.add("low_confidence")
                continue

            quote = (raw.evidence_quote or "").strip()
            if not quote or not verify_quote(quote, source_text):
                counters.add("evidence_quote")
                continue

            count, count_reason = ground_count(raw.count, source_text, quote)
            if count is not None and count_is_estimated(count, quote):
                # Deterministic: an estimated number is not a stated count.
                count = None
                counters.add("estimated_count")
            if count_reason == "percentage":
                counters.add("count_percentage")
            elif count_reason:
                counters.add("count_invalid" if count_reason in ("invalid", "not_people") else "count_not_grounded")

            city = raw.location.city if raw.location else None
            country = raw.location.country if raw.location else None
            city, country, location_dropped = ground_location(city, country, source_text)
            if location_dropped:
                counters.add("location_not_grounded")

            title_free_skills, title_drops = clean_job_title_skills(raw.skills_mentioned)
            if title_drops:
                counters.add("job_title_skill", title_drops)
            specific_skills, generic_drops = drop_generic_skills(title_free_skills)
            if generic_drops:
                counters.add("generic_skill", generic_drops)
            skills, dropped_skills = ground_skills(specific_skills, source_text)
            if action in ("LAYOFF", "HIRING_FREEZE"):
                # Skill demand ignores these actions; a layoff article's skills are noise.
                skills, dropped_skills = [], []
            if dropped_skills:
                counters.add("skill_not_grounded", len(dropped_skills))

            signal = ValidatedCareerSignal(
                company=company.canonical,
                company_raw=company.raw,
                action=action,
                timing=timing,
                count=count,
                location_city=city,
                location_country=country,
                skills_mentioned=skills,
                confidence=round(confidence, 3),
                evidence_quote=quote,
            )

            key = signal.dedupe_key()
            if key in validated:
                counters.add("duplicate")
                existing = validated[key]
                existing.skills_mentioned = sorted(
                    set(existing.skills_mentioned) | set(signal.skills_mentioned)
                )
                existing.confidence = max(existing.confidence, signal.confidence)
                if len(signal.evidence_quote) > len(existing.evidence_quote):
                    existing.evidence_quote = signal.evidence_quote
                continue

            validated[key] = signal
            counters.add("verified")

        return list(validated.values())

    # -- persistence --------------------------------------------------------

    def _skip_roundup(self, db: Session, article: Article, reason: str) -> ArticleOutcome:
        """Record a roundup-style article as done with zero signals, without fetching or calling the LLM."""
        title = article.title or ""
        content_hash = compute_content_hash(ROUNDUP_TEXT_SOURCE, title)
        existing = get_extraction(db, article.id)
        already_recorded = bool(
            existing
            and existing.status == "succeeded"
            and existing.content_hash == content_hash
            and (existing.signals_count or 0) == 0
        )
        if not already_recorded:
            try:
                replace_signals(db, article.id, [])
                upsert_extraction(
                    db,
                    article.id,
                    content_hash=content_hash,
                    text_source=ROUNDUP_TEXT_SOURCE,
                    status="succeeded",
                    attempts=(existing.attempts or 0) + 1 if existing else 1,
                    model_used=None,
                    signals_count=0,
                    error=None,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.exception("CareerRadar roundup persistence failed for article %s", article.id)
                return ArticleOutcome(
                    article.id, title, "failed",
                    error=f"persistence_error:{type(exc).__name__}",
                )
        return ArticleOutcome(
            article.id, title, "skipped_roundup",
            text_source=ROUNDUP_TEXT_SOURCE, error=f"roundup:{reason}",
        )

    def _prefilter_skip(self, db: Session, article: Article) -> Optional[ArticleOutcome]:
        """Return a skipped_not_career outcome when the article is not career news, else None.

        The ledger hash includes FILTER_VERSION, so bumping the version re-opens these
        rows. Articles that already have a real extraction result are left untouched.
        """
        title = article.title or ""
        if career_relevance(title, article.summary):
            return None

        existing = get_extraction(db, article.id)
        if (
            existing
            and existing.status == "succeeded"
            and existing.text_source != NOT_CAREER_TEXT_SOURCE
        ):
            # Already extracted (or deliberately recorded) by the normal path: keep it.
            return ArticleOutcome(article.id, title, "skipped", error="already_processed")

        content_hash = compute_content_hash(
            NOT_CAREER_TEXT_SOURCE, f"{FILTER_VERSION}\n{title}\n{article.summary or ''}"
        )
        already_recorded = bool(
            existing
            and existing.status == "succeeded"
            and existing.content_hash == content_hash
        )
        if not already_recorded:
            try:
                upsert_extraction(
                    db,
                    article.id,
                    content_hash=content_hash,
                    text_source=NOT_CAREER_TEXT_SOURCE,
                    status="succeeded",
                    attempts=(existing.attempts or 0) + 1 if existing else 1,
                    model_used=None,
                    signals_count=0,
                    error=None,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.exception("CareerRadar pre-filter persistence failed for article %s", article.id)
                return ArticleOutcome(
                    article.id, title, "failed",
                    error=f"persistence_error:{type(exc).__name__}",
                )
        return ArticleOutcome(
            article.id, title, "skipped_not_career",
            text_source=NOT_CAREER_TEXT_SOURCE, error="not_career",
        )

    def _persist_zero_signal(
        self,
        db: Session,
        article: Article,
        content_hash: str,
        context: BodyContext,
        attempts: int,
        reason: str,
    ) -> ArticleOutcome:
        try:
            replace_signals(db, article.id, [])
            upsert_extraction(
                db,
                article.id,
                content_hash=content_hash,
                text_source=context.text_source,
                status="succeeded",
                attempts=attempts,
                model_used=None,
                signals_count=0,
                error=None,
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("CareerRadar zero-signal persistence failed for article %s", article.id)
            return ArticleOutcome(
                article.id, article.title or "", "failed",
                text_source=context.text_source, body_length=context.body_length,
                error=f"persistence_error:{type(exc).__name__}",
            )
        return ArticleOutcome(
            article.id, article.title or "", "succeeded", signals_count=0,
            text_source=context.text_source, body_length=context.body_length,
            error=reason,
        )

    def _mark_failed(
        self,
        db: Session,
        article: Article,
        content_hash: str,
        context: BodyContext,
        attempts: int,
        error: str,
        model_used: Optional[str],
    ) -> ArticleOutcome:
        try:
            upsert_extraction(
                db,
                article.id,
                content_hash=content_hash,
                text_source=context.text_source,
                status="failed",
                attempts=attempts,
                model_used=model_used,
                signals_count=0,
                error=(error or "")[:MAX_ERROR_CHARS],
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("CareerRadar failure persistence failed for article %s", article.id)
            error = f"persistence_error:{type(exc).__name__}"

        return ArticleOutcome(
            article.id, article.title or "", "failed",
            text_source=context.text_source, body_length=context.body_length,
            error=(error or "")[:MAX_ERROR_CHARS],
        )


# Shared instance used by the scheduler hook and the backfill script.
career_extractor = CareerExtractor()
