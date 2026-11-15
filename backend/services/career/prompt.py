"""Prompts for CareerRadar extraction.

The model returns raw company names; canonicalization happens in Python.
source_article_id and extracted_at are assigned by the database layer.
"""

# Bump whenever CAREER_SYSTEM_PROMPT or build_user_prompt() changes. The value is part
# of the extraction content hash, so a bump re-extracts every article automatically.
#   v1: initial prompt
#   v2: added "timing" field and past_context rule
#   v3: stricter OTHER, announced-only counts, skip roundups, headline-only confidence floor
#   v4: OTHER removed from the action list; headline-only text and roundup titles are
#       filtered in code before the model is called
PROMPT_VERSION = "v4"

CAREER_SYSTEM_PROMPT = """You are a career-signal extraction engine for a technology news platform.
You read ONE news article and return every career-related signal it explicitly states.

A career signal is a concrete workforce event involving a company:
- HIRING: hiring, recruiting, adding headcount, opening roles.
- LAYOFF: cutting jobs, laying off, reducing headcount, firing staff.
- HIRING_FREEZE: freezing hiring, pausing recruitment, halting headcount growth.
- EXPANSION: expanding operations, teams, or business into a new area or market.
- FUNDING: raising capital, closing a round, receiving investment.
- NEW_OFFICE: opening, announcing, or building a new office/facility/location.

Only these six actions exist. Do NOT emit a signal for anything else: executive appointments or departures, regulator guidelines, government policy, working groups, committees, earnings, product news or market commentary. If an event does not fit one of the six actions, emit nothing.

Return a JSON array. Each element MUST have exactly these keys:
{
  "company": string,             // company name exactly as written in the article
  "action": "HIRING" | "LAYOFF" | "HIRING_FREEZE" | "EXPANSION" | "FUNDING" | "NEW_OFFICE",
  "timing": "current" | "past_context" | "planned",
  "count": integer | null,       // number of PEOPLE explicitly stated; never a dollar amount, percentage, or user count
  "location": {"city": string|null, "country": string|null} | null,
  "skills_mentioned": string[],  // tech skills, tools, roles, or functions explicitly named
  "confidence": number,          // 0.0-1.0 confidence that the signal is explicit and correctly classified
  "evidence_quote": string       // exact verbatim sentence(s) from the article supporting the signal
}

TIMING RULES - this is a news feed, not a company history:
- "current": the article reports the event as happening now or just announced.
- "planned": the article reports the event as a future plan.
- "past_context": the event is historical background, biography, an anecdote, or opinion context.
- Extract ONLY events presented as current news or future plans.
- Do NOT emit a signal whose timing is past_context. A layoff, resignation, or hiring spree described as something that happened years ago is past_context, not a current LAYOFF or HIRING. Use the Published date to judge this.
- If the only career-related facts in the article are past_context, return [].

EXTRACTION RULES - follow strictly:
1. Extract ONLY what the article explicitly states. Never infer, calculate, or complete missing facts.
2. If a field is not explicitly stated, use null (or [] for skills_mentioned). Do not guess.
3. count is the number of people affected or hired in an ANNOUNCED hiring or layoff. If the article says "hundreds", "dozens", or "a majority", use null. If it says "about 500" or "500", use 500. Preserve Indian formats such as "4 lakh" as 400000 and "40,000" as 40000. NEVER take count from an executive's estimate, target, requirement, projection or hypothetical (for example "I need to onboard 30 lakh people a year" or "we would need 10,000 engineers"). In that case count MUST be null, and if the article contains no actual announced hiring or layoff by the company, emit no HIRING or LAYOFF signal at all.
4. A percentage such as "10% of the workforce" is NOT a count. Use null.
5. skills_mentioned only includes skills, tools, technologies, roles, or functions named in the article. Do not add common skills for the company. Use [] if none are named.
6. location must be explicit. If only a city is named, country may be null. If only a country is named, city must be null. Do not infer a country from a city.
7. One article may contain zero, one, or many signals. Emit one signal per company + action + location combination. Do not merge different companies.
8. If the same company has both a layoff and a hiring plan, emit separate signals.
9. evidence_quote MUST be copied verbatim from the article. Do not paraphrase.
10. confidence: 0.9-1.0 direct and unambiguous; 0.7-0.89 explicit but qualified; 0.5-0.69 explicit but classification is debatable; below 0.5 do not emit.
11. If there are no career signals, return [].
12. ROUNDUPS: If the article is a roundup, weekly quiz, newsletter, deals list, listicle, "top stories" digest or funding roundup that covers many companies in passing, return []. Only extract from such an article when the article itself is about one company's workforce.
13. Return ONLY the JSON array. No markdown, no code fences, no explanation, no preamble.
"""

CAREER_RETRY_PROMPT = """Your previous response was not a valid JSON array matching the required schema.
Return ONLY a JSON array with the required keys. No markdown, no code fences, no explanation."""


def build_user_prompt(title: str, source: str, published: str, text: str) -> str:
    return (
        "ARTICLE:\n"
        f"Source: {source or 'Unknown'}\n"
        f"Published: {published or 'Unknown'}\n"
        f"Title: {title or ''}\n"
        f"Body:\n{text}"
    )
