"""Pydantic schemas for the raw LLM response and validated signals."""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ACTIONS = (
    "HIRING",
    "LAYOFF",
    "HIRING_FREEZE",
    "EXPANSION",
    "FUNDING",
    "NEW_OFFICE",
    "OTHER",
)

TIMINGS = ("current", "past_context", "planned")


class CareerLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: Optional[str] = None
    country: Optional[str] = None

    @field_validator("city", "country", mode="before")
    @classmethod
    def _clean(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class CareerSignalLLM(BaseModel):
    """Shape the model is asked to return. Validation is intentionally strict."""

    model_config = ConfigDict(extra="ignore")

    company: str
    action: str
    timing: str = "current"
    count: Any = None
    location: Optional[CareerLocation] = None
    skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float
    evidence_quote: str = ""

    @field_validator("company", "action", mode="before")
    @classmethod
    def _required_text(cls, value):
        return str(value).strip() if value is not None else ""

    @field_validator("timing", mode="before")
    @classmethod
    def _timing_text(cls, value):
        if value is None:
            return "current"
        return str(value).strip().lower().replace("-", "_").replace(" ", "_") or "current"

    @field_validator("skills_mentioned", mode="before")
    @classmethod
    def _skills_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("evidence_quote", mode="before")
    @classmethod
    def _quote_text(cls, value):
        return str(value).strip() if value is not None else ""


class ValidatedCareerSignal(BaseModel):
    """A signal that passed every grounding check, ready for persistence."""

    company: str
    company_raw: str
    action: str
    timing: str = "current"
    count: Optional[int] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float
    evidence_quote: str = ""

    def dedupe_key(self) -> tuple:
        return (
            self.company.lower(),
            self.action,
            self.timing,
            self.count,
            (self.location_city or "").lower(),
            (self.location_country or "").lower(),
        )
