"""Mode registry and base interface."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scorer.extractor import SiteSignals


class ScoringResult:
    """Standardized scoring result from any mode."""

    def __init__(
        self,
        industry: str = "Unknown",
        business_model: str = "Unknown",
        headcount_estimate: str = "Unknown",
        crypto_adoption_likelihood: str = "Low",
        risk_flags: list[str] | None = None,
        score: int = 1,
        confidence: str = "Low",
        reason_short: str = "",
        reasons_bullets: list[str] | None = None,
        opener: str = "",
        next_action: str = "",
    ):
        self.industry = industry
        self.business_model = business_model
        self.headcount_estimate = headcount_estimate
        self.crypto_adoption_likelihood = crypto_adoption_likelihood
        self.risk_flags = risk_flags or []
        self.score = max(1, min(10, score))
        self.confidence = confidence
        self.reason_short = reason_short
        self.reasons_bullets = reasons_bullets or []
        self.opener = opener
        self.next_action = next_action

    @property
    def category(self) -> str:
        if self.score >= 8:
            return "A"
        if self.score >= 6:
            return "B"
        if self.score >= 4:
            return "C"
        return "Reject"

    def to_dict(self) -> dict:
        return {
            "industry": self.industry,
            "business_model": self.business_model,
            "headcount_estimate": self.headcount_estimate,
            "crypto_adoption_likelihood": self.crypto_adoption_likelihood,
            "risk_flags": self.risk_flags,
            "score": self.score,
            "category": self.category,
            "confidence": self.confidence,
            "reason_short": self.reason_short,
            "reasons_bullets": self.reasons_bullets,
            "opener": self.opener,
            "next_action": self.next_action,
        }


class BaseMode:
    """Base class for scoring modes."""

    mode_id: str = ""
    mode_name: str = ""

    def score(self, signals: SiteSignals, domain: str) -> ScoringResult:
        raise NotImplementedError


# Mode registry
_MODES: dict[str, BaseMode] = {}


def register_mode(mode: BaseMode) -> None:
    _MODES[mode.mode_id] = mode


def get_mode(mode_id: str) -> BaseMode:
    if mode_id not in _MODES:
        raise ValueError(f"Unknown mode: {mode_id}. Available: {list(_MODES.keys())}")
    return _MODES[mode_id]
