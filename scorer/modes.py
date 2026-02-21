"""Mode registry and base interface."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scorer.extractor import SiteSignals
    from scorer.models import SubScores, FlowConfirmation


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
        use_cases: list[str] | None = None,
        sub_scores: SubScores | None = None,
        flow_confirmation: FlowConfirmation | None = None,
        evidence_summary: list[dict] | None = None,
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
        self.use_cases = use_cases or []
        self.sub_scores = sub_scores
        self.flow_confirmation = flow_confirmation
        self.evidence_summary = evidence_summary or []

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
        result = {
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
            "use_cases": self.use_cases,
            "evidence_summary": self.evidence_summary,
        }
        if self.sub_scores:
            result["sub_scores"] = self.sub_scores.to_dict()
        else:
            result["sub_scores"] = {"score_a": 0, "score_b": 0, "score_c": 0,
                                    "gate_a_passed": False, "gate_b_passed": False, "gate_c_passed": False}
        if self.flow_confirmation:
            result["flow_confirmation"] = {
                "reasoning": self.flow_confirmation.reasoning_summary,
                "false_positive_risks": self.flow_confirmation.false_positive_risks,
            }
        return result


class BaseMode:
    """Base class for scoring modes."""

    mode_id: str = ""
    mode_name: str = ""

    def score(self, signals: SiteSignals, domain: str, classification: dict | None = None) -> ScoringResult:
        raise NotImplementedError


# Mode registry
_MODES: dict[str, BaseMode] = {}


def register_mode(mode: BaseMode) -> None:
    _MODES[mode.mode_id] = mode


def get_mode(mode_id: str) -> BaseMode:
    if mode_id not in _MODES:
        raise ValueError(f"Unknown mode: {mode_id}. Available: {list(_MODES.keys())}")
    return _MODES[mode_id]
