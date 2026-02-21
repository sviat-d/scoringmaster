"""Shared data models for the scoring pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Page types for content classification ──
PAGE_TYPES = (
    "homepage", "pricing", "checkout", "terms", "affiliate",
    "payouts", "api_docs", "faq", "about", "careers", "blog", "other",
)

# ── Target needs ──
NEED_A = "A"  # crypto acceptance need
NEED_B = "B"  # stablecoin treasury / conversion need
NEED_C = "C"  # mass payout need


@dataclass
class CrawledPage:
    """A single crawled page with metadata."""
    url: str
    text: str
    title: str = ""
    headings: list[str] = field(default_factory=list)
    page_type: str = "other"


@dataclass
class EvidenceItem:
    """A single piece of evidence extracted from a page."""
    signal_id: str
    matched_phrase: str
    snippet: str  # 200-400 chars around match
    url: str
    page_type: str
    base_weight: int  # 1-4
    target_need: str  # A, B, or C


@dataclass
class FlowConfirmation:
    """LLM semantic confirmation of financial flows."""
    outbound_obligations: bool = False
    outbound_confidence: float = 0.0
    inbound_payment_acceptance: bool = False
    inbound_confidence: float = 0.0
    cross_border_fx_exposure: bool = False
    cross_border_confidence: float = 0.0
    likely_mass_payouts: bool = False
    mass_payout_confidence: float = 0.0
    likely_stablecoin_treasury: bool = False
    stablecoin_confidence: float = 0.0
    reasoning_summary: str = ""
    false_positive_risks: list[str] = field(default_factory=list)


@dataclass
class SubScores:
    """Three sub-scores for the three product needs."""
    score_a: int = 0  # crypto acceptance (0-20)
    score_b: int = 0  # stablecoin treasury (0-20)
    score_c: int = 0  # mass payout (0-20)
    gate_a_passed: bool = False
    gate_b_passed: bool = False
    gate_c_passed: bool = False
    gate_a_reason: str = ""
    gate_b_reason: str = ""
    gate_c_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "score_a": self.score_a,
            "score_b": self.score_b,
            "score_c": self.score_c,
            "gate_a_passed": self.gate_a_passed,
            "gate_b_passed": self.gate_b_passed,
            "gate_c_passed": self.gate_c_passed,
        }
