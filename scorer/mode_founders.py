"""Founders PL scoring mode — secondary mode for product IT founders community."""

from scorer.extractor import SiteSignals
from scorer.modes import BaseMode, ScoringResult, register_mode
from scorer import llm

# These indicate service companies → reject
SERVICE_KEYWORDS = [
    "agency", "outsourcing", "outsource", "studio", "consulting",
    "custom development", "custom dev", "software development services",
    "web development services", "design agency", "marketing agency",
    "digital agency", "it services", "staff augmentation",
    "dedicated team", "hire developers",
]

# Product indicators → allow
PRODUCT_KEYWORDS = [
    "saas", "platform", "our product", "our app", "mobile app",
    "software product", "hardware", "embedded", "iot",
    "download", "sign up", "get started", "free trial",
    "pricing", "plans", "subscription",
]

# LLM business types that indicate product companies
PRODUCT_BUSINESS_TYPES = {"product", "marketplace"}
SERVICE_BUSINESS_TYPES = {"agency", "service"}
NON_TARGET_TYPES = {"media", "other"}


class FoundersPLMode(BaseMode):
    mode_id = "founders_pl"
    mode_name = "Founders PL (Product IT Founders)"

    def score(self, signals: SiteSignals, domain: str, classification: dict | None = None) -> ScoringResult:
        # Hard reject
        if signals.hard_reject:
            return ScoringResult(
                industry=signals.top_industry,
                score=1,
                reason_short=f"Hard reject: {signals.hard_reject_reason}",
                reasons_bullets=["Detected hard-reject signal"],
                confidence="High",
            )

        if signals.non_business and not signals.industries:
            return ScoringResult(
                industry="Non-business",
                score=1,
                reason_short="Non-business website",
                reasons_bullets=["Not a business website"],
                confidence="Med",
            )

        if not signals.industries and signals.raw_text_length < 200 and not classification:
            return ScoringResult(
                score=1,
                reason_short="Could not extract meaningful content",
                reasons_bullets=["Website empty or inaccessible"],
                confidence="Low",
            )

        if classification:
            return self._score_with_classification(classification, signals, domain)
        else:
            return self._score_keywords_only(signals, domain)

    def _score_with_classification(
        self, classification: dict, signals: SiteSignals, domain: str
    ) -> ScoringResult:
        """Score using LLM classification for Founders PL mode."""
        llm_industry_code = classification.get("industry", "unknown")
        llm_industry_name = llm.map_industry_code(llm_industry_code)
        business_type = classification.get("business_type", "unknown")
        is_content_site = classification.get("is_content_site", False)
        llm_confidence = classification.get("confidence", "medium")
        primary_business = classification.get("primary_business", "")

        reasons: list[str] = []
        score = 5  # baseline for founders
        confidence = "Med"

        # ── Content site / non-target → reject ──
        if is_content_site or business_type in NON_TARGET_TYPES:
            return ScoringResult(
                industry=llm_industry_name,
                business_model=business_type,
                headcount_estimate=signals.headcount_estimate,
                score=1,
                confidence="High" if llm_confidence == "high" else "Med",
                reason_short=f"Non-target: {llm_industry_name} ({business_type})",
                reasons_bullets=[f"LLM: {primary_business}", f"Business type: {business_type}"],
                next_action="Skip",
            )

        # ── Service/agency → reject for Founders PL ──
        if business_type in SERVICE_BUSINESS_TYPES:
            return ScoringResult(
                industry=llm_industry_name,
                business_model="Service",
                headcount_estimate=signals.headcount_estimate,
                score=2,
                confidence="High" if llm_confidence == "high" else "Med",
                reason_short="Service/agency company — not a fit for Founders PL",
                reasons_bullets=[
                    f"LLM: {primary_business}",
                    "Service/agency/consulting company — reject for Founders PL",
                ],
                next_action="Skip",
            )

        # ── Product company → good fit ──
        is_product = business_type in PRODUCT_BUSINESS_TYPES
        if is_product:
            score += 2
            reasons.append(f"LLM: product company — {primary_business}")
        else:
            reasons.append(f"LLM: {primary_business} (type: {business_type})")

        # Industry context
        reasons.append(f"Industry: {llm_industry_name}")

        # Headcount rules
        headcount = signals.headcount_estimate
        if headcount == "Unknown":
            score = max(score - 1, 4)
            confidence = "Low"
            reasons.append("Headcount unknown — manual check if CEO lead")
        elif headcount in ("1-5",):
            score = max(score - 1, 3)
            reasons.append("Very small team (1-5) — verify product stage")
        elif headcount == "10+":
            score += 1
            reasons.append("Team 10+ — established company")

        # API signals suggest product company
        if signals.operational_signals.get("api_integrations", 0) > 0:
            if not is_product:
                score += 1
                reasons.append("API/integration signals suggest product company")

        # Active hiring
        if signals.has_careers_page:
            score += 1
            reasons.append("Active hiring — growing product team")

        # LLM confidence boosts our confidence
        if llm_confidence == "high" and is_product:
            confidence = "High"
        elif llm_confidence == "high":
            confidence = "Med"

        score = max(1, min(10, score))
        business_model = "Product" if is_product else business_type.title()

        return ScoringResult(
            industry=llm_industry_name,
            business_model=business_model,
            headcount_estimate=headcount,
            crypto_adoption_likelihood="N/A",
            risk_flags=signals.risk_flags,
            score=score,
            confidence=confidence,
            reason_short=_summarize_founders(llm_industry_name, score, is_product),
            reasons_bullets=reasons,
            opener=_generate_opener_founders(domain, is_product, llm_industry_name),
            next_action=_suggest_action_founders(score),
        )

    def _score_keywords_only(self, signals: SiteSignals, domain: str) -> ScoringResult:
        """Fallback: keyword-only scoring for Founders PL."""
        top = signals.top_industry
        reasons: list[str] = []
        score = 5  # baseline
        confidence = "Med"
        business_model = "Unknown"
        is_product = False

        # Content site detection
        if signals.is_content_site:
            reasons.append("Detected as content/news site")
            score -= 2

        # Industry-based classification
        product_industries = {"SaaS (General)", "Crypto / Fintech", "Gaming / Esports / Digital Goods"}

        if top in product_industries:
            is_product = True
            business_model = "Product"
            reasons.append(f"Product company: {top}")
        elif top == "Unknown":
            confidence = "Low"
            reasons.append("Industry unclear")

        # API signals suggest product
        if signals.operational_signals.get("api_integrations", 0) > 2:
            is_product = True
            reasons.append("Strong API/integration signals suggest product company")

        # Headcount
        headcount = signals.headcount_estimate
        if headcount == "Unknown":
            score = max(score - 1, 4)
            confidence = "Low"
            reasons.append("Headcount unknown — manual check if CEO lead")
        elif headcount == "1-5":
            score = max(score - 1, 3)
            reasons.append("Very small team (1-5) — verify product stage")
        elif headcount == "10+":
            score += 1
            reasons.append("Team 10+ — established company")

        # Product company boost
        if is_product:
            score += 2
            reasons.append("Identified as product company")

        # Active hiring
        if signals.has_careers_page:
            score += 1
            reasons.append("Active hiring — growing product team")

        # No LLM note
        reasons.append("Note: LLM classification unavailable, using keyword-only scoring")

        score = max(1, min(10, score))

        return ScoringResult(
            industry=top,
            business_model=business_model if business_model != "Unknown" else _guess_model(signals),
            headcount_estimate=headcount,
            crypto_adoption_likelihood="N/A",
            risk_flags=signals.risk_flags,
            score=score,
            confidence=confidence,
            reason_short=_summarize_founders(top, score, is_product),
            reasons_bullets=reasons,
            opener=_generate_opener_founders(domain, is_product, top),
            next_action=_suggest_action_founders(score),
        )


def _guess_model(signals: SiteSignals) -> str:
    if signals.operational_signals.get("api_integrations", 0) > 0:
        return "Product"
    return "Unknown"


def _summarize_founders(industry: str, score: int, is_product: bool) -> str:
    if score >= 8:
        return f"Strong match: product company in {industry}"
    if score >= 6:
        return f"Potential match: {'product' if is_product else 'possibly product'} in {industry}"
    if score >= 4:
        return f"Needs manual review: {industry}"
    return f"Likely not a fit: {industry}"


def _generate_opener_founders(domain: str, is_product: bool, industry: str) -> str:
    if is_product:
        return (
            f"Noticed {domain} is building a product in {industry.lower()}. "
            "We have a community of product IT founders — would love to connect."
        )
    return ""


def _suggest_action_founders(score: int) -> str:
    if score >= 8:
        return "Invite to community"
    if score >= 6:
        return "Reach out, verify product focus"
    if score >= 4:
        return "Manual review needed"
    return "Skip"


# Auto-register
register_mode(FoundersPLMode())
