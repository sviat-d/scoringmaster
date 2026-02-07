"""Founders PL scoring mode — secondary mode for product IT founders community."""

from scorer.extractor import SiteSignals
from scorer.modes import BaseMode, ScoringResult, register_mode

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


class FoundersPLMode(BaseMode):
    mode_id = "founders_pl"
    mode_name = "Founders PL (Product IT Founders)"

    def score(self, signals: SiteSignals, domain: str) -> ScoringResult:
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

        if not signals.industries and signals.raw_text_length < 200:
            return ScoringResult(
                score=1,
                reason_short="Could not extract meaningful content",
                reasons_bullets=["Website empty or inaccessible"],
                confidence="Low",
            )

        # Check for service vs product company using raw text
        # We need to check all page text, but signals already processed industries
        # Use industry detection as proxy

        top = signals.top_industry
        reasons: list[str] = []
        score = 5  # baseline for founders
        confidence = "Med"
        business_model = "Unknown"
        is_service = False
        is_product = False

        # Industry-based classification
        service_industries = set()
        product_industries = {"SaaS (General)", "Crypto / Fintech", "Gaming / Esports / Digital Goods"}

        if top in product_industries:
            is_product = True
            business_model = "Product"
            reasons.append(f"Product company: {top}")
        elif top == "Unknown":
            confidence = "Low"
            reasons.append("Industry unclear")

        # Operational signals that suggest product
        if signals.operational_signals.get("api_integrations", 0) > 2:
            is_product = True
            reasons.append("Strong API/integration signals suggest product company")

        # Headcount rules for CEO filtering
        headcount = signals.headcount_estimate
        if headcount == "Unknown":
            score = max(score - 1, 4)
            confidence = "Low"
            reasons.append("Headcount unknown — manual check if CEO lead")

        elif headcount in ("1-5", "5-10"):
            # Small team — might be too small for founders community
            if headcount == "1-5":
                score = max(score - 1, 3)
                reasons.append("Very small team (1-5) — verify product stage")

        elif headcount == "10+":
            score += 1
            reasons.append("Team 10+ — established company")

        # Product company boost
        if is_product:
            score += 2
            reasons.append("Identified as product company")
        elif is_service:
            score = 2
            reasons.append("Service/agency/outsourcing company — reject for Founders PL")
            return ScoringResult(
                industry=top,
                business_model="Service",
                headcount_estimate=headcount,
                score=2,
                confidence=confidence,
                reason_short="Service company — not a fit for Founders PL",
                reasons_bullets=reasons,
            )

        # Has own platform/product signals
        if signals.has_careers_page:
            score += 1
            reasons.append("Active hiring — growing product team")

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
