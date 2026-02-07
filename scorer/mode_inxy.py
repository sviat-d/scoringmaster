"""Inxy Leads scoring mode — primary mode for crypto payment processing leads."""

from scorer.extractor import SiteSignals
from scorer.modes import BaseMode, ScoringResult, register_mode

# Industries with historically high crypto adoption
HIGH_CRYPTO_INDUSTRIES = {
    "Affiliate / CPA Marketing",
    "iGaming & Betting",
    "Adult / Webcam",
    "Hosting / Infrastructure",
    "VPN / Privacy / Security",
    "Freelance / Contractor Platform",
    "Global Payroll / Payouts",
    "Gaming / Esports / Digital Goods",
    "Crypto / Fintech",
    "High-Risk Ecommerce",
}

# Secondary / infrastructure targets
SECONDARY_INDUSTRIES = {
    "Payment Orchestration / PSP",
    "Affiliate Tracking Software",
    "Marketplace",
}

# Industries that serve high-crypto clients → also strong signal
INFRA_SERVING_CRYPTO = {
    "Hosting / Infrastructure",
    "Payment Orchestration / PSP",
    "Affiliate Tracking Software",
    "SaaS (General)",
}


class InxyLeadsMode(BaseMode):
    mode_id = "inxy_leads"
    mode_name = "Inxy Leads (Crypto Payments)"

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
                reason_short="Non-business website (blog, NGO, personal site)",
                reasons_bullets=["Not a business website"],
                confidence="Med",
            )

        if not signals.industries and signals.raw_text_length < 200:
            return ScoringResult(
                score=1,
                reason_short="Could not extract meaningful content from website",
                reasons_bullets=["Website empty or inaccessible"],
                confidence="Low",
            )

        top = signals.top_industry
        reasons: list[str] = []
        score = 3  # baseline
        confidence = "Med"
        crypto_likelihood = "Low"
        business_model = _infer_business_model(signals)

        # ── Industry scoring ──
        if top in HIGH_CRYPTO_INDUSTRIES:
            score += 4
            crypto_likelihood = "High"
            reasons.append(f"Industry '{top}' has high crypto adoption historically")
        elif top in SECONDARY_INDUSTRIES:
            score += 3
            crypto_likelihood = "Medium"
            reasons.append(f"Industry '{top}' serves crypto-adjacent clients")
        elif top in INFRA_SERVING_CRYPTO:
            score += 2
            crypto_likelihood = "Medium"
            reasons.append(f"Infrastructure/SaaS serving high-crypto verticals")
        elif top != "Unknown":
            score += 0
            reasons.append(f"Industry '{top}' has low crypto adoption signal")

        # ── Operational signal boosts ──
        if signals.has_crypto_signals:
            score += 2
            crypto_likelihood = "High"
            reasons.append("Explicit crypto/stablecoin payment signals found")

        if signals.has_mass_payment_signals:
            score += 1
            if crypto_likelihood != "High":
                crypto_likelihood = "Medium"
            reasons.append("Mass payment / payout signals found")

        if signals.has_global_payment_signals:
            score += 1
            if crypto_likelihood == "Low":
                crypto_likelihood = "Medium"
            reasons.append("Cross-border / multi-currency signals found")

        if signals.operational_signals.get("api_integrations", 0) > 0:
            score += 1
            reasons.append("API / integration-ready platform")

        if signals.operational_signals.get("partners", 0) > 0:
            score += 1
            reasons.append("Partner / reseller program detected")

        # ── Multiple high-crypto industries detected ──
        high_matches = [i for i in signals.industries if i in HIGH_CRYPTO_INDUSTRIES]
        if len(high_matches) >= 2:
            score += 1
            reasons.append(f"Multiple high-crypto industries: {', '.join(high_matches[:3])}")

        # ── Risk flags (lower confidence, not score) ──
        if signals.risk_flags:
            confidence = "Med" if confidence == "High" else "Low"
            reasons.append(f"Risk flags: {', '.join(signals.risk_flags)}")

        # ── Low content confidence ──
        if signals.raw_text_length < 1000:
            confidence = "Low"
            reasons.append("Limited website content available")

        # Cap score
        score = max(1, min(10, score))

        # Generate opener
        opener = _generate_opener(top, signals, domain)
        next_action = _suggest_next_action(score, signals)

        return ScoringResult(
            industry=top,
            business_model=business_model,
            headcount_estimate=signals.headcount_estimate,
            crypto_adoption_likelihood=crypto_likelihood,
            risk_flags=signals.risk_flags,
            score=score,
            confidence=confidence,
            reason_short=_summarize(top, score, crypto_likelihood),
            reasons_bullets=reasons,
            opener=opener,
            next_action=next_action,
        )


def _infer_business_model(signals: SiteSignals) -> str:
    industries = set(signals.industries.keys())
    product_signals = {
        "SaaS (General)", "Payment Orchestration / PSP",
        "Affiliate Tracking Software", "Crypto / Fintech",
    }
    service_signals = {
        "Freelance / Contractor Platform", "Global Payroll / Payouts",
    }

    is_product = bool(industries & product_signals)
    is_service = bool(industries & service_signals)

    if is_product and is_service:
        return "Hybrid"
    if is_product:
        return "Product"
    if is_service:
        return "Service"
    if signals.operational_signals.get("api_integrations", 0) > 2:
        return "Product"
    return "Unknown"


def _summarize(industry: str, score: int, crypto_likelihood: str) -> str:
    if score >= 8:
        return f"Strong lead: {industry} with {crypto_likelihood.lower()} crypto adoption likelihood"
    if score >= 6:
        return f"Promising lead: {industry}, crypto adoption {crypto_likelihood.lower()}"
    if score >= 4:
        return f"Needs review: {industry}, crypto signals unclear"
    return f"Low relevance: {industry}"


def _generate_opener(industry: str, signals: SiteSignals, domain: str) -> str:
    if industry in HIGH_CRYPTO_INDUSTRIES:
        if signals.has_crypto_signals:
            return (
                f"I noticed {domain} already works with crypto payments. "
                "We help companies like yours streamline crypto processing with "
                "lower fees and faster settlements."
            )
        return (
            f"Many companies in {industry.lower()} are adopting crypto payments "
            "to reduce processing costs and reach global customers. "
            "Would love to share how Inxy helps with that."
        )
    if signals.has_mass_payment_signals or signals.has_global_payment_signals:
        return (
            f"I see {domain} handles international payments. "
            "Crypto rails can significantly reduce cross-border fees. "
            "Happy to show how Inxy makes this seamless."
        )
    return ""


def _suggest_next_action(score: int, signals: SiteSignals) -> str:
    if score >= 8:
        return "Priority outreach — personalized email + LinkedIn"
    if score >= 6:
        return "Standard outreach sequence"
    if score >= 4:
        return "Manual review before outreach"
    return "Skip or deprioritize"


# Auto-register
register_mode(InxyLeadsMode())
