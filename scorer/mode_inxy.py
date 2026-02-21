"""Inxy Leads scoring mode — primary mode for crypto payment processing leads."""

from scorer.extractor import SiteSignals
from scorer.modes import BaseMode, ScoringResult, register_mode
from scorer import llm

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
    "Dev Studio / IT Outsourcing",
    "Creator / Royalty Platform",
}

# Industries that serve high-crypto clients → also strong signal
INFRA_SERVING_CRYPTO = {
    "Hosting / Infrastructure",
    "Payment Orchestration / PSP",
    "Affiliate Tracking Software",
    "SaaS (General)",
    "eSIM / Telecom",
    "Bug Bounty / Rewards",
}

# LLM industry codes mapped to the same categories
HIGH_CRYPTO_CODES = {
    "affiliate_cpa", "igaming", "adult", "hosting", "vpn_privacy",
    "freelance_contractor", "payroll_payouts", "gaming_esports",
    "crypto_fintech", "high_risk_ecommerce",
}
SECONDARY_CODES = {
    "psp_orchestration", "affiliate_tracking", "marketplace",
    "dev_studio", "creator_platform",
}
INFRA_CODES = {
    "hosting", "psp_orchestration", "affiliate_tracking", "saas",
    "esim_telecom", "bug_bounty",
}

# ── Use case mapping by industry code ──
# pay_in:  company accepts crypto from customers/clients
# pay_out: company does mass stablecoin payouts to recipients
# exchange: crypto<>fiat conversion, treasury, settlement rails
USE_CASE_BY_INDUSTRY = {
    "affiliate_cpa": ["pay_out"],
    "igaming": ["pay_in"],
    "adult": ["pay_in"],
    "hosting": ["pay_in"],
    "vpn_privacy": ["pay_in", "pay_out"],
    "freelance_contractor": ["pay_out"],
    "payroll_payouts": ["pay_out"],
    "marketplace": ["pay_in", "pay_out"],
    "gaming_esports": ["pay_in", "pay_out"],
    "crypto_fintech": ["exchange"],
    "high_risk_ecommerce": ["pay_in"],
    "psp_orchestration": ["exchange"],
    "affiliate_tracking": ["pay_out"],
    "esim_telecom": ["pay_in"],
    "dev_studio": ["pay_in", "pay_out"],
    "creator_platform": ["pay_in", "pay_out"],
    "bug_bounty": ["pay_out"],
    "saas": ["pay_in"],
    "ecommerce": ["pay_in"],
    "agency": ["pay_in"],
}

# Use case mapping by keyword-detected industry (display names)
USE_CASE_BY_INDUSTRY_NAME = {
    "Affiliate / CPA Marketing": ["pay_out"],
    "iGaming & Betting": ["pay_in"],
    "Adult / Webcam": ["pay_in"],
    "Hosting / Infrastructure": ["pay_in"],
    "VPN / Privacy / Security": ["pay_in", "pay_out"],
    "Freelance / Contractor Platform": ["pay_out"],
    "Global Payroll / Payouts": ["pay_out"],
    "Marketplace": ["pay_in", "pay_out"],
    "Gaming / Esports / Digital Goods": ["pay_in", "pay_out"],
    "Crypto / Fintech": ["exchange"],
    "High-Risk Ecommerce": ["pay_in"],
    "Payment Orchestration / PSP": ["exchange"],
    "Affiliate Tracking Software": ["pay_out"],
    "eSIM / Telecom": ["pay_in"],
    "Dev Studio / IT Outsourcing": ["pay_in", "pay_out"],
    "Creator / Royalty Platform": ["pay_in", "pay_out"],
    "Bug Bounty / Rewards": ["pay_out"],
    "SaaS (General)": ["pay_in"],
    "Agency / Consulting": ["pay_in"],
}


class InxyLeadsMode(BaseMode):
    mode_id = "inxy_leads"
    mode_name = "Inxy Leads (Crypto Payments)"

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
                reason_short="Non-business website (blog, NGO, personal site)",
                reasons_bullets=["Not a business website"],
                confidence="Med",
            )

        if not signals.industries and signals.raw_text_length < 200 and not classification:
            return ScoringResult(
                score=1,
                reason_short="Could not extract meaningful content from website",
                reasons_bullets=["Website empty or inaccessible"],
                confidence="Low",
            )

        # ── Determine industry: LLM primary, keywords fallback ──
        if classification:
            return self._score_with_classification(classification, signals, domain)
        else:
            return self._score_keywords_only(signals, domain)

    def _score_with_classification(
        self, classification: dict, signals: SiteSignals, domain: str
    ) -> ScoringResult:
        """Score using LLM classification (primary) + keyword signals (secondary)."""
        llm_industry_code = classification.get("industry", "unknown")
        llm_industry_name = llm.map_industry_code(llm_industry_code)
        business_type = classification.get("business_type", "unknown")
        is_content_site = classification.get("is_content_site", False)
        llm_confidence = classification.get("confidence", "medium")
        primary_business = classification.get("primary_business", "")

        reasons: list[str] = []
        score = 3  # baseline
        crypto_likelihood = "Low"
        signal_strength = 0

        # ── Content site / non-target early exit ──
        if is_content_site or llm_industry_code in llm.NON_TARGET_INDUSTRIES:
            reasons.append(f"LLM: {primary_business}")
            reasons.append(f"Business type: {business_type}")

            # Still check if they have crypto operational signals
            if signals.has_crypto_signals:
                reasons.append("Note: crypto payment keywords found on site")
                return ScoringResult(
                    industry=llm_industry_name,
                    business_model=business_type,
                    headcount_estimate=signals.headcount_estimate,
                    crypto_adoption_likelihood="Low",
                    risk_flags=signals.risk_flags,
                    score=3,
                    confidence="Med",
                    reason_short=f"Non-target business ({llm_industry_name}) but mentions crypto",
                    reasons_bullets=reasons,
                    next_action="Manual review — non-target but crypto mentions",
                    use_cases=["pay_in"],
                )

            return ScoringResult(
                industry=llm_industry_name,
                business_model=business_type,
                headcount_estimate=signals.headcount_estimate,
                score=1,
                confidence="High" if llm_confidence == "high" else "Med",
                reason_short=f"Non-target: {llm_industry_name}",
                reasons_bullets=reasons,
                next_action="Skip or deprioritize",
            )

        # ── Agency: not auto-reject but uncertain ICP, score modestly ──
        if llm_industry_code == "agency":
            score += 1
            reasons.append(f"LLM: '{llm_industry_name}' — potential use case: crypto B2B invoices")
            reasons.append("Agency ICP unclear — needs discovery to confirm crypto demand")

            if signals.has_crypto_signals:
                score += 2
                signal_strength += 2
                crypto_likelihood = "Medium"
                reasons.append("Explicit crypto/stablecoin payment signals found on site")
            if signals.has_global_payment_signals:
                score += 1
                signal_strength += 1
                reasons.append("Cross-border / multi-currency signals found")

            score = max(1, min(10, score))
            use_cases = ["pay_in"]
            if signals.has_mass_payment_signals:
                use_cases.append("pay_out")
                reasons.append("Mass payment / payout signals found")

            confidence = "Med" if llm_confidence == "high" else "Low"
            return ScoringResult(
                industry=llm_industry_name,
                business_model=business_type,
                headcount_estimate=signals.headcount_estimate,
                crypto_adoption_likelihood=crypto_likelihood,
                risk_flags=signals.risk_flags,
                score=score,
                confidence=confidence,
                reason_short=f"Agency/consulting — potential crypto invoices use case",
                reasons_bullets=reasons,
                opener=_generate_opener(llm_industry_name, signals, domain),
                next_action="Manual review before outreach" if score >= 4 else "Skip or deprioritize",
                use_cases=use_cases,
            )

        # ── Industry scoring based on LLM classification ──
        if llm_industry_code in HIGH_CRYPTO_CODES:
            score += 4
            crypto_likelihood = "High"
            signal_strength += 2
            reasons.append(f"LLM: '{llm_industry_name}' — high crypto adoption industry")
        elif llm_industry_code in SECONDARY_CODES:
            score += 3
            crypto_likelihood = "Medium"
            signal_strength += 1
            reasons.append(f"LLM: '{llm_industry_name}' — serves crypto-adjacent clients")
        elif llm_industry_code in INFRA_CODES:
            score += 2
            crypto_likelihood = "Medium"
            signal_strength += 1
            reasons.append(f"LLM: infrastructure/SaaS serving high-crypto verticals")
        elif llm_industry_code == "ecommerce":
            score += 1
            reasons.append(f"LLM: regular e-commerce ({primary_business})")
        else:
            reasons.append(f"LLM: '{llm_industry_name}' ({primary_business})")

        # ── Operational signal boosts (from keywords) ──
        if signals.has_crypto_signals:
            score += 2
            crypto_likelihood = "High"
            signal_strength += 2
            reasons.append("Explicit crypto/stablecoin payment signals found on site")

        if signals.has_mass_payment_signals:
            score += 1
            signal_strength += 1
            if crypto_likelihood != "High":
                crypto_likelihood = "Medium"
            reasons.append("Mass payment / payout signals found")

        if signals.has_global_payment_signals:
            score += 1
            signal_strength += 1
            if crypto_likelihood == "Low":
                crypto_likelihood = "Medium"
            reasons.append("Cross-border / multi-currency signals found")

        if signals.operational_signals.get("api_integrations", 0) > 0:
            score += 1
            signal_strength += 1
            reasons.append("API / integration-ready platform")

        if signals.operational_signals.get("partners", 0) > 0:
            score += 1
            signal_strength += 1
            reasons.append("Partner / reseller program detected")

        # ── Confidence calculation ──
        has_enough_text = signals.raw_text_length >= 2000
        has_some_text = signals.raw_text_length >= 1000

        # LLM classification boosts confidence
        if llm_confidence == "high" and signal_strength >= 1:
            confidence = "High"
        elif llm_confidence == "high" or (signal_strength >= 2 and has_some_text):
            confidence = "High"
        elif signal_strength >= 1 and has_some_text:
            confidence = "Med"
        elif has_some_text:
            confidence = "Med"
        else:
            confidence = "Low"
            reasons.append("Limited website content available")

        # Risk flags lower confidence by one step
        if signals.risk_flags:
            if confidence == "High":
                confidence = "Med"
            elif confidence == "Med":
                confidence = "Low"
            reasons.append(f"Risk flags: {', '.join(signals.risk_flags)}")

        # Cap score
        score = max(1, min(10, score))

        # ── Use case detection ──
        use_cases = _detect_use_cases(llm_industry_code, signals)

        # Generate opener
        business_model = business_type if business_type != "other" else _infer_business_model(signals)
        opener = _generate_opener(llm_industry_name, signals, domain)
        next_action = _suggest_next_action(score, signals)

        return ScoringResult(
            industry=llm_industry_name,
            business_model=business_model,
            headcount_estimate=signals.headcount_estimate,
            crypto_adoption_likelihood=crypto_likelihood,
            risk_flags=signals.risk_flags,
            score=score,
            confidence=confidence,
            reason_short=_summarize(llm_industry_name, score, crypto_likelihood),
            reasons_bullets=reasons,
            opener=opener,
            next_action=next_action,
            use_cases=use_cases,
        )

    def _score_keywords_only(self, signals: SiteSignals, domain: str) -> ScoringResult:
        """Fallback: score using only keyword signals (when LLM unavailable)."""
        top = signals.top_industry
        reasons: list[str] = []
        score = 3  # baseline
        crypto_likelihood = "Low"
        business_model = _infer_business_model(signals)
        signal_strength = 0

        # Content site detection lowers score
        if signals.is_content_site:
            reasons.append("Detected as content/news site (keyword pattern)")
            score -= 1

        # ── Industry scoring ──
        if top in HIGH_CRYPTO_INDUSTRIES:
            score += 4
            crypto_likelihood = "High"
            signal_strength += 2
            reasons.append(f"Industry '{top}' has high crypto adoption historically")
        elif top in SECONDARY_INDUSTRIES:
            score += 3
            crypto_likelihood = "Medium"
            signal_strength += 1
            reasons.append(f"Industry '{top}' serves crypto-adjacent clients")
        elif top in INFRA_SERVING_CRYPTO:
            score += 2
            crypto_likelihood = "Medium"
            signal_strength += 1
            reasons.append(f"Infrastructure/SaaS serving high-crypto verticals")
        elif top != "Unknown":
            score += 0
            reasons.append(f"Industry '{top}' has low crypto adoption signal")

        # ── Operational signal boosts ──
        if signals.has_crypto_signals:
            score += 2
            crypto_likelihood = "High"
            signal_strength += 2
            reasons.append("Explicit crypto/stablecoin payment signals found")

        if signals.has_mass_payment_signals:
            score += 1
            signal_strength += 1
            if crypto_likelihood != "High":
                crypto_likelihood = "Medium"
            reasons.append("Mass payment / payout signals found")

        if signals.has_global_payment_signals:
            score += 1
            signal_strength += 1
            if crypto_likelihood == "Low":
                crypto_likelihood = "Medium"
            reasons.append("Cross-border / multi-currency signals found")

        if signals.operational_signals.get("api_integrations", 0) > 0:
            score += 1
            signal_strength += 1
            reasons.append("API / integration-ready platform")

        if signals.operational_signals.get("partners", 0) > 0:
            score += 1
            signal_strength += 1
            reasons.append("Partner / reseller program detected")

        # ── Multiple high-crypto industries detected ──
        high_matches = [i for i in signals.industries if i in HIGH_CRYPTO_INDUSTRIES]
        if len(high_matches) >= 2:
            score += 1
            signal_strength += 1
            reasons.append(f"Multiple high-crypto industries: {', '.join(high_matches[:3])}")

        # ── Confidence calculation ──
        has_enough_text = signals.raw_text_length >= 2000
        has_some_text = signals.raw_text_length >= 1000

        if signal_strength >= 3 and has_enough_text:
            confidence = "High"
        elif signal_strength >= 2 and has_some_text:
            confidence = "High"
        elif signal_strength >= 1 and has_some_text:
            confidence = "Med"
        elif has_some_text:
            confidence = "Med"
        else:
            confidence = "Low"
            reasons.append("Limited website content available")

        # Note: no LLM available
        if confidence != "Low":
            reasons.append("Note: LLM classification unavailable, using keyword-only scoring")
            if confidence == "High":
                confidence = "Med"  # downgrade without LLM confirmation

        # Risk flags lower confidence by one step
        if signals.risk_flags:
            if confidence == "High":
                confidence = "Med"
            elif confidence == "Med":
                confidence = "Low"
            reasons.append(f"Risk flags: {', '.join(signals.risk_flags)}")

        # Cap score
        score = max(1, min(10, score))

        # ── Use case detection ──
        use_cases = _detect_use_cases_by_name(top, signals)

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
            use_cases=use_cases,
        )


def _detect_use_cases(industry_code: str, signals: SiteSignals) -> list[str]:
    """Detect relevant product use cases based on industry code and signals."""
    use_cases = list(USE_CASE_BY_INDUSTRY.get(industry_code, []))
    # Enrich from operational signals
    if signals.has_crypto_signals and "pay_in" not in use_cases:
        use_cases.append("pay_in")
    if signals.has_mass_payment_signals and "pay_out" not in use_cases:
        use_cases.append("pay_out")
    return use_cases


def _detect_use_cases_by_name(industry_name: str, signals: SiteSignals) -> list[str]:
    """Detect use cases from keyword-detected industry (display name)."""
    use_cases = list(USE_CASE_BY_INDUSTRY_NAME.get(industry_name, []))
    if signals.has_crypto_signals and "pay_in" not in use_cases:
        use_cases.append("pay_in")
    if signals.has_mass_payment_signals and "pay_out" not in use_cases:
        use_cases.append("pay_out")
    return use_cases


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
    all_target_industries = HIGH_CRYPTO_INDUSTRIES | SECONDARY_INDUSTRIES | INFRA_SERVING_CRYPTO
    if industry in all_target_industries:
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
