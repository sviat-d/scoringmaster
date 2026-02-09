"""LLM-based business classification — primary classification method."""

import os
import json
import logging

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import AsyncOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return None
            _client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            logger.info("openai package not installed, LLM classification disabled")
            return None
    return _client


def is_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


# ── Business classification prompt ──

CLASSIFY_SYSTEM_PROMPT = """You are a business analyst. Your task is to determine what a company DOES based on its website text.

CRITICAL RULES:
1. Classify the company's OWN business, NOT topics they write about.
   - A news portal writing about crypto is "Media / News", NOT "Crypto / Fintech"
   - A marketing agency offering affiliate services to clients is "Agency / Consulting", NOT "Affiliate / CPA Marketing"
   - A blog about gambling is "Media / News", NOT "iGaming & Betting"

2. Look for clear signals of the company's primary activity:
   - Do they SELL a product/service? What is it?
   - Do they have PRICING pages? What are they charging for?
   - Do they have a SIGN UP flow? For what?
   - Are they writing ARTICLES/NEWS? Then they're media.

Return ONLY valid JSON with these fields:
{
  "primary_business": "1-2 sentence description of what this company does",
  "industry": "one of the INDUSTRY_CODES below",
  "business_type": "product | service | marketplace | media | agency | other",
  "is_content_site": true/false,
  "confidence": "high | medium | low"
}

INDUSTRY_CODES (use exactly one):
- "affiliate_cpa" — Affiliate networks, CPA networks, performance marketing platforms
- "igaming" — Online casinos, betting platforms, sportsbooks
- "adult" — Adult content platforms, webcam sites
- "hosting" — Web hosting, VPS, dedicated servers, data centers
- "vpn_privacy" — VPN services, privacy tools
- "freelance_contractor" — Freelance marketplaces, contractor platforms
- "payroll_payouts" — Payroll services, mass payout platforms
- "marketplace" — Multi-vendor marketplaces, e-commerce platforms with multiple sellers
- "gaming_esports" — Gaming, esports, digital goods platforms
- "crypto_fintech" — Crypto exchanges, wallets, DeFi, fintech, payment processors
- "high_risk_ecommerce" — Supplements, nutra, CBD, forex tools
- "psp_orchestration" — Payment orchestration, payment service providers, billing platforms
- "affiliate_tracking" — Affiliate tracking software, conversion attribution
- "saas" — SaaS products, cloud platforms (not fitting other categories)
- "ecommerce" — Regular online stores, retail
- "agency" — Marketing agencies, development agencies, consulting firms
- "media" — News portals, blogs, content sites, magazines
- "education" — Educational platforms, courses, training programs
- "other" — Anything not fitting above categories
- "unknown" — Cannot determine from available text"""


async def classify_business(domain: str, text: str, mode_id: str) -> dict | None:
    """Classify what business a website represents using LLM.

    Returns dict with: primary_business, industry, business_type,
    is_content_site, confidence. Or None on failure.
    """
    client = _get_client()
    if client is None:
        return None

    # Truncate to fit context while keeping enough signal
    truncated = text[:6000]

    mode_context = ""
    if mode_id == "inxy_leads":
        mode_context = (
            "\nContext: We're evaluating leads for a crypto payment processor. "
            "We need to know the company's ACTUAL business to assess if they "
            "could benefit from crypto payment processing."
        )
    elif mode_id == "founders_pl":
        mode_context = (
            "\nContext: We're looking for product IT founders. "
            "We need to know if this is a product company vs agency/service."
        )

    user_prompt = f"""Domain: {domain}
{mode_context}

Website content:
{truncated}

Classify this company's business. Return JSON only."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        result = json.loads(content)

        # Validate required fields
        required = {"primary_business", "industry", "business_type"}
        if not required.issubset(result.keys()):
            logger.warning(f"LLM missing fields for {domain}: {result.keys()}")
            return None

        return result
    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON for {domain}: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM classification failed for {domain}: {e}")
        return None


# ── Industry code mapping to our scoring categories ──

# Map LLM industry codes to the internal industry names used by scoring modes
INDUSTRY_CODE_MAP = {
    "affiliate_cpa": "Affiliate / CPA Marketing",
    "igaming": "iGaming & Betting",
    "adult": "Adult / Webcam",
    "hosting": "Hosting / Infrastructure",
    "vpn_privacy": "VPN / Privacy / Security",
    "freelance_contractor": "Freelance / Contractor Platform",
    "payroll_payouts": "Global Payroll / Payouts",
    "marketplace": "Marketplace",
    "gaming_esports": "Gaming / Esports / Digital Goods",
    "crypto_fintech": "Crypto / Fintech",
    "high_risk_ecommerce": "High-Risk Ecommerce",
    "psp_orchestration": "Payment Orchestration / PSP",
    "affiliate_tracking": "Affiliate Tracking Software",
    "saas": "SaaS (General)",
    "ecommerce": "Ecommerce",
    "agency": "Agency / Consulting",
    "media": "Media / News",
    "education": "Education",
    "other": "Other",
    "unknown": "Unknown",
}

# Industries where LLM classification means "not a real lead"
NON_TARGET_INDUSTRIES = {"media", "education", "agency", "other", "unknown"}


def map_industry_code(code: str) -> str:
    """Map LLM industry code to display name."""
    return INDUSTRY_CODE_MAP.get(code, code)


def is_non_target(classification: dict) -> bool:
    """Check if LLM classification indicates a non-target business."""
    industry = classification.get("industry", "unknown")
    is_content = classification.get("is_content_site", False)
    return industry in NON_TARGET_INDUSTRIES or is_content
