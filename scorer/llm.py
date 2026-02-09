"""LLM-based business classification — primary classification method.

Supports two providers:
- Anthropic (Claude) — preferred, set ANTHROPIC_API_KEY
- OpenAI (GPT) — fallback, set OPENAI_API_KEY
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_anthropic_client = None
_openai_client = None


def _get_provider() -> str:
    """Determine which LLM provider to use."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return ""


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            from anthropic import AsyncAnthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            _anthropic_client = AsyncAnthropic(api_key=api_key)
        except ImportError:
            logger.info("anthropic package not installed")
            return None
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import AsyncOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return None
            _openai_client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            logger.info("openai package not installed")
            return None
    return _openai_client


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


# ── Business classification prompt ──

CLASSIFY_SYSTEM_PROMPT = """You are a business analyst. Your task is to determine what a company DOES based on its website text.

CRITICAL RULES:
1. Classify the company's OWN business, NOT topics they write about.
   - A news portal writing about crypto is "media", NOT "crypto_fintech"
   - A marketing agency offering affiliate services to clients is "agency", NOT "affiliate_cpa"
   - A blog about gambling is "media", NOT "igaming"
   - An IT outsourcing company is "agency", NOT "saas"
   - A company that MENTIONS crypto in passing is NOT "crypto_fintech" unless crypto is their core business

2. Look for clear signals of the company's primary activity:
   - Do they SELL a product/service? What is it?
   - Do they have PRICING pages? What are they charging for?
   - Do they have a SIGN UP flow? For what?
   - Are they writing ARTICLES/NEWS? Then they're media.
   - Do they list services for clients? Then they're agency/service.

3. For non-English websites: translate and understand the content, don't just look for English keywords.

Return ONLY valid JSON with these fields:
{
  "primary_business": "1-2 sentence description of what this company does",
  "industry": "one of the INDUSTRY_CODES below",
  "business_type": "product | service | marketplace | media | agency | other",
  "is_content_site": true/false,
  "confidence": "high | medium | low"
}

INDUSTRY_CODES (use exactly one):
- "affiliate_cpa" — Affiliate networks, CPA networks (NOT agencies that offer affiliate marketing services)
- "igaming" — Online casinos, betting platforms, sportsbooks (NOT news about gambling)
- "adult" — Adult content platforms, webcam sites
- "hosting" — Web hosting, VPS, dedicated servers, data centers
- "vpn_privacy" — VPN services, privacy tools
- "freelance_contractor" — Freelance marketplaces, contractor platforms
- "payroll_payouts" — Payroll services, mass payout platforms
- "marketplace" — Multi-vendor marketplaces, e-commerce platforms with multiple sellers
- "gaming_esports" — Gaming, esports, digital goods platforms
- "crypto_fintech" — Crypto exchanges, wallets, DeFi, fintech platforms, payment processors
- "high_risk_ecommerce" — Supplements, nutra, CBD, forex tools
- "psp_orchestration" — Payment orchestration, payment service providers, billing platforms
- "affiliate_tracking" — Affiliate tracking software, conversion attribution
- "saas" — SaaS products, cloud platforms (not fitting other categories)
- "ecommerce" — Regular online stores, retail
- "agency" — Marketing agencies, development agencies, consulting firms, IT outsourcing
- "media" — News portals, blogs, content sites, magazines
- "education" — Educational platforms, courses, training programs
- "other" — Anything not fitting above categories
- "unknown" — Cannot determine from available text"""


async def classify_business(domain: str, text: str, mode_id: str) -> dict | None:
    """Classify what business a website represents using LLM.

    Returns dict with: primary_business, industry, business_type,
    is_content_site, confidence. Or None on failure.
    """
    provider = _get_provider()
    if not provider:
        return None

    # Truncate to fit context while keeping enough signal
    truncated = text[:6000]

    mode_context = ""
    if mode_id == "inxy_leads":
        mode_context = (
            "\nContext: We're evaluating leads for a crypto payment processor. "
            "We need to know the company's ACTUAL business to assess if they "
            "could benefit from crypto payment processing. "
            "Be strict: only classify as crypto_fintech if crypto IS their core business."
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
        if provider == "anthropic":
            content = await _call_anthropic(user_prompt)
        else:
            content = await _call_openai(user_prompt)

        if not content:
            return None

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

        logger.info(f"LLM [{provider}] classified {domain}: {result.get('industry')} / {result.get('business_type')}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON for {domain}: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM classification failed for {domain}: {e}")
        return None


async def _call_anthropic(user_prompt: str) -> str | None:
    """Call Anthropic Claude API."""
    client = _get_anthropic_client()
    if client is None:
        return None

    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=300,
        system=CLASSIFY_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.content[0].text.strip()


async def _call_openai(user_prompt: str) -> str | None:
    """Call OpenAI API."""
    client = _get_openai_client()
    if client is None:
        return None

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


# ── Industry code mapping to our scoring categories ──

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

NON_TARGET_INDUSTRIES = {"media", "education", "agency", "other", "unknown"}


def map_industry_code(code: str) -> str:
    """Map LLM industry code to display name."""
    return INDUSTRY_CODE_MAP.get(code, code)


def is_non_target(classification: dict) -> bool:
    """Check if LLM classification indicates a non-target business."""
    industry = classification.get("industry", "unknown")
    is_content = classification.get("is_content_site", False)
    return industry in NON_TARGET_INDUSTRIES or is_content
