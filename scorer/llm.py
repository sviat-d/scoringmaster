"""LLM-based business classification — primary classification method.

Primary provider:
- Anthropic (Claude Haiku 4.5) — best value for classification, set ANTHROPIC_API_KEY

Additional providers (if configured):
- OpenAI (GPT) — set OPENAI_API_KEY
- Google Gemini (Flash) — free tier, set GEMINI_API_KEY
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_anthropic_client = None
_openai_client = None
_gemini_model = None


def _can_import(module_name: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _get_provider() -> str:
    """Determine default LLM provider (quality-first: Anthropic > OpenAI > Gemini)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        if _can_import("anthropic"):
            return "anthropic"
        logger.warning("ANTHROPIC_API_KEY is set but 'anthropic' package is not installed. Run: pip install anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        if _can_import("openai"):
            return "openai"
        logger.warning("OPENAI_API_KEY is set but 'openai' package is not installed. Run: pip install openai")
    if os.environ.get("GEMINI_API_KEY"):
        if _can_import("google.generativeai"):
            return "gemini"
        logger.warning("GEMINI_API_KEY is set but 'google-generativeai' package is not installed. Run: pip install google-generativeai")
    return ""


def get_available_providers() -> list[dict]:
    """Return list of available LLM providers for UI display."""
    providers = []
    if os.environ.get("GEMINI_API_KEY") and _can_import("google.generativeai"):
        providers.append({"id": "gemini", "name": "Gemini Flash (free)"})
    if os.environ.get("ANTHROPIC_API_KEY") and _can_import("anthropic"):
        providers.append({"id": "anthropic", "name": "Claude Haiku 4.5"})
    if os.environ.get("OPENAI_API_KEY") and _can_import("openai"):
        providers.append({"id": "openai", "name": "GPT-4o Mini"})
    return providers


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


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return None
            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=CLASSIFY_SYSTEM_PROMPT,
            )
        except ImportError:
            logger.info("google-generativeai package not installed")
            return None
    return _gemini_model


def is_available() -> bool:
    return bool(_get_provider())


# ── Business classification prompt ──

CLASSIFY_SYSTEM_PROMPT = """You are a business analyst. Your task is to determine what a company DOES based on its website text.

CRITICAL RULES:
1. Classify the company's OWN business, NOT topics they write about.
   - A news portal writing about crypto is "media", NOT "crypto_fintech"
   - A marketing agency offering affiliate services to clients is "agency", NOT "affiliate_cpa"
   - A blog about gambling is "media", NOT "igaming"
   - A software dev studio / IT outsourcing company with distributed teams is "dev_studio", NOT "agency" or "saas"
   - A pure consulting / marketing agency (no dev teams) is "agency"
   - A company that MENTIONS crypto in passing is NOT "crypto_fintech" unless crypto is their core business
   - An eSIM / travel connectivity provider is "esim_telecom", NOT "ecommerce"
   - A bug bounty / security reward platform is "bug_bounty", NOT "saas"
   - A creator donation / royalty platform is "creator_platform", NOT "marketplace"

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
- "gaming_esports" — Gaming, esports, digital goods platforms, skin/item marketplaces
- "crypto_fintech" — Crypto exchanges, wallets, DeFi, fintech platforms, payment processors
- "high_risk_ecommerce" — Supplements, nutra, CBD, forex tools
- "psp_orchestration" — Payment orchestration, payment service providers, billing platforms
- "affiliate_tracking" — Affiliate tracking software, conversion attribution
- "esim_telecom" — eSIM providers, travel connectivity, mobile virtual operators (MVNO)
- "dev_studio" — Software development agencies/studios with distributed teams, IT outsourcing companies (NOT pure consulting)
- "creator_platform" — Creator economy platforms, royalty distribution, donation/tip platforms
- "bug_bounty" — Bug bounty platforms, security reward programs, vulnerability disclosure platforms
- "saas" — SaaS products, cloud platforms (not fitting other categories)
- "ecommerce" — Regular online stores, retail
- "agency" — Marketing agencies, consulting firms (NOT dev studios with distributed teams — use dev_studio)
- "media" — News portals, blogs, content sites, magazines
- "education" — Educational platforms, courses, training programs
- "other" — Anything not fitting above categories
- "unknown" — Cannot determine from available text"""


async def classify_business(domain: str, text: str, mode_id: str, provider: str = "") -> dict | None:
    """Classify what business a website represents using LLM.

    Args:
        provider: explicit provider ("anthropic", "openai", "gemini") or "" for auto.

    Returns dict with: primary_business, industry, business_type,
    is_content_site, confidence. Or None on failure.
    """
    if not provider:
        provider = _get_provider()
    if not provider:
        logger.debug(f"LLM classification skipped for {domain}: no provider configured")
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
        elif provider == "gemini":
            content = await _call_gemini(user_prompt)
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
        model="claude-haiku-4-5-20251001",
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


async def _call_gemini(user_prompt: str) -> str | None:
    """Call Google Gemini API."""
    import google.generativeai as genai

    model = _get_gemini_model()
    if model is None:
        return None

    response = await model.generate_content_async(
        contents=user_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=300,
            response_mime_type="application/json",
        ),
    )

    return response.text.strip()


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
    "esim_telecom": "eSIM / Telecom",
    "dev_studio": "Dev Studio / IT Outsourcing",
    "creator_platform": "Creator / Royalty Platform",
    "bug_bounty": "Bug Bounty / Rewards",
    "saas": "SaaS (General)",
    "ecommerce": "Ecommerce",
    "agency": "Agency / Consulting",
    "media": "Media / News",
    "education": "Education",
    "other": "Other",
    "unknown": "Unknown",
}

NON_TARGET_INDUSTRIES = {"media", "education", "other", "unknown"}


def map_industry_code(code: str) -> str:
    """Map LLM industry code to display name."""
    return INDUSTRY_CODE_MAP.get(code, code)


def is_non_target(classification: dict) -> bool:
    """Check if LLM classification indicates a non-target business."""
    industry = classification.get("industry", "unknown")
    is_content = classification.get("is_content_site", False)
    return industry in NON_TARGET_INDUSTRIES or is_content
