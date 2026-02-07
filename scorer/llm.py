"""Optional LLM-based classification using OpenAI API."""

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


SYSTEM_PROMPT = """You are a B2B lead classification expert. Analyze the website text and return a JSON object with these fields:

{
  "industry": "detected industry name",
  "business_model": "product|service|hybrid|unknown",
  "crypto_adoption_likelihood": "high|medium|low",
  "risk_flags": ["list", "of", "flags"],
  "headcount_estimate": "1-5|5-10|10+|unknown",
  "score_1_to_10": 7,
  "confidence": "low|med|high",
  "reason_short": "one sentence summary",
  "reasons_bullets": ["reason 1", "reason 2"]
}

Industries with historically HIGH crypto adoption (boost score even if crypto not mentioned):
- Affiliate networks / CPA marketing
- iGaming & betting
- Adult / webcam platforms
- Hosting providers (VPS, dedicated, offshore)
- VPN / privacy tools
- Freelance & contractor platforms
- Global payroll & payouts
- Marketplaces with international sellers
- Gaming, esports, digital goods
- Crypto-adjacent fintech

Score 8-10: Strong industry + operational signals
Score 6-7: Medium signals or infrastructure play
Score 4-5: Ambiguous, needs manual review
Score 1-3: Irrelevant or reject

Return ONLY valid JSON, no markdown fences."""


async def classify_with_llm(domain: str, text: str, mode: str) -> dict | None:
    """Classify a lead using LLM. Returns structured dict or None on failure."""
    client = _get_client()
    if client is None:
        return None

    # Truncate text to fit context
    truncated = text[:8000]

    mode_context = ""
    if mode == "inxy_leads":
        mode_context = (
            "Context: Scoring leads for Inxy, a crypto payment processor. "
            "Focus on companies that could benefit from crypto payments, "
            "mass payouts, or crypto-fiat conversions."
        )
    elif mode == "founders_pl":
        mode_context = (
            "Context: Identifying product IT founders. "
            "Reject agencies/outsourcing/consulting. "
            "Accept SaaS, platforms, hardware, product studios. "
            "CEO acceptable only if headcount >= 10."
        )

    user_prompt = f"""Domain: {domain}
Mode: {mode}
{mode_context}

Website content:
{truncated}

Classify this lead and return JSON."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON for {domain}: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM classification failed for {domain}: {e}")
        return None


def merge_llm_with_rules(rules_result: dict, llm_result: dict | None) -> dict:
    """Merge LLM output with rules-based result. Rules always override on hard decisions."""
    if llm_result is None:
        return rules_result

    merged = dict(rules_result)

    # LLM can refine these soft fields
    if rules_result.get("industry") == "Unknown" and llm_result.get("industry"):
        merged["industry"] = llm_result["industry"]

    if rules_result.get("business_model") == "Unknown" and llm_result.get("business_model"):
        merged["business_model"] = llm_result["business_model"]

    if rules_result.get("headcount_estimate") == "Unknown" and llm_result.get("headcount_estimate"):
        merged["headcount_estimate"] = llm_result["headcount_estimate"]

    # LLM can enrich reasons
    llm_reasons = llm_result.get("reasons_bullets", [])
    if llm_reasons:
        existing = set(merged.get("reasons_bullets", []))
        for r in llm_reasons:
            if r not in existing:
                merged.setdefault("reasons_bullets", []).append(r)

    # LLM can suggest opener if rules didn't
    if not merged.get("opener") and llm_result.get("reason_short"):
        pass  # Don't auto-generate opener from LLM reason

    # Rules ALWAYS override score and category for hard rejects
    if rules_result.get("score", 0) <= 2:
        return merged  # Keep rules score for rejects

    # For non-rejects, average scores with rules having 2x weight
    llm_score = llm_result.get("score_1_to_10")
    if llm_score and isinstance(llm_score, (int, float)):
        rules_score = rules_result.get("score", 5)
        merged["score"] = round((rules_score * 2 + llm_score) / 3)
        merged["score"] = max(1, min(10, merged["score"]))

        # Recalculate category
        s = merged["score"]
        if s >= 8:
            merged["category"] = "A"
        elif s >= 6:
            merged["category"] = "B"
        elif s >= 4:
            merged["category"] = "C"
        else:
            merged["category"] = "Reject"

    return merged
