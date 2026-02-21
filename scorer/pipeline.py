"""Main scoring pipeline: crawl → LLM classify → extract evidence → score → confirm → result."""

import asyncio
import logging

from scorer.crawler import crawl_domain_with_retry, pages_to_text, DEFAULT_MAX_PAGES, EXTENDED_MAX_PAGES
from scorer.extractor import extract_signals
from scorer.modes import get_mode
from scorer import llm

# Import modes to trigger registration
import scorer.mode_inxy  # noqa: F401
import scorer.mode_founders  # noqa: F401

logger = logging.getLogger(__name__)

CONCURRENCY_LIMIT = 3
# Only run flow confirmation for leads scoring at or above this threshold
FLOW_CONFIRM_THRESHOLD = 6


async def score_single(domain: str, mode_id: str, cache: dict, llm_provider: str = "") -> dict:
    """Score a single domain. Returns result dict."""
    domain = domain.strip().lower()

    if not domain or domain in ("n/a", "na", "-", ""):
        return {"score": 0, "category": "Reject", "reason_short": "No domain provided"}

    # Remove protocol if accidentally included
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

    # Check cache
    cache_key = f"{domain}:{mode_id}"
    if cache_key in cache:
        return cache[cache_key]

    # 1) Crawl — returns list[CrawledPage]
    pages = await crawl_domain_with_retry(domain, DEFAULT_MAX_PAGES)

    # 2) If too little content, try crawling more pages
    all_text = pages_to_text(pages)
    if len(all_text) < 1000 and pages:
        extra_pages = await crawl_domain_with_retry(domain, EXTENDED_MAX_PAGES)
        if len(pages_to_text(extra_pages)) > len(all_text):
            pages = extra_pages
            all_text = pages_to_text(pages)

    # 3) LLM business classification (primary method)
    classification = None
    if llm.is_available() and pages:
        classification = await llm.classify_business(domain, all_text, mode_id, provider=llm_provider)
        if classification:
            logger.info(
                f"LLM classified {domain}: "
                f"{classification.get('industry')} / {classification.get('business_type')}"
            )

    # 4) Extract evidence-based signals (per-page, with sub-scores and gates)
    signals = extract_signals(pages)

    # 5) Score using mode with LLM classification + evidence signals
    mode = get_mode(mode_id)
    rules_result = mode.score(signals, domain, classification)

    # 6) LLM flow confirmation for promising leads (score >= threshold)
    if rules_result.score >= FLOW_CONFIRM_THRESHOLD and llm.is_available() and signals.evidence:
        industry_name = rules_result.industry
        flow_conf = await llm.confirm_financial_flows(
            domain, signals.top_evidence(15), industry_name, provider=llm_provider,
        )
        if flow_conf:
            rules_result.flow_confirmation = flow_conf
            # LLM confirmation can adjust confidence
            _apply_flow_confirmation(rules_result, flow_conf)

    result = rules_result.to_dict()
    cache[cache_key] = result
    return result


def _apply_flow_confirmation(result, flow_conf) -> None:
    """Adjust scoring based on LLM flow confirmation."""
    from scorer.models import FlowConfirmation

    reasons = result.reasons_bullets

    # If LLM says no outbound but we scored high on payout signals → downgrade
    if not flow_conf.outbound_obligations and result.sub_scores:
        if result.sub_scores.score_c > 6:
            reasons.append(
                f"LLM flow check: outbound obligations NOT confirmed "
                f"(confidence: {flow_conf.outbound_confidence:.1f})"
            )

    # If LLM confirms flows with high confidence → boost confidence
    max_conf = max(
        flow_conf.outbound_confidence,
        flow_conf.inbound_confidence,
        flow_conf.cross_border_confidence,
    )
    if max_conf >= 0.8 and result.confidence != "High":
        result.confidence = "High"
        reasons.append("LLM flow confirmation: high confidence in financial flows")
    elif max_conf < 0.3 and result.confidence == "High":
        result.confidence = "Med"
        reasons.append("LLM flow confirmation: low confidence, downgraded")

    # Add false positive risks
    if flow_conf.false_positive_risks:
        reasons.append(f"FP risks: {'; '.join(flow_conf.false_positive_risks[:3])}")

    if flow_conf.reasoning_summary:
        reasons.append(f"LLM flow analysis: {flow_conf.reasoning_summary}")


async def score_leads(
    rows: list[dict], domain_col: str, mode_id: str, llm_provider: str = ""
) -> list[dict]:
    """Score all leads from CSV rows. Returns enriched rows."""
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    cache: dict = {}

    async def _process_row(row: dict) -> dict:
        domain = row.get(domain_col, "").strip()
        if not domain:
            row["_result"] = {
                "score": 0,
                "category": "Reject",
                "reason_short": "No domain provided",
            }
            row["_category"] = "Reject"
            return row

        async with semaphore:
            result = await score_single(domain, mode_id, cache, llm_provider=llm_provider)
            row["_result"] = result
            row["_category"] = result.get("category", "Reject")
            return row

    tasks = [_process_row(row) for row in rows]
    enriched = await asyncio.gather(*tasks)

    return list(enriched)
