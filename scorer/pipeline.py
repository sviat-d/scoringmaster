"""Main scoring pipeline: crawl → LLM classify → extract signals → score → result."""

import asyncio
import logging

from scorer.crawler import crawl_domain_with_retry, DEFAULT_MAX_PAGES, EXTENDED_MAX_PAGES
from scorer.extractor import extract_signals
from scorer.modes import get_mode
from scorer import llm

# Import modes to trigger registration
import scorer.mode_inxy  # noqa: F401
import scorer.mode_founders  # noqa: F401

logger = logging.getLogger(__name__)

CONCURRENCY_LIMIT = 3


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

    # 1) Crawl
    pages = await crawl_domain_with_retry(domain, DEFAULT_MAX_PAGES)

    # 2) If too little content, try crawling more pages
    all_text = "\n".join(pages.values())
    if len(all_text) < 1000 and pages:
        extra_pages = await crawl_domain_with_retry(domain, EXTENDED_MAX_PAGES)
        if len("\n".join(extra_pages.values())) > len(all_text):
            pages = extra_pages
            all_text = "\n".join(pages.values())

    # 3) LLM business classification (primary method)
    classification = None
    if llm.is_available() and pages:
        classification = await llm.classify_business(domain, all_text, mode_id, provider=llm_provider)
        if classification:
            logger.info(
                f"LLM classified {domain}: "
                f"{classification.get('industry')} / {classification.get('business_type')}"
            )

    # 4) Extract keyword-based signals (operational signals, risk flags, headcount)
    signals = extract_signals(pages)

    # 5) Score using mode with LLM classification + keyword signals
    mode = get_mode(mode_id)
    rules_result = mode.score(signals, domain, classification)
    result = rules_result.to_dict()

    cache[cache_key] = result
    return result


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
