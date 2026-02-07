"""Main scoring pipeline: crawl → extract → score → optional LLM → result."""

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


async def score_single(domain: str, mode_id: str, cache: dict) -> dict:
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

    # 2) Extract signals
    signals = extract_signals(pages)

    # 3) If confidence would be low, crawl more pages
    if signals.raw_text_length < 1000 and pages:
        extra_pages = await crawl_domain_with_retry(domain, EXTENDED_MAX_PAGES)
        if len(extra_pages) > len(pages):
            pages = extra_pages
            signals = extract_signals(pages)

    # 4) Rules-based scoring
    mode = get_mode(mode_id)
    rules_result = mode.score(signals, domain)
    result = rules_result.to_dict()

    # 5) Optional LLM enrichment
    if llm.is_available() and pages:
        all_text = "\n".join(pages.values())
        llm_result = await llm.classify_with_llm(domain, all_text, mode_id)
        result = llm.merge_llm_with_rules(result, llm_result)

    cache[cache_key] = result
    return result


async def score_leads(
    rows: list[dict], domain_col: str, mode_id: str
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
            result = await score_single(domain, mode_id, cache)
            row["_result"] = result
            row["_category"] = result.get("category", "Reject")
            return row

    tasks = [_process_row(row) for row in rows]
    enriched = await asyncio.gather(*tasks)

    return list(enriched)
