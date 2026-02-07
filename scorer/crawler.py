"""Website crawler: fetches pages via httpx, optional Playwright fallback."""

import os
import re
import asyncio
import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAX_CHARS_PER_PAGE = 12_000
DEFAULT_MAX_PAGES = 3
EXTENDED_MAX_PAGES = 5
REQUEST_TIMEOUT = 15.0

ABOUT_PATHS = ["/about", "/about-us", "/company", "/team", "/our-team"]
PRODUCT_PATHS = [
    "/product", "/products", "/platform", "/pricing",
    "/api", "/integrations", "/careers", "/jobs",
    "/solutions", "/services", "/features",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def crawl_domain(domain: str, max_pages: int = DEFAULT_MAX_PAGES) -> dict[str, str]:
    """Crawl a domain and return {url: extracted_text} dict."""
    base_url = f"https://{domain}"
    results: dict[str, str] = {}

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
        headers=HEADERS,
        verify=False,
    ) as client:
        # 1) Homepage
        homepage_text = await _fetch_and_extract(client, base_url)
        if homepage_text:
            results[base_url] = homepage_text
        else:
            # Try http fallback
            http_url = f"http://{domain}"
            homepage_text = await _fetch_and_extract(client, http_url)
            if homepage_text:
                results[http_url] = homepage_text
                base_url = http_url

        if not results:
            return results

        # 2) Discover links from homepage, prioritize about/product pages
        pages_left = max_pages - len(results)
        if pages_left <= 0:
            return results

        candidate_urls = _prioritize_urls(base_url, homepage_text or "")

        for url in candidate_urls:
            if pages_left <= 0:
                break
            if url in results:
                continue
            text = await _fetch_and_extract(client, url)
            if text:
                results[url] = text
                pages_left -= 1

    return results


def _prioritize_urls(base_url: str, homepage_text: str) -> list[str]:
    """Generate prioritized list of URLs to crawl."""
    urls = []
    # About pages first
    for path in ABOUT_PATHS:
        urls.append(base_url.rstrip("/") + path)
    # Then product/feature pages
    for path in PRODUCT_PATHS:
        urls.append(base_url.rstrip("/") + path)
    return urls


async def _fetch_and_extract(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch URL and extract visible text."""
    try:
        resp = await client.get(url)
        if resp.status_code >= 400:
            return None
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None
        return _extract_text(resp.text)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return None


def _extract_text(html: str) -> str:
    """Extract visible text from HTML, clean and truncate."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, nav, footer, header noise
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text[:MAX_CHARS_PER_PAGE]


async def crawl_domain_with_retry(
    domain: str, max_pages: int = DEFAULT_MAX_PAGES
) -> dict[str, str]:
    """Crawl with a single retry on failure."""
    try:
        result = await crawl_domain(domain, max_pages)
        if result:
            return result
    except Exception:
        pass

    await asyncio.sleep(1)
    try:
        return await crawl_domain(domain, max_pages)
    except Exception as e:
        logger.warning(f"Crawl failed for {domain}: {e}")
        return {}
