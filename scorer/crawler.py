"""Website crawler: fetches pages via httpx, classifies page types by content."""

import os
import re
import asyncio
import logging

import httpx
from bs4 import BeautifulSoup

from scorer.models import CrawledPage

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
    "/partners", "/affiliate", "/affiliates",
    "/payouts", "/payments", "/developers",
    "/terms", "/terms-of-service", "/refund-policy",
    "/faq", "/help",
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

# ── Page type classification (by content, not URL) ──
PAGE_TYPE_SIGNALS: dict[str, list[str]] = {
    "pricing": [
        "pricing", "plans", "subscription", "per month", "/mo", "free trial",
        "enterprise plan", "pro plan", "starter plan", "pricing table",
        "billed annually", "billed monthly", "compare plans",
    ],
    "checkout": [
        "checkout", "payment method", "pay now", "add to cart", "buy now",
        "order summary", "credit card", "pay with", "accepted payments",
        "payment options", "secure payment",
    ],
    "terms": [
        "terms of service", "terms and conditions", "refund policy",
        "privacy policy", "acceptable use", "legal notice", "disclaimer",
        "cancellation policy", "chargeback policy", "dispute resolution",
    ],
    "affiliate": [
        "affiliate program", "partner program", "become a partner",
        "referral program", "earn commission", "join our network",
        "affiliate dashboard", "partner dashboard", "commission rate",
    ],
    "payouts": [
        "payout", "withdrawal", "earnings", "payment schedule",
        "payout method", "cash out", "payment terms", "settlement",
        "get paid", "your balance", "minimum payout", "payout frequency",
    ],
    "api_docs": [
        "api documentation", "api reference", "developer docs", "sdk",
        "webhook", "endpoint", "rest api", "getting started",
        "authentication", "api key", "code examples",
    ],
    "faq": [
        "faq", "frequently asked", "help center", "support center",
        "knowledge base", "how to", "troubleshoot", "common questions",
    ],
    "about": [
        "about us", "our story", "our team", "our mission", "who we are",
        "company overview", "founded in", "our values", "leadership team",
    ],
    "careers": [
        "careers", "open positions", "we're hiring", "join our team",
        "work with us", "job openings", "current openings",
    ],
    "blog": [
        "blog", "latest posts", "recent articles", "published on",
        "read more", "posted by", "author:", "written by",
    ],
}


def _classify_page_type(title: str, headings: list[str], text: str) -> str:
    """Classify page type by content analysis (not URL)."""
    # Combine title + headings + first 1000 chars of text for classification
    signal_text = f"{title} {' '.join(headings)} {text[:1000]}".lower()

    scores: dict[str, int] = {}
    for page_type, keywords in PAGE_TYPE_SIGNALS.items():
        score = 0
        for kw in keywords:
            if kw in signal_text:
                score += 1
        if score > 0:
            scores[page_type] = score

    if not scores:
        return "homepage" if not title or "home" in title.lower() else "other"

    best = max(scores, key=scores.get)
    # Require at least 2 matches for confident classification
    if scores[best] >= 2:
        return best
    return "homepage" if not title or "home" in title.lower() else "other"


async def crawl_domain(domain: str, max_pages: int = DEFAULT_MAX_PAGES) -> list[CrawledPage]:
    """Crawl a domain and return list of CrawledPage objects."""
    base_url = f"https://{domain}"
    results: list[CrawledPage] = []

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
        headers=HEADERS,
        verify=False,
    ) as client:
        # 1) Homepage
        homepage = await _fetch_and_extract(client, base_url)
        if homepage:
            results.append(homepage)
        else:
            # Try http fallback
            http_url = f"http://{domain}"
            homepage = await _fetch_and_extract(client, http_url)
            if homepage:
                results.append(homepage)
                base_url = http_url

        if not results:
            return results

        # 2) Discover links from homepage, prioritize about/product pages
        pages_left = max_pages - len(results)
        if pages_left <= 0:
            return results

        seen_urls = {p.url for p in results}
        candidate_urls = _prioritize_urls(base_url)

        for url in candidate_urls:
            if pages_left <= 0:
                break
            if url in seen_urls:
                continue
            page = await _fetch_and_extract(client, url)
            if page:
                results.append(page)
                seen_urls.add(url)
                pages_left -= 1

    return results


def _prioritize_urls(base_url: str) -> list[str]:
    """Generate prioritized list of URLs to crawl."""
    urls = []
    # About pages first
    for path in ABOUT_PATHS:
        urls.append(base_url.rstrip("/") + path)
    # Then product/feature pages
    for path in PRODUCT_PATHS:
        urls.append(base_url.rstrip("/") + path)
    return urls


async def _fetch_and_extract(client: httpx.AsyncClient, url: str) -> CrawledPage | None:
    """Fetch URL and extract visible text with metadata."""
    try:
        resp = await client.get(url)
        if resp.status_code >= 400:
            return None
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None
        return _extract_page(url, resp.text)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return None


def _extract_page(url: str, html: str) -> CrawledPage:
    """Extract visible text, title, headings from HTML and classify page type."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract headings
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"], limit=20):
        h_text = tag.get_text(strip=True)
        if h_text:
            headings.append(h_text)

    # Remove script, style, nav, footer, header noise
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text[:MAX_CHARS_PER_PAGE]

    # Classify page type by content
    page_type = _classify_page_type(title, headings, text)

    return CrawledPage(
        url=url,
        text=text,
        title=title,
        headings=headings,
        page_type=page_type,
    )


# ── Backward-compatible helpers ──

def pages_to_dict(pages: list[CrawledPage]) -> dict[str, str]:
    """Convert CrawledPage list to old-style {url: text} dict."""
    return {p.url: p.text for p in pages}


def pages_to_text(pages: list[CrawledPage]) -> str:
    """Concatenate all page texts into one string."""
    return "\n".join(p.text for p in pages)


async def crawl_domain_with_retry(
    domain: str, max_pages: int = DEFAULT_MAX_PAGES
) -> list[CrawledPage]:
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
        return []
