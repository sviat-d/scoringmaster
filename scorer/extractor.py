"""Extract structured signals from crawled website text."""

import re
import logging

logger = logging.getLogger(__name__)


def _count_keyword(text: str, keyword: str) -> int:
    """Count keyword occurrences using word boundaries to avoid substring matches."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return len(re.findall(pattern, text))


# ─── Industry keyword groups ───
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Affiliate / CPA Marketing": [
        "affiliate network", "cpa network", "cpa marketing",
        "revshare", "offer wall", "affiliate offer",
        "smartlink", "direct advertiser", "media buying",
    ],
    "iGaming & Betting": [
        "casino", "betting", "sportsbook", "odds", "igaming", "gambling",
        "poker", "slots", "wagering", "bookmaker", "sports betting",
    ],
    "Adult / Webcam": [
        "adult content", "adult entertainment", "webcam model", "cam model",
        "xxx", "live cam", "nsfw", "adult site", "cam site",
    ],
    "Hosting / Infrastructure": [
        "hosting", "vps", "dedicated server", "cloud server", "colocation",
        "data center", "offshore hosting", "dmca", "bare metal",
        "managed hosting", "reseller hosting", "web hosting",
    ],
    "VPN / Privacy / Security": [
        "vpn service", "vpn provider", "no-logs policy", "anonymous browsing",
        "end-to-end encrypted", "proxy service", "secure browsing",
        "wireguard", "openvpn", "vpn app", "vpn server",
    ],
    "Freelance / Contractor Platform": [
        "freelance", "freelancer", "contractor", "gig economy",
        "talent marketplace", "remote workers", "independent contractor",
    ],
    "Global Payroll / Payouts": [
        "payroll", "mass payout", "mass payment", "global payout",
        "contractor payment", "payouts", "payment disbursement",
    ],
    "Marketplace": [
        "marketplace", "sellers", "merchants", "vendors",
        "multi-vendor", "e-commerce platform", "merchant platform",
    ],
    "Gaming / Esports / Digital Goods": [
        "gaming", "esports", "skins", "digital goods", "tournament",
        "in-game", "game items", "virtual items", "game keys",
    ],
    "Crypto / Fintech": [
        "crypto", "cryptocurrency", "bitcoin", "ethereum", "stablecoin",
        "usdt", "usdc", "crypto wallet", "defi", "blockchain",
        "payment service provider", "fintech platform", "neobank",
        "crypto exchange", "smart contract", "web3 platform",
    ],
    "High-Risk Ecommerce": [
        "nutra", "supplements", "forex tool", "cbd", "hemp",
        "dietary supplement", "weight loss",
    ],
    "Payment Orchestration / PSP": [
        "payment orchestration", "payment aggregator", "merchant aggregator",
        "billing platform", "subscription billing", "recurring billing",
        "payment api", "checkout api",
    ],
    "Affiliate Tracking Software": [
        "tracking software", "affiliate tracking", "postback",
        "conversion tracking", "attribution platform",
        "traffic tracking", "click tracking",
    ],
    "SaaS (General)": [
        "saas", "software as a service", "cloud platform",
        "our platform", "saas platform", "saas product",
        "free trial", "sign up free", "get started",
    ],
    "eSIM / Telecom": [
        "esim", "e-sim", "travel sim", "data plan", "mobile data",
        "roaming", "travel connectivity", "mvno", "virtual operator",
        "local data", "international data plan",
    ],
    "Dev Studio / IT Outsourcing": [
        "software development", "dev team", "development team",
        "outsourcing", "outstaffing", "dedicated team",
        "nearshore", "offshore development", "remote developers",
        "custom development", "full-stack team", "development agency",
    ],
    "Creator / Royalty Platform": [
        "creator", "creators", "royalty", "royalties", "donate",
        "donation", "tip jar", "support creator", "fan funding",
        "creator payout", "creator economy", "monetize content",
    ],
    "Bug Bounty / Rewards": [
        "bug bounty", "vulnerability", "security reward",
        "responsible disclosure", "bounty program", "hacker",
        "security researcher", "vulnerability disclosure",
    ],
}

# ── Operational signal keywords ──
OPERATIONAL_KEYWORDS: dict[str, list[str]] = {
    "mass_payments": [
        "mass payment", "mass payout", "bulk payment", "batch payment",
        "disbursement",
    ],
    "global_payments": [
        "global payment", "cross-border", "international payment",
        "multi-currency", "worldwide payment",
    ],
    "api_integrations": [
        "our api", "api documentation", "api reference", "webhook",
        "sdk", "developer portal", "rest api", "graphql",
        "api key", "api endpoint",
    ],
    "crypto_payments": [
        "crypto payment", "bitcoin payment", "accept crypto",
        "pay with crypto", "stablecoin payment", "usdt", "usdc",
        "crypto checkout", "crypto gateway",
    ],
    "partners": [
        "partner program", "become a partner", "reseller program",
        "white label", "white-label", "referral program",
    ],
}

# ── Headcount signals ──
HEADCOUNT_PATTERNS = [
    (r"team of (\d+)", lambda m: int(m.group(1))),
    (r"(\d+)\+?\s*(?:employees|team members|people|staff)", lambda m: int(m.group(1))),
    (r"(\d+)\+?\s*(?:человек|сотрудник)", lambda m: int(m.group(1))),
]

CAREERS_KEYWORDS = [
    "careers", "jobs", "we're hiring", "join our team",
    "open positions", "work with us", "job openings",
]

# ── Risk flags ──
RISK_KEYWORDS: dict[str, list[str]] = {
    "iGaming": ["casino", "betting", "gambling", "sportsbook", "igaming", "poker", "slots"],
    "Adult": ["adult", "webcam", "xxx", "nsfw", "porn"],
    "Forex / CFD": ["forex", "cfd", "trading platform", "metatrader", "mt4", "mt5"],
    "High-Risk Ecommerce": ["nutra", "supplements", "cbd", "hemp"],
    "Offshore": ["offshore", "dmca-resistant", "bulletproof hosting", "anonymous hosting"],
}

# ── Hard reject ──
HARD_REJECT_KEYWORDS = [
    "guaranteed returns", "ponzi", "get rich quick", "carding",
    "stolen data", "credit card dump", "fullz", "cvv shop",
]

NON_BUSINESS_KEYWORDS = [
    "personal blog", "my blog", "nonprofit", "ngo",
    "registered charity", "charitable foundation",
    "government agency", "government website",
]

# ── Content / media site detection ──
CONTENT_SITE_KEYWORDS = [
    "published on", "posted on", "author:", "written by",
    "read more", "latest news", "breaking news", "press release",
    "editorial", "journalist", "redakcja", "artykuł",
    "opublikowano", "news archive", "blog post", "article",
]

# Minimum keyword matches required to classify an industry (fallback mode)
MIN_INDUSTRY_MATCHES = 3


class SiteSignals:
    """Structured signals extracted from a website."""

    def __init__(self):
        self.industries: dict[str, int] = {}  # industry -> match count
        self.operational_signals: dict[str, int] = {}
        self.risk_flags: list[str] = []
        self.hard_reject: bool = False
        self.hard_reject_reason: str = ""
        self.non_business: bool = False
        self.headcount_estimate: str = "Unknown"
        self.has_careers_page: bool = False
        self.is_content_site: bool = False
        self.content_site_score: int = 0  # how many content-site signals found
        self.raw_text_length: int = 0

    @property
    def top_industry(self) -> str:
        if not self.industries:
            return "Unknown"
        return max(self.industries, key=self.industries.get)

    @property
    def has_crypto_signals(self) -> bool:
        return self.operational_signals.get("crypto_payments", 0) > 0

    @property
    def has_mass_payment_signals(self) -> bool:
        return self.operational_signals.get("mass_payments", 0) > 0

    @property
    def has_global_payment_signals(self) -> bool:
        return self.operational_signals.get("global_payments", 0) > 0


def extract_signals(pages: dict[str, str]) -> SiteSignals:
    """Analyze all crawled pages and extract structured signals."""
    signals = SiteSignals()

    all_text = "\n".join(pages.values()).lower()
    signals.raw_text_length = len(all_text)

    if not all_text.strip():
        return signals

    # Hard reject check
    for kw in HARD_REJECT_KEYWORDS:
        if _count_keyword(all_text, kw) > 0:
            signals.hard_reject = True
            signals.hard_reject_reason = f"Detected: {kw}"
            return signals

    # Non-business check
    for kw in NON_BUSINESS_KEYWORDS:
        if _count_keyword(all_text, kw) > 0:
            signals.non_business = True

    # Content site detection (before industry, so we can adjust threshold)
    content_score = 0
    for kw in CONTENT_SITE_KEYWORDS:
        content_score += _count_keyword(all_text, kw)
    signals.content_site_score = content_score
    if content_score >= 5:
        signals.is_content_site = True

    # Industry detection (with minimum threshold to reduce false positives)
    # Content sites get a higher threshold since keywords often appear in articles
    threshold = MIN_INDUSTRY_MATCHES * 2 if signals.is_content_site else MIN_INDUSTRY_MATCHES
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        count = 0
        distinct_matches = 0
        for kw in keywords:
            kw_count = _count_keyword(all_text, kw)
            if kw_count > 0:
                distinct_matches += 1
            count += kw_count
        # Require multiple distinct keyword matches to classify
        if distinct_matches >= threshold:
            signals.industries[industry] = count

    # Operational signals
    for signal_name, keywords in OPERATIONAL_KEYWORDS.items():
        count = 0
        for kw in keywords:
            count += _count_keyword(all_text, kw)
        if count > 0:
            signals.operational_signals[signal_name] = count

    # Risk flags
    for flag, keywords in RISK_KEYWORDS.items():
        for kw in keywords:
            if _count_keyword(all_text, kw) > 0:
                if flag not in signals.risk_flags:
                    signals.risk_flags.append(flag)
                break

    # Headcount estimation
    for pattern, extractor in HEADCOUNT_PATTERNS:
        match = re.search(pattern, all_text)
        if match:
            num = extractor(match)
            if num >= 10:
                signals.headcount_estimate = "10+"
            elif num >= 5:
                signals.headcount_estimate = "5-10"
            else:
                signals.headcount_estimate = "1-5"
            break

    # Careers page detection
    for kw in CAREERS_KEYWORDS:
        if _count_keyword(all_text, kw) > 0:
            signals.has_careers_page = True
            if signals.headcount_estimate == "Unknown":
                signals.headcount_estimate = "10+"
            break

    return signals
