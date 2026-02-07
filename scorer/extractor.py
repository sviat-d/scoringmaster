"""Extract structured signals from crawled website text."""

import re
import logging

logger = logging.getLogger(__name__)


# ─── Industry keyword groups ───
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Affiliate / CPA Marketing": [
        "affiliate", "cpa network", "cpa marketing", "performance marketing",
        "revshare", "cost per action", "cost per lead", "affiliate program",
        "publisher", "advertiser", "offer wall", "tracking link",
    ],
    "iGaming & Betting": [
        "casino", "betting", "sportsbook", "odds", "igaming", "gambling",
        "poker", "slots", "wagering", "bookmaker", "sports betting",
    ],
    "Adult / Webcam": [
        "adult", "webcam", "cam model", "content creator", "xxx",
        "live cam", "adult entertainment", "nsfw",
    ],
    "Hosting / Infrastructure": [
        "hosting", "vps", "dedicated server", "cloud server", "colocation",
        "data center", "offshore hosting", "dmca", "bare metal",
        "managed hosting", "reseller hosting", "web hosting",
    ],
    "VPN / Privacy / Security": [
        "vpn", "no-logs", "privacy", "anonymous", "encrypted",
        "proxy", "secure browsing", "wireguard", "openvpn",
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
        "usdt", "usdc", "wallet", "defi", "blockchain", "web3",
        "payment gateway", "psp", "payment service provider",
        "payment processing", "fintech",
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
        "platform", "dashboard", "api",
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
        "api", "integration", "webhook", "sdk", "developer",
        "rest api", "graphql",
    ],
    "crypto_payments": [
        "crypto payment", "bitcoin payment", "accept crypto",
        "pay with crypto", "stablecoin payment", "usdt", "usdc",
        "crypto checkout", "crypto gateway",
    ],
    "partners": [
        "partner program", "partners", "reseller", "white label",
        "white-label", "referral",
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
    "charity", "foundation", "government",
]


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
        if kw in all_text:
            signals.hard_reject = True
            signals.hard_reject_reason = f"Detected: {kw}"
            return signals

    # Non-business check
    for kw in NON_BUSINESS_KEYWORDS:
        if kw in all_text:
            signals.non_business = True

    # Industry detection
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        count = 0
        for kw in keywords:
            occurrences = all_text.count(kw)
            count += occurrences
        if count > 0:
            signals.industries[industry] = count

    # Operational signals
    for signal_name, keywords in OPERATIONAL_KEYWORDS.items():
        count = 0
        for kw in keywords:
            count += all_text.count(kw)
        if count > 0:
            signals.operational_signals[signal_name] = count

    # Risk flags
    for flag, keywords in RISK_KEYWORDS.items():
        for kw in keywords:
            if kw in all_text:
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
        if kw in all_text:
            signals.has_careers_page = True
            if signals.headcount_estimate == "Unknown":
                signals.headcount_estimate = "10+"
            break

    return signals
