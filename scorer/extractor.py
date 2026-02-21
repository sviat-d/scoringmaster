"""Extract structured evidence from crawled website pages.

Key design: we extract financial action evidence (not just keywords) and tag each
piece with a target need (A/B/C), weight, source URL and page_type.
"""

import re
import logging
from dataclasses import dataclass, field

from scorer.models import CrawledPage, EvidenceItem, SubScores, NEED_A, NEED_B, NEED_C

logger = logging.getLogger(__name__)

SNIPPET_RADIUS = 200  # chars before/after match for snippet


# ─── Signal definitions ───
# Each signal has: id, keywords, weight (1-4), target_need (A/B/C)

@dataclass
class SignalDef:
    signal_id: str
    keywords: list[str]
    weight: int
    target_need: str
    category: str  # inbound / outbound / crossborder / industry / risk


# ── Inbound signals → target A (crypto acceptance need) ──
INBOUND_SIGNALS = [
    SignalDef("crypto_checkout", [
        "crypto payment", "bitcoin payment", "accept crypto", "pay with crypto",
        "crypto checkout", "crypto gateway", "pay with bitcoin",
        "pay with usdt", "pay with usdc",
    ], weight=4, target_need=NEED_A, category="inbound"),
    SignalDef("stablecoin_acceptance", [
        "stablecoin payment", "usdt", "usdc", "accept stablecoin",
        "tether", "stablecoin checkout",
    ], weight=3, target_need=NEED_A, category="inbound"),
    SignalDef("digital_goods", [
        "digital goods", "instant delivery", "digital download", "virtual item",
        "game key", "license key", "digital product", "instant access",
        "digital service", "virtual goods",
    ], weight=3, target_need=NEED_A, category="inbound"),
    SignalDef("chargeback_fraud_pain", [
        "chargeback", "dispute", "fraud prevention", "fraud protection",
        "refund policy", "no refund", "non-refundable", "chargeback rate",
        "friendly fraud", "payment dispute", "CNP fraud",
    ], weight=2, target_need=NEED_A, category="inbound"),
    SignalDef("payment_methods_listed", [
        "payment methods", "accepted payments", "we accept", "pay with",
        "payment options", "supported payments",
    ], weight=2, target_need=NEED_A, category="inbound"),
    SignalDef("checkout_language", [
        "checkout", "add to cart", "buy now", "purchase", "order now",
        "subscribe now", "start free trial",
    ], weight=1, target_need=NEED_A, category="inbound"),
]

# ── Outbound signals → target C (mass payout need) ──
OUTBOUND_SIGNALS = [
    SignalDef("payout_language", [
        "we pay", "get paid", "payouts", "withdraw earnings",
        "withdraw funds", "cash out", "your earnings",
        "payment to affiliates", "payment to partners",
        "commissions paid", "earn money", "your balance",
    ], weight=3, target_need=NEED_C, category="outbound"),
    SignalDef("payout_schedule", [
        "payout schedule", "payment schedule", "payout frequency",
        "paid weekly", "paid monthly", "paid daily",
        "net-7", "net-14", "net-15", "net-30", "net-60",
        "settlement schedule", "settlement terms",
        "bi-weekly payout", "weekly payout", "monthly payout",
        "minimum payout", "payout threshold",
    ], weight=4, target_need=NEED_C, category="outbound"),
    SignalDef("actor_roles", [
        "affiliates", "publishers", "partners", "creators", "sellers",
        "providers", "freelancers", "contractors", "influencers",
        "webmasters", "streamers", "drivers", "workers",
    ], weight=2, target_need=NEED_C, category="outbound"),
    SignalDef("mass_payment_explicit", [
        "mass payment", "mass payout", "bulk payment", "batch payment",
        "disbursement", "bulk transfer", "mass disbursement",
    ], weight=3, target_need=NEED_C, category="outbound"),
    SignalDef("partner_program", [
        "affiliate program", "partner program", "become a partner",
        "referral program", "reseller program", "white label",
        "join our network", "earn commission",
    ], weight=2, target_need=NEED_C, category="outbound"),
]

# ── Cross-border / treasury signals → target B ──
CROSSBORDER_SIGNALS = [
    SignalDef("stablecoin_explicit", [
        "usdt", "usdc", "stablecoin", "tether", "dai",
        "crypto settlement", "stablecoin settlement",
    ], weight=4, target_need=NEED_B, category="crossborder"),
    SignalDef("cross_border_language", [
        "cross-border", "international payment", "international clients",
        "worldwide", "global payment", "multi-currency",
        "worldwide payment", "international transfer",
        "global reach", "operating globally",
    ], weight=3, target_need=NEED_B, category="crossborder"),
    SignalDef("invoice_billing", [
        "invoice", "invoicing", "invoice-based", "retainer",
        "milestone payment", "project-based billing",
        "net terms", "payment terms",
    ], weight=3, target_need=NEED_B, category="crossborder"),
    SignalDef("multi_currency_pricing", [
        "usd", "eur", "gbp", "multi-currency", "currency conversion",
        "local currency", "fx rate", "exchange rate",
    ], weight=2, target_need=NEED_B, category="crossborder"),
    SignalDef("api_integrations", [
        "our api", "api documentation", "api reference", "webhook",
        "sdk", "developer portal", "rest api", "graphql",
        "api key", "api endpoint",
    ], weight=1, target_need=NEED_B, category="crossborder"),
]

ALL_FINANCIAL_SIGNALS = INBOUND_SIGNALS + OUTBOUND_SIGNALS + CROSSBORDER_SIGNALS


# ─── Industry keyword groups (unchanged from v1 for backward compat) ───
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

MIN_INDUSTRY_MATCHES = 3

# Page types where evidence is weaker (not business pages)
WEAK_PAGE_TYPES = {"blog", "faq", "about"}


# ─── Core extraction ───

def _count_keyword(text: str, keyword: str) -> int:
    """Count keyword occurrences using word boundaries."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return len(re.findall(pattern, text))


def _find_keyword_matches(text: str, keyword: str) -> list[int]:
    """Find all match positions for a keyword in text."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return [m.start() for m in re.finditer(pattern, text)]


def _extract_snippet(text: str, position: int, radius: int = SNIPPET_RADIUS) -> str:
    """Extract a snippet of text around a position."""
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _extract_evidence_from_page(
    page: CrawledPage,
    signal: SignalDef,
) -> list[EvidenceItem]:
    """Extract evidence items for a signal from a single page."""
    text_lower = page.text.lower()
    items: list[EvidenceItem] = []
    seen_phrases: set[str] = set()

    for keyword in signal.keywords:
        positions = _find_keyword_matches(text_lower, keyword)
        if not positions:
            continue
        # Only take first occurrence per keyword per page (avoid noise)
        pos = positions[0]
        if keyword in seen_phrases:
            continue
        seen_phrases.add(keyword)

        snippet = _extract_snippet(page.text, pos)

        # Downgrade weight for evidence from weak page types
        weight = signal.weight
        if page.page_type in WEAK_PAGE_TYPES:
            weight = max(1, weight - 1)

        items.append(EvidenceItem(
            signal_id=signal.signal_id,
            matched_phrase=keyword,
            snippet=snippet,
            url=page.url,
            page_type=page.page_type,
            base_weight=weight,
            target_need=signal.target_need,
        ))

    return items


# ─── Evidence gates ───

def _evaluate_gates(evidence: list[EvidenceItem]) -> tuple[bool, bool, bool, str, str, str]:
    """Evaluate minimum evidence gates for A, B, C.

    Returns (gate_a, gate_b, gate_c, reason_a, reason_b, reason_c).
    """
    # Group evidence by target and signal
    a_signals = {e.signal_id for e in evidence if e.target_need == NEED_A}
    b_signals = {e.signal_id for e in evidence if e.target_need == NEED_B}
    c_signals = {e.signal_id for e in evidence if e.target_need == NEED_C}

    a_by_signal = {}
    b_by_signal = {}
    c_by_signal = {}
    for e in evidence:
        bucket = {NEED_A: a_by_signal, NEED_B: b_by_signal, NEED_C: c_by_signal}.get(e.target_need)
        if bucket is not None:
            bucket.setdefault(e.signal_id, []).append(e)

    # Gate A: crypto acceptance needs evidence from business pages (not blog/faq)
    # Evidence from weak pages alone cannot pass the gate
    a_strong = [e for e in evidence if e.target_need == NEED_A and e.page_type not in WEAK_PAGE_TYPES]
    a_strong_signals = {e.signal_id for e in a_strong}

    has_digital_goods = "digital_goods" in a_strong_signals
    has_chargeback = "chargeback_fraud_pain" in a_strong_signals
    has_crypto_checkout = "crypto_checkout" in a_strong_signals or "stablecoin_acceptance" in a_strong_signals
    gate_a = has_crypto_checkout or (has_digital_goods and has_chargeback)
    reason_a = ""
    if not gate_a:
        if a_signals:
            reason_a = "Gate A: need crypto checkout OR (digital goods + chargeback evidence)"
        else:
            reason_a = "Gate A: no inbound payment evidence found"

    # Gate B: treasury needs 2+ of: cross-border, invoice, multi-currency, stablecoin
    # Use non-weak-page evidence only for gate evaluation
    b_strong = [e for e in evidence if e.target_need == NEED_B and e.page_type not in WEAK_PAGE_TYPES]
    b_strong_signals = {e.signal_id for e in b_strong}
    b_factors = sum([
        "cross_border_language" in b_strong_signals,
        "invoice_billing" in b_strong_signals,
        "multi_currency_pricing" in b_strong_signals,
        "stablecoin_explicit" in b_strong_signals,
    ])
    gate_b = b_factors >= 2
    reason_b = ""
    if not gate_b:
        if b_signals:
            reason_b = f"Gate B: only {b_factors}/2 treasury factors (need cross-border + invoice/multi-currency/stablecoin)"
        else:
            reason_b = "Gate B: no cross-border/treasury evidence found"

    # Gate C: mass payout needs outbound flow evidence from non-weak pages
    c_strong = [e for e in evidence if e.target_need == NEED_C and e.page_type not in WEAK_PAGE_TYPES]
    c_strong_signals = {e.signal_id for e in c_strong}
    has_payout_lang = "payout_language" in c_strong_signals
    has_payout_schedule = "payout_schedule" in c_strong_signals
    has_mass_explicit = "mass_payment_explicit" in c_strong_signals
    has_actors = "actor_roles" in c_strong_signals
    has_partner = "partner_program" in c_strong_signals

    # Need payout language + (schedule OR actors) OR explicit mass payment
    gate_c = (
        has_mass_explicit
        or (has_payout_lang and (has_payout_schedule or has_actors))
        or (has_partner and has_payout_lang)
        or (has_partner and has_payout_schedule)
    )
    reason_c = ""
    if not gate_c:
        if c_signals:
            reason_c = "Gate C: need payout flow evidence (payout language + schedule/actors)"
        else:
            reason_c = "Gate C: no outbound payment evidence found"

    return gate_a, gate_b, gate_c, reason_a, reason_b, reason_c


def compute_sub_scores(evidence: list[EvidenceItem]) -> SubScores:
    """Compute gated sub-scores from evidence items."""
    gate_a, gate_b, gate_c, reason_a, reason_b, reason_c = _evaluate_gates(evidence)

    # Sum weights by target need
    raw_a = sum(e.base_weight for e in evidence if e.target_need == NEED_A)
    raw_b = sum(e.base_weight for e in evidence if e.target_need == NEED_B)
    raw_c = sum(e.base_weight for e in evidence if e.target_need == NEED_C)

    # Cap at 20
    CAP_UNGATED = 6
    score_a = min(20, raw_a) if gate_a else min(CAP_UNGATED, raw_a)
    score_b = min(20, raw_b) if gate_b else min(CAP_UNGATED, raw_b)
    score_c = min(20, raw_c) if gate_c else min(CAP_UNGATED, raw_c)

    return SubScores(
        score_a=score_a,
        score_b=score_b,
        score_c=score_c,
        gate_a_passed=gate_a,
        gate_b_passed=gate_b,
        gate_c_passed=gate_c,
        gate_a_reason=reason_a,
        gate_b_reason=reason_b,
        gate_c_reason=reason_c,
    )


# ─── SiteSignals (backward-compatible + evidence) ───

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
        self.content_site_score: int = 0
        self.raw_text_length: int = 0
        # New: evidence-based data
        self.evidence: list[EvidenceItem] = []
        self.sub_scores: SubScores = SubScores()
        self.page_types_found: list[str] = []

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

    def top_evidence(self, n: int = 10) -> list[EvidenceItem]:
        """Return top N evidence items sorted by weight descending."""
        return sorted(self.evidence, key=lambda e: e.base_weight, reverse=True)[:n]


# ── Legacy operational keyword groups (still used for backward-compat scoring) ──
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


def extract_signals(pages: list[CrawledPage] | dict[str, str]) -> SiteSignals:
    """Analyze all crawled pages and extract structured signals.

    Accepts either list[CrawledPage] (new) or dict[str, str] (legacy compat).
    """
    signals = SiteSignals()

    # Handle legacy dict input
    if isinstance(pages, dict):
        pages = [CrawledPage(url=url, text=text) for url, text in pages.items()]

    if not pages:
        return signals

    all_text = "\n".join(p.text for p in pages).lower()
    signals.raw_text_length = len(all_text)
    signals.page_types_found = list({p.page_type for p in pages})

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

    # Content site detection
    content_score = 0
    for kw in CONTENT_SITE_KEYWORDS:
        content_score += _count_keyword(all_text, kw)
    signals.content_site_score = content_score
    if content_score >= 5:
        signals.is_content_site = True

    # Industry detection (keyword fallback)
    threshold = MIN_INDUSTRY_MATCHES * 2 if signals.is_content_site else MIN_INDUSTRY_MATCHES
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        count = 0
        distinct_matches = 0
        for kw in keywords:
            kw_count = _count_keyword(all_text, kw)
            if kw_count > 0:
                distinct_matches += 1
            count += kw_count
        if distinct_matches >= threshold:
            signals.industries[industry] = count

    # Legacy operational signals (backward compat for main score)
    for signal_name, keywords in OPERATIONAL_KEYWORDS.items():
        count = 0
        for kw in keywords:
            count += _count_keyword(all_text, kw)
        if count > 0:
            signals.operational_signals[signal_name] = count

    # ── NEW: Evidence-based extraction (per-page) ──
    for page in pages:
        for signal_def in ALL_FINANCIAL_SIGNALS:
            items = _extract_evidence_from_page(page, signal_def)
            signals.evidence.extend(items)

    # Compute gated sub-scores from evidence
    signals.sub_scores = compute_sub_scores(signals.evidence)

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
