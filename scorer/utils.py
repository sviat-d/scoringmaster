"""Utility functions for the scorer."""

import re


def clean_domain(raw: str) -> str:
    """Normalize a domain string."""
    domain = raw.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/")[0]
    domain = domain.split("?")[0]
    domain = domain.strip(".")
    return domain


def is_valid_domain(domain: str) -> bool:
    """Basic domain validation."""
    if not domain or len(domain) < 3:
        return False
    if "." not in domain:
        return False
    # Basic pattern: at least one dot, no spaces
    return bool(re.match(r"^[a-z0-9][a-z0-9\-\.]+\.[a-z]{2,}$", domain))
