"""
Single source of truth for controlled vocabulary sets used across all tools.

To add a new value (doc_type, domain, language), edit only this file.
All tools import from here — no local redefinitions.
"""

VALID_DOC_TYPES = {
    # Proposal documents
    "executive-summary",
    "concept-note",
    "policy-brief",
    "full-proposal",
    "eoi",
    # Reports and assessments
    "report",
    "annual-report",
    "monitoring-report",
    "financial-report",
    "assessment",
    "governance-review",
    # Operational documents
    "email",
    "tor",
    "general",
    # Social media
    "facebook-post",
    "linkedin-post",
    "instagram-caption",
}

VALID_DOMAINS = {
    "srhr",
    "governance",
    "climate",
    "general",
    "m-and-e",
    "health",
    "finance",
    "org",
}

# Passage and correction language values (binary: source language of text)
VALID_LANGUAGES = {"en", "pt"}

# Term language values (includes "both" for cross-language terms)
VALID_LANGUAGES_TERMS = {"en", "pt", "both"}

# Channel vocabulary for style profiles — the publishing surface a profile targets.
# Unknown values warn but do not block saving (same pattern as style labels).
VALID_CHANNELS = {
    # Social media
    "linkedin",
    "facebook",
    "instagram",
    "twitter",
    "whatsapp",
    "tiktok",
    # Long-form digital
    "blog",
    "newsletter",
    "substack",
    # Professional written
    "email",
    "report",
    "proposal",
    "executive-summary",
    "tor",
    "press-release",
    "presentation",
    # Catch-all
    "general",
}
