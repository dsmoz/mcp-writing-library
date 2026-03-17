"""
Writing style registry for the writing library.

Single source of truth for all valid style values.
Import VALID_STYLES for validation, call list_styles() for MCP tool response.
"""

STYLES: dict[str, dict] = {
    # Structural styles
    "narrative": {
        "category": "structural",
        "description": "Story-led writing. Opens with context or anecdote, builds to a systemic observation.",
    },
    "data-driven": {
        "category": "structural",
        "description": "Evidence-anchored. Leads with statistics or findings, explains causation. Used in M&E reports.",
    },
    "argumentative": {
        "category": "structural",
        "description": "Thesis-driven. States a position, builds the case with evidence. Used in policy briefs.",
    },
    "minimalist": {
        "category": "structural",
        "description": "Short, crisp sentences. No padding. High information density. Used in executive summaries.",
    },
    # Tonal/voice styles
    "formal": {
        "category": "tonal",
        "description": "Institutional register. Third person, passive voice acceptable. UN system documents.",
    },
    "conversational": {
        "category": "tonal",
        "description": "Direct and accessible. Uses 'we' and 'you'. LinkedIn, capacity-building materials.",
    },
    "donor-facing": {
        "category": "tonal",
        "description": "Results-oriented. Outcomes, indicators, sustainability vocabulary. Proposals and progress reports.",
    },
    "advocacy": {
        "category": "tonal",
        "description": "Mobilising language. Rights-based framing, urgency without hyperbole. Position papers.",
    },
    # Author/source styles
    "undp": {
        "category": "source",
        "description": "UNDP HDR register. Discursive transitions, measured language, capability framing. No em-dashes.",
    },
    "global-fund": {
        "category": "source",
        "description": "Global Fund report style. Indicator-heavy, results chain explicit, supply chain vocabulary.",
    },
    "danilo-voice": {
        "category": "source",
        "description": "Danilo da Silva's personal voice. Analytical but direct, Southern African context, rhythm variation.",
    },
    # Anti-pattern styles (negative examples)
    "ai-sounding": {
        "category": "anti-pattern",
        "description": "AI-generated prose patterns. Em-dashes, Furthermore/Moreover, delve, leverage, unprecedented. Negative examples.",
    },
    "bureaucratic": {
        "category": "anti-pattern",
        "description": "Dense nominalisations, passive constructions, filler phrases. Negative examples.",
    },
    "jargon-heavy": {
        "category": "anti-pattern",
        "description": "Stacked sector buzzwords without explanation. Negative examples.",
    },
}

VALID_STYLES: set[str] = set(STYLES.keys())


def list_styles() -> dict:
    """Return all style definitions grouped by category."""
    by_category: dict[str, list] = {}
    for name, info in STYLES.items():
        cat = info["category"]
        by_category.setdefault(cat, [])
        by_category[cat].append({"style": name, "description": info["description"]})
    return {
        "success": True,
        "total": len(STYLES),
        "styles": by_category,
    }
