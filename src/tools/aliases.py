"""
Filter aliases — bridge documented enum vocabulary to the values that are
actually populated in the corpus today.

The skill-facing enum in editorial-review SKILL.md advertises broader
filter vocabulary (e.g. ``domain="health"``, ``doc_type="annual-report"``)
than the corpus is tagged with. Rather than narrow the documented vocabulary
or force a corpus retag, we expand a filter value at query time to the set
of payload values that satisfy the intent.

How it works:

- ``expand(field, value)`` returns a list of payload values the caller's
  intent should match. The original value is always first in the list so
  that exact-match entries score normally; aliases come after.
- ``search_passages`` / ``search_terms`` walk the expansion list and run
  one upstream ``semantic_search`` per expanded value, then merge results
  by ``document_id`` keeping the max score. The cost is bounded — alias
  fan-out is small (<=4 values).

Adding a new alias: edit one of the maps below. Aliases are read once at
import; do not mutate at runtime.
"""
from typing import Optional


# Each key maps a documented enum value to the payload value(s) actually
# present in the corpus. The key itself is implicitly also a match target
# (kept in the expansion so future backfill works without code change).
DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    # HIV / KP / PLHIV content is tagged "srhr" in the corpus today; the
    # editorial-review skill exposes "health" as the natural label for it.
    "health": ("srhr",),
}

DOC_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    # No entries are tagged "annual-report" — annual-report style passages
    # live under "report" or "monitoring-report".
    "annual-report": ("report", "monitoring-report"),
}

_FIELD_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "domain": DOMAIN_ALIASES,
    "doc_type": DOC_TYPE_ALIASES,
}


def expand(field: str, value: Optional[str]) -> list[str]:
    """Return the payload values a filter ``field=value`` should match.

    The requested value is always first; aliases follow. Unknown fields and
    values without aliases return ``[value]`` so the caller can treat every
    response uniformly.
    """
    if value is None:
        return []
    aliases = _FIELD_MAP.get(field, {}).get(value, ())
    if not aliases:
        return [value]
    return [value, *aliases]


def has_alias(field: str, value: Optional[str]) -> bool:
    """True iff ``field=value`` expands to more than the literal value."""
    return value is not None and bool(_FIELD_MAP.get(field, {}).get(value))
