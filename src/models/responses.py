"""Permissive Pydantic v2 response models.

Each model expresses the success/error duality of the corresponding
MCP tool in a single schema. Only `success` is required; every other
field is Optional so both branches validate against the same model.

FastMCP reads these annotations to emit `outputSchema` and normalise
structured-content payloads for MCP clients. Direct Python callers
(including the test suite) still receive the raw dict unchanged —
the `@mcp.tool()` decorator does not wrap the function.
"""
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")
    success: bool
    error: Optional[str] = None


class SimilarityResult(_Base):
    """Shared by check_internal_similarity and check_external_similarity."""
    overall_similarity_pct: Optional[float] = None
    verdict: Optional[str] = None
    verdict_threshold_pct: Optional[float] = None
    sentences_checked: Optional[int] = None
    flagged_sentences: Optional[List[Any]] = None
    # External-specific / no-key fallback
    reason: Optional[str] = None
    message: Optional[str] = None
    fallback_instructions: Optional[str] = None
    key_sentences: Optional[List[Any]] = None


class VerifyClaimsResult(_Base):
    overall_evidence_score: Optional[float] = None
    verdict: Optional[str] = None
    total_claims: Optional[int] = None
    verified_count: Optional[int] = None
    claims: Optional[List[Any]] = None
    domain: Optional[str] = None
    note: Optional[str] = None


class RubricScoreResult(_Base):
    framework: Optional[str] = None
    section: Optional[str] = None
    text_length: Optional[int] = None
    criteria_matched: Optional[int] = None
    overall_score: Optional[float] = None
    verdict: Optional[str] = None
    criteria: Optional[List[Any]] = None
    doc_context: Optional[str] = None


class StructureCheckResult(_Base):
    framework: Optional[str] = None
    doc_type: Optional[str] = None
    template_document_id: Optional[str] = None
    total_sections: Optional[int] = None
    required_sections: Optional[int] = None
    present_count: Optional[int] = None
    partial_count: Optional[int] = None
    missing_count: Optional[int] = None
    verdict: Optional[str] = None
    scoring_method: Optional[str] = None
    sections: Optional[List[Any]] = None
    missing_required: Optional[List[str]] = None


class PatternScoreResult(_Base):
    """Union shape across score_writing_patterns modes (ai | semantic-ai | poetry | song | fiction)."""
    mode: Optional[str] = None
    language: Optional[str] = None
    doc_type: Optional[str] = None
    overall_score: Optional[float] = None
    verdict: Optional[str] = None
    threshold: Optional[float] = None
    categories: Optional[Any] = None
    summary: Optional[Any] = None
    word_count: Optional[int] = None
    # ai
    page_equivalent: Optional[float] = None
    # poetry / song
    line_count: Optional[int] = None
    stanza_count: Optional[int] = None
    # fiction
    paragraph_count: Optional[int] = None
    sentence_count: Optional[int] = None
    dialogue_line_count: Optional[int] = None
    # semantic-ai
    likelihood: Optional[float] = None
    ai_mean_similarity: Optional[float] = None
    human_mean_similarity: Optional[float] = None
    ai_sample_count: Optional[int] = None
    human_sample_count: Optional[int] = None
    method: Optional[str] = None
    note: Optional[str] = None


class VocabularyFlagResult(_Base):
    flagged_count: Optional[int] = None
    verdict: Optional[str] = None
    flagged: Optional[List[Any]] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    word_count: Optional[int] = None


class StyleProfileSearchResult(_Base):
    results: Optional[List[Any]] = None
    total: Optional[int] = None
