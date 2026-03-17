"""
MCP Writing Library Server — FastMCP tool definitions.

Tools:
    search_passages          — semantic search for exemplary writing passages
    add_passage              — store a new exemplary passage
    list_styles              — list all valid writing style labels
    search_terms             — semantic search in terminology dictionary
    add_term                 — add a new terminology entry
    get_library_stats        — collection point counts
    setup_collections        — create/verify Qdrant collections (admin)
    check_internal_similarity — detect similarity against library passages
    check_external_similarity — detect similarity against web content (Tavily)
    score_external_similarity — score pre-fetched search results for similarity
"""
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("writing-library")


@mcp.tool()
def search_passages(
    query: str,
    doc_type: Optional[str] = None,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    style: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Search for exemplary writing passages by semantic similarity.

    Use this to find model paragraphs when drafting or reviewing documents.
    Returns passages ranked by relevance with quality notes explaining what makes each one effective.

    Args:
        query: What you need (e.g. "executive summary opening about health equity")
        doc_type: Filter by type: executive-summary|concept-note|policy-brief|report|email|general
        language: Filter by language: en|pt
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        style: Filter by style: narrative|data-driven|argumentative|minimalist|
               formal|conversational|donor-facing|advocacy|
               undp|global-fund|danilo-voice|
               ai-sounding|bureaucratic|jargon-heavy
               Call list_styles() to see descriptions.
        top_k: Number of results (default 5, max 20)

    Returns:
        List of matching passages with scores, quality notes, tags, and style labels
    """
    from src.tools.passages import search_passages as _search
    return _search(query=query, doc_type=doc_type, language=language, domain=domain, style=style, top_k=top_k)


@mcp.tool()
def add_passage(
    text: str,
    doc_type: str = "general",
    language: str = "en",
    domain: str = "general",
    quality_notes: str = "",
    tags: Optional[List[str]] = None,
    source: str = "manual",
    style: Optional[List[str]] = None,
) -> dict:
    """
    Store an exemplary writing passage in the library.

    Use this when you produce a passage you are proud of, or to seed the library
    with models from reference documents (UNDP HDR, Global Fund reports, etc.).

    Args:
        text: The passage (one or more paragraphs)
        doc_type: Context: executive-summary|concept-note|policy-brief|report|email|general
        language: Language: en|pt
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        quality_notes: What makes this passage good (helps future retrieval)
        tags: Labels for retrieval e.g. ["discursive", "findings", "contrast"]
        source: Where the passage came from (e.g. "undp-hdr-2024", "manual")
        style: Style labels e.g. ["narrative", "donor-facing"].
               Call list_styles() to see all valid values.
               Unknown values are warned but do not block the save.

    Returns:
        document_id, chunks_created, and warnings on success
    """
    from src.tools.passages import add_passage as _add
    return _add(
        text=text, doc_type=doc_type, language=language, domain=domain,
        quality_notes=quality_notes, tags=tags or [], source=source,
        style=style or [],
    )


@mcp.tool()
def list_styles() -> dict:
    """
    Return all writing style labels with descriptions, grouped by category.

    Use before calling add_passage or search_passages with a style argument
    to see what values are valid.

    Categories:
        structural   — narrative, data-driven, argumentative, minimalist
        tonal        — formal, conversational, donor-facing, advocacy
        source       — undp, global-fund, danilo-voice
        anti-pattern — ai-sounding, bureaucratic, jargon-heavy

    Returns:
        styles grouped by category with descriptions, total count
    """
    from src.tools.styles import list_styles as _list
    return _list()


@mcp.tool()
def search_terms(
    query: str,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 8,
) -> dict:
    """
    Search the terminology dictionary for preferred consultant vocabulary.

    Use this when choosing between terms, or when reviewing text for inappropriate language.
    Returns preferred terms, what to avoid, and why — with good/bad usage examples.

    Args:
        query: What you're looking for (e.g. "person living with HIV", "leverage", "unprecedented")
        domain: Filter by domain: srhr|governance|climate|general|m-and-e
        language: Filter by language: en|pt
        top_k: Number of results (default 8)

    Returns:
        List of terminology entries with preferred/avoid pairs and examples
    """
    from src.tools.terms import search_terms as _search
    return _search(query=query, domain=domain, language=language, top_k=top_k)


@mcp.tool()
def add_term(
    preferred: str,
    avoid: str = "",
    domain: str = "general",
    language: str = "en",
    why: str = "",
    example_bad: str = "",
    example_good: str = "",
) -> dict:
    """
    Add a terminology entry to the dictionary.

    Use this to codify vocabulary preferences — consultant language, UNDP standards,
    people-first terminology, or sector-specific expressions.

    Args:
        preferred: The term to use (e.g. "rights-holder", "key populations")
        avoid: Term to avoid (e.g. "victim", "vulnerable groups")
        domain: Thematic area: srhr|governance|climate|general|m-and-e
        language: Language: en|pt|both
        why: Reason for preference (e.g. "deficit framing; UNDP 2024 standard")
        example_bad: Example of poor usage
        example_good: Example of correct usage

    Returns:
        document_id on success
    """
    from src.tools.terms import add_term as _add
    return _add(
        preferred=preferred, avoid=avoid, domain=domain, language=language,
        why=why, example_bad=example_bad, example_good=example_good,
    )


@mcp.tool()
def get_library_stats() -> dict:
    """
    Return point counts for both Qdrant collections.

    Use this to verify the library is populated and working.

    Returns:
        Stats for writing_passages and writing_terms collections
    """
    from src.tools.collections import get_stats
    return get_stats()


@mcp.tool()
def setup_collections() -> dict:
    """
    Create or verify Qdrant collections for the writing library.

    Run this once on first setup, or to verify collections exist after a Qdrant restart.

    Returns:
        Status for each collection (created|already_exists|error)
    """
    from src.tools.collections import setup_collections as _setup
    return _setup()


@mcp.tool()
def check_internal_similarity(
    text: str,
    threshold: float = 0.85,
    top_k_per_sentence: int = 3,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Check if a passage is too similar to content already in the writing library.

    Splits the text into sentences and searches each against the writing_passages
    collection. Useful for deduplication before adding new passages.

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold to flag a sentence (default 0.85)
        top_k_per_sentence: Max matches to retrieve per sentence (default 3)
        verdict_threshold_pct: % of sentences flagged to trigger overall "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), and list of flagged sentences
        with their matching library entries (document_id, score, excerpt, source)
    """
    from src.tools.plagiarism import check_internal_similarity as _check
    return _check(
        text=text,
        threshold=threshold,
        top_k_per_sentence=top_k_per_sentence,
        verdict_threshold_pct=verdict_threshold_pct,
    )


@mcp.tool()
def check_external_similarity(
    text: str,
    threshold: float = 0.75,
    max_sentences: int = 3,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Search the web for content similar to the given text using the Tavily API.

    Requires TAVILY_API_KEY in the environment. If not set, returns key sentences
    and fallback instructions for manual search via mcp__MCP_DOCKER__tavily_search,
    followed by a call to score_external_similarity().

    Args:
        text: The passage to check
        threshold: Cosine similarity threshold to flag a match (default 0.75)
        max_sentences: Number of key sentences to search (default 3, reduces API usage)
        verdict_threshold_pct: % of sentences flagged to trigger overall "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), and list of flagged sentences
        with their web sources (url, title, score, snippet).
        If TAVILY_API_KEY is missing: success=False with key_sentences and fallback_instructions.
    """
    from src.tools.plagiarism import check_external_similarity as _check
    return _check(
        text=text,
        threshold=threshold,
        max_sentences=max_sentences,
        verdict_threshold_pct=verdict_threshold_pct,
    )


@mcp.tool()
def score_external_similarity(
    text: str,
    search_results: list,
    threshold: float = 0.75,
    verdict_threshold_pct: float = 30.0,
) -> dict:
    """
    Score similarity between text and pre-fetched web search results.

    Use this after manually searching with mcp__MCP_DOCKER__tavily_search
    when TAVILY_API_KEY is not configured. Pass search results directly.

    Args:
        text: The original text to check
        search_results: List of {url, content, title} dicts from Tavily search
        threshold: Cosine similarity threshold (default 0.75)
        verdict_threshold_pct: % of sentences flagged to trigger "flagged" verdict (default 30)

    Returns:
        overall_similarity_pct, verdict (clean|flagged), and flagged sentences with web sources
    """
    from src.tools.plagiarism import score_external_similarity as _score
    return _score(
        text=text,
        search_results=search_results,
        threshold=threshold,
        verdict_threshold_pct=verdict_threshold_pct,
    )
