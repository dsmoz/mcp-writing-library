"""
Contextual Retrieval - Hybrid Metadata + LLM Context Generation

This module implements Anthropic's Contextual Retrieval methodology
for improving RAG search accuracy by 35-49%.

Two approaches are supported:

1. **Metadata-based context** (FAST - default):
   - Generates context from document metadata (title, type, tags, date, entities)
   - No LLM required, instant generation
   - Zero hallucination risk - purely factual
   - Based on mcp-zotero-qdrant's proven approach

2. **LLM-based context** (SLOW - optional):
   - Uses LLM to generate semantic context per chunk
   - ~2-5 seconds per chunk with local LLM
   - Better for documents without good metadata
   - Only use when metadata is insufficient

Usage:
    from kbase.core.context_generator import (
        build_metadata_context,  # Fast, no LLM
        ContextGenerator,        # LLM-based (optional)
        init_llm_client,
    )

    # Fast approach: Metadata-based context (recommended)
    context = build_metadata_context(
        title="Angola VNR 2024",
        doc_type="report",
        tags=["SDG", "health", "climate"],
        creators=["Ministry of Planning"],
        date="2024",
        chunk_info={"chunk_index": 0, "total_chunks": 10}
    )

    # Slow approach: LLM-based context (only if needed)
    init_llm_client(base_url="http://localhost:1234/v1", ...)
    generator = ContextGenerator()
    contextualized_chunks = generator.generate_contexts(...)
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# FAST APPROACH: Metadata-based Context (No LLM required)
# =============================================================================

def build_metadata_context(
    title: str = None,
    doc_type: str = None,
    tags: List[str] = None,
    creators: List[str] = None,
    date: str = None,
    abstract: str = None,
    chunk_info: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
) -> str:
    """
    Build context prefix from document metadata WITHOUT using LLM.

    This is the FAST approach - instant generation, zero hallucination risk.
    Based on mcp-zotero-qdrant's proven methodology.

    Args:
        title: Document title
        doc_type: Document type (report, article, book, etc.)
        tags: List of topic tags
        creators: List of author/creator names
        date: Publication date
        abstract: Document abstract (for topic extraction if no tags)
        chunk_info: Dict with chunk_index, total_chunks, start_page, end_page
        metadata: Additional metadata dict (fallback for any field)

    Returns:
        Context string to prepend to chunk before embedding

    Example output:
        "This content is from a report titled 'Angola VNR 2024' about SDG progress,
        health improvements, climate vulnerabilities. Key entities: Ministry of Planning,
        United Nations. Time period: 2024. Location: section 3 of 15, pages 45-48."
    """
    # Extract from metadata dict if individual params not provided
    if metadata:
        title = title or metadata.get("title")
        doc_type = doc_type or metadata.get("doc_type") or metadata.get("itemType") or metadata.get("source_type")
        tags = tags or _extract_tags_from_metadata(metadata)
        creators = creators or _extract_creators_from_metadata(metadata)
        date = date or metadata.get("date")
        abstract = abstract or metadata.get("abstractNote") or metadata.get("abstract")

    parts = []

    # Part 1: Source type and title
    type_desc = _get_type_description(doc_type) if doc_type else "document"
    parts.append(f"This content is from a {type_desc}")

    if title and title != "Untitled Document":
        parts.append(f"titled '{title}'")

    # Part 2: Topics (from tags or abstract)
    topics = tags[:5] if tags else _extract_topics_from_abstract(abstract)
    if topics:
        parts.append(f"about {', '.join(topics)}")

    # Part 3: Key entities (from creators)
    if creators:
        entity_list = creators[:3]  # Max 3 entities
        parts.append(f"Key entities: {', '.join(entity_list)}")

    # Part 4: Time period
    if date:
        time_str = _extract_years_from_date(date)
        if time_str:
            parts.append(f"Time period: {time_str}")

    # Part 5: Location within document
    if chunk_info:
        location = _build_location_string(chunk_info)
        if location:
            parts.append(f"Location: {location}")

    # Join parts with appropriate punctuation
    if not parts:
        return ""

    # Build sentence
    context = parts[0]
    for i, part in enumerate(parts[1:], 1):
        if part.startswith("titled"):
            context += f" {part}"
        elif part.startswith("about"):
            context += f". {part}"
        elif part.startswith("Key entities"):
            context += f". {part}"
        elif part.startswith("Time period"):
            context += f". {part}"
        elif part.startswith("Location"):
            context += f". {part}"
        else:
            context += f". {part}"

    if not context.endswith("."):
        context += "."

    return context


def _get_type_description(doc_type: str) -> str:
    """Map document types to readable descriptions."""
    type_map = {
        # Zotero types
        "journalArticle": "journal article",
        "book": "book",
        "bookSection": "book chapter",
        "report": "report",
        "thesis": "thesis",
        "conferencePaper": "conference paper",
        "webpage": "webpage",
        "document": "document",
        "manuscript": "manuscript",
        "presentation": "presentation",
        "dataset": "dataset",
        "attachment": "attachment",
        # Common types
        "pdf": "PDF document",
        "docx": "Word document",
        "xlsx": "spreadsheet",
        "md": "markdown document",
        "file": "file",
        "manual": "manually added content",
        "web": "web content",
    }
    return type_map.get(doc_type, doc_type if doc_type else "document")


def _extract_tags_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Extract tags from various metadata formats."""
    tags = metadata.get("tags", [])
    if not tags:
        return []

    # Handle Zotero tag format: [{"tag": "name"}, ...]
    if tags and isinstance(tags[0], dict):
        return [tag.get("tag", "") for tag in tags if tag.get("tag")]

    # Handle simple list format: ["tag1", "tag2", ...]
    return [str(tag) for tag in tags if tag]


def _extract_creators_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Extract creator names from various metadata formats."""
    creators = metadata.get("creators", [])
    if not creators:
        # Try alternative fields
        authors = metadata.get("authors", [])
        if authors:
            return authors if isinstance(authors[0], str) else []
        return []

    names = []
    for creator in creators[:5]:  # Max 5 creators
        if isinstance(creator, dict):
            # Zotero format: {"firstName": "John", "lastName": "Smith"}
            if creator.get("name"):
                names.append(creator["name"])
            elif creator.get("lastName"):
                name = creator["lastName"]
                if creator.get("firstName"):
                    name = f"{creator['firstName']} {name}"
                names.append(name)
        elif isinstance(creator, str):
            names.append(creator)

    return names


def _extract_topics_from_abstract(abstract: str) -> List[str]:
    """Extract topic keywords from abstract text."""
    if not abstract:
        return []

    # Get first sentence
    first_sentence = abstract.split('.')[0] if '.' in abstract else abstract[:200]

    # Find capitalized phrases (proper nouns, key concepts)
    # Pattern: 2-4 capitalized words
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', first_sentence)

    # Filter out common words
    common = {"The", "This", "That", "These", "Those", "In", "On", "At", "For", "With"}
    topics = [t for t in capitalized if t not in common and len(t) > 3]

    return topics[:3]  # Max 3 topics


def _extract_years_from_date(date: str) -> str:
    """Extract year(s) from date string."""
    if not date:
        return ""

    years = re.findall(r'\b(?:19|20)\d{2}\b', date)

    if years:
        if len(years) == 1:
            return years[0]
        elif len(years) >= 2:
            return f"{years[0]}-{years[-1]}"

    return date  # Return original if can't parse


def _build_location_string(chunk_info: Dict[str, Any]) -> str:
    """Build location string from chunk info."""
    if not chunk_info:
        return ""

    parts = []

    # Page numbers
    start_page = chunk_info.get("start_page")
    end_page = chunk_info.get("end_page")
    if start_page and end_page:
        if start_page == end_page:
            parts.append(f"page {start_page}")
        else:
            parts.append(f"pages {start_page}-{end_page}")
    elif start_page:
        parts.append(f"page {start_page}")

    # Chunk position
    chunk_idx = chunk_info.get("chunk_index", 0)
    total_chunks = chunk_info.get("total_chunks", 1)
    if total_chunks > 1:
        parts.append(f"section {chunk_idx + 1} of {total_chunks}")

    return ", ".join(parts)


def validate_metadata_quality(
    title: str = None,
    doc_type: str = None,
    tags: List[str] = None,
    creators: List[str] = None,
    date: str = None,
    abstract: str = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Validate if metadata has sufficient information for good context generation.

    Returns:
        dict: {
            "is_valid": bool,      # True if metadata is sufficient
            "missing_fields": list, # Fields that are missing
            "quality_score": float, # 0.0 to 1.0
            "recommendation": str,  # "use_metadata" or "use_llm"
        }
    """
    # Extract from metadata dict if needed
    if metadata:
        title = title or metadata.get("title")
        doc_type = doc_type or metadata.get("doc_type") or metadata.get("itemType")
        tags = tags or _extract_tags_from_metadata(metadata)
        creators = creators or _extract_creators_from_metadata(metadata)
        date = date or metadata.get("date")
        abstract = abstract or metadata.get("abstractNote") or metadata.get("abstract")

    missing = []

    if not title or title == "Untitled Document":
        missing.append("title")

    if not tags or len(tags) == 0:
        missing.append("tags")

    if not creators or len(creators) == 0:
        missing.append("creators")

    if not date:
        missing.append("date")

    if not abstract:
        missing.append("abstract")

    # Calculate quality score
    total_fields = 5
    found_fields = total_fields - len(missing)
    quality_score = found_fields / total_fields

    # Valid if title exists and at least 2 other fields
    is_valid = "title" not in missing and len(missing) <= 3

    return {
        "is_valid": is_valid,
        "missing_fields": missing,
        "quality_score": quality_score,
        "recommendation": "use_metadata" if is_valid else "use_llm",
    }


def ensure_document_metadata(
    content: str,
    title: str = None,
    existing_metadata: Dict[str, Any] = None,
    generate_if_poor: bool = True,
    max_content_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Ensure document has good metadata for context generation.

    This function should be called when adding a document to:
    1. Validate existing metadata quality
    2. Generate metadata using LLM if poor and LLM is available
    3. Return enriched metadata ready for database storage

    Args:
        content: Document text content
        title: Document title (if known)
        existing_metadata: Any existing metadata dict
        generate_if_poor: If True, generate metadata when quality is poor
        max_content_chars: Max chars to send to LLM for metadata generation

    Returns:
        dict: {
            "metadata": dict,         # Enriched metadata ready for storage
            "was_generated": bool,    # True if metadata was LLM-generated
            "quality_validation": dict,  # Result of quality validation
        }

    Example:
        result = ensure_document_metadata(
            content=doc_content,
            title="My Document",
            existing_metadata={"tags": ["topic1"]},
        )
        # Store result["metadata"] in document's doc_metadata field
    """
    # Build metadata from existing + title
    metadata = dict(existing_metadata) if existing_metadata else {}
    if title and title != "Untitled":
        metadata["title"] = title

    # Validate quality
    quality = validate_metadata_quality(metadata=metadata)

    result = {
        "metadata": metadata,
        "was_generated": False,
        "quality_validation": quality,
    }

    # If metadata is good, return as-is
    if quality["is_valid"]:
        logger.info(
            "Metadata quality is good, no generation needed",
            quality_score=quality["quality_score"],
        )
        return result

    # If poor and generation requested, try to generate
    if generate_if_poor and content:
        if is_contextual_retrieval_available():
            try:
                logger.info(
                    "Metadata quality is poor, generating with LLM",
                    missing_fields=quality["missing_fields"],
                )
                generated = generate_metadata_from_content(
                    content=content,
                    existing_metadata=metadata,
                    max_content_chars=max_content_chars,
                )

                # Merge generated into metadata (generated values take precedence for missing fields)
                for key, value in generated.items():
                    if key.startswith("_"):
                        # Keep generation markers
                        metadata[key] = value
                    elif key not in metadata or not metadata[key]:
                        # Fill in missing fields
                        metadata[key] = value

                # Map generated fields to standard metadata fields
                if "key_entities" in generated and generated["key_entities"]:
                    metadata["creators"] = generated["key_entities"]
                if "summary" in generated and generated["summary"]:
                    metadata["abstract"] = generated["summary"]

                result["metadata"] = metadata
                result["was_generated"] = True

                logger.info(
                    "Metadata generated successfully",
                    title=metadata.get("title"),
                    tags=metadata.get("tags"),
                )

            except Exception as e:
                logger.warning(
                    "Failed to generate metadata, using original",
                    error=str(e),
                )
        else:
            logger.debug(
                "LLM not available for metadata generation",
                missing_fields=quality["missing_fields"],
            )

    return result


# =============================================================================
# METADATA GENERATION (LLM-based - only when needed)
# =============================================================================

METADATA_GENERATION_PROMPT = """Analyze this document and extract structured metadata.

<document>
{document}
</document>

Generate metadata in the following JSON format:
{{
    "title": "descriptive title if document has no clear title",
    "summary": "1-2 sentence summary of the document's main topic and purpose",
    "tags": ["topic1", "topic2", "topic3"],  // 3-5 relevant topic tags
    "doc_type": "report|article|memo|proposal|assessment|budget|plan|other",
    "key_entities": ["entity1", "entity2"],  // organizations, people, places mentioned
    "language": "en|pt|other"
}}

IMPORTANT:
- Extract the ACTUAL title from the document if present, don't invent one
- Tags should be specific topics, not generic words
- Key entities are proper nouns: organization names, people, places
- If information is not present, use null for that field

Respond ONLY with the JSON, no other text."""


def generate_metadata_from_content(
    content: str,
    existing_metadata: Dict[str, Any] = None,
    max_content_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Generate document metadata using LLM when metadata is poor/missing.

    This function:
    1. Takes document content (first ~8000 chars)
    2. Uses LLM to extract title, tags, summary, entities
    3. Returns structured metadata to be stored in database

    Args:
        content: Document text content
        existing_metadata: Any existing metadata to preserve
        max_content_chars: Max chars to send to LLM (default 8000)

    Returns:
        dict: Generated metadata including:
            - title: Document title
            - summary: Short summary
            - tags: List of topic tags
            - doc_type: Document type
            - key_entities: List of entities
            - language: Detected language
            - _generated: True (flag indicating LLM-generated)

    Raises:
        RuntimeError: If LLM client not initialized
    """
    if not is_contextual_retrieval_available():
        raise RuntimeError(
            "LLM client not initialized. Call init_llm_client() first."
        )

    # Truncate content
    doc_sample = content[:max_content_chars]
    if len(content) > max_content_chars:
        doc_sample += "\n\n[Document truncated...]"

    # Build prompt
    prompt = METADATA_GENERATION_PROMPT.format(document=doc_sample)

    try:
        response = _llm_client.chat.completions.create(
            model=_llm_config["model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic for metadata
            max_tokens=500,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import json

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json") or line.startswith("```"):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        generated = json.loads(response_text)

        # Merge with existing metadata (preserve existing values)
        if existing_metadata:
            for key, value in existing_metadata.items():
                if key not in generated or not generated[key]:
                    generated[key] = value

        # Mark as generated
        generated["_generated"] = True
        generated["_generation_model"] = _llm_config.get("model", "unknown")

        logger.info(
            "Generated metadata from content",
            title=generated.get("title"),
            tags=generated.get("tags"),
            doc_type=generated.get("doc_type"),
        )

        return generated

    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse LLM response as JSON",
            error=str(e),
            response=response_text[:200] if response_text else None,
        )
        # Return basic metadata
        return {
            "title": existing_metadata.get("title") if existing_metadata else None,
            "summary": None,
            "tags": [],
            "doc_type": "document",
            "key_entities": [],
            "_generated": False,
            "_error": str(e),
        }

    except Exception as e:
        logger.error(
            "Failed to generate metadata",
            error=str(e),
        )
        raise


def generate_document_context(
    content: str,
    max_content_chars: int = 12000,
) -> str:
    """
    Generate a SINGLE document-level context using LLM.

    Unlike per-chunk context generation, this generates ONE context
    for the entire document, which is then prepended to ALL chunks.

    This is much faster than per-chunk generation:
    - 1 LLM call per document instead of N calls per chunk
    - ~5 seconds total instead of ~5 seconds × N chunks

    Args:
        content: Full document text
        max_content_chars: Max chars to send to LLM

    Returns:
        Context string describing the document

    Raises:
        RuntimeError: If LLM client not initialized
    """
    if not is_contextual_retrieval_available():
        raise RuntimeError(
            "LLM client not initialized. Call init_llm_client() first."
        )

    # Truncate content
    doc_sample = content[:max_content_chars]
    if len(content) > max_content_chars:
        doc_sample += "\n\n[Document truncated...]"

    prompt = f"""Analyze this document and write a brief context description (2-3 sentences) that captures:
1. What type of document this is
2. The main topic and purpose
3. Key entities mentioned (organizations, people, places)

<document>
{doc_sample}
</document>

Write the context in the SAME LANGUAGE as the document.
Respond ONLY with the context description, nothing else."""

    try:
        response = _llm_client.chat.completions.create(
            model=_llm_config["model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200,
        )

        context = response.choices[0].message.content.strip()

        logger.info(
            "Generated document-level context",
            context_preview=context[:100],
        )

        return context

    except Exception as e:
        logger.error(
            "Failed to generate document context",
            error=str(e),
        )
        raise


# =============================================================================
# LLM CLIENT (for optional LLM-based operations)
# =============================================================================

# Global LLM client
_llm_client = None
_llm_config = {
    "base_url": None,
    "api_key": None,
    "model": None,
    "temperature": 0.0,
    "max_tokens": 500,
    "timeout": 60,
}


def init_llm_client(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    timeout: int = 60,
) -> None:
    """
    Initialize the LLM client for context generation.

    Args:
        base_url: OpenAI-compatible API base URL (e.g., http://localhost:1234/v1)
        api_key: API key (for LM Studio, any string works)
        model: Model name (e.g., google/gemma-3n-e4b)
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Max tokens for context response
        timeout: Request timeout in seconds
    """
    global _llm_client, _llm_config

    _llm_config.update({
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    })

    try:
        from openai import OpenAI

        _llm_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        logger.info(
            "LLM client initialized for contextual retrieval",
            base_url=base_url,
            model=model,
        )

    except ImportError:
        logger.warning(
            "openai package not installed, contextual retrieval disabled. "
            "Install with: pip install openai"
        )
        _llm_client = None


def get_llm_client():
    """Get the global LLM client."""
    return _llm_client


def is_contextual_retrieval_available() -> bool:
    """Check if contextual retrieval is available."""
    return _llm_client is not None


# Delimiter for separating context from content
CONTEXT_DELIMITER = "\n---\n"


@dataclass
class ContextualizedChunk:
    """A text chunk with its generated context prefix."""

    original_text: str
    context: str
    contextualized_text: str  # context + CONTEXT_DELIMITER + original_text
    chunk_index: int

    @property
    def text_for_embedding(self) -> str:
        """Text to be embedded (context + chunk)."""
        return self.contextualized_text


# Anthropic's recommended prompt for contextual retrieval
CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# Multilingual version (Portuguese/English) - improved language matching
CONTEXT_PROMPT_TEMPLATE_MULTILINGUAL = """<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Write ONE short sentence describing what this chunk is about and where it fits in the document.

IMPORTANT: Respond in the SAME LANGUAGE as the document.
- Se o documento estiver em Português, responda em Português.
- If the document is in English, respond in English.

Answer only with ONE sentence, nothing else."""


class ContextGenerator:
    """
    Generates contextual prefixes for document chunks using LLM.

    This implements Anthropic's Contextual Retrieval methodology:
    1. For each chunk, send the full document + chunk to LLM
    2. LLM generates a short context explaining where chunk fits
    3. Context is prepended to chunk BEFORE embedding
    4. The combined text is embedded, capturing semantic context

    Performance: ~2-5 seconds per chunk with local LLM (Gemma 3n 4B)
    """

    def __init__(
        self,
        prompt_template: str = None,
        max_document_chars: int = 12000,
        multilingual: bool = True,
    ):
        """
        Initialize the context generator.

        Args:
            prompt_template: Custom prompt template (uses default if None)
            max_document_chars: Max chars of document to include in prompt.
                               Default 12000 (~3000 tokens) works with 8K context models.
                               For larger models, increase this value.
            multilingual: Use multilingual prompt template
        """
        if prompt_template:
            self.prompt_template = prompt_template
        elif multilingual:
            self.prompt_template = CONTEXT_PROMPT_TEMPLATE_MULTILINGUAL
        else:
            self.prompt_template = CONTEXT_PROMPT_TEMPLATE

        self.max_document_chars = max_document_chars

    def generate_context_for_chunk(
        self,
        full_document: str,
        chunk_text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate context for a single chunk.

        Args:
            full_document: Full document text
            chunk_text: Text of the chunk
            title: Optional document title
            metadata: Optional metadata

        Returns:
            Context string to prepend to chunk
        """
        if not is_contextual_retrieval_available():
            logger.debug("LLM not available, using metadata-based context")
            return self._build_metadata_context(title, metadata)

        # Truncate document if too long
        doc_for_prompt = full_document[:self.max_document_chars]
        if len(full_document) > self.max_document_chars:
            doc_for_prompt += "\n\n[Document truncated...]"

        # Build prompt
        prompt = self.prompt_template.format(
            document=doc_for_prompt,
            chunk=chunk_text,
        )

        try:
            response = _llm_client.chat.completions.create(
                model=_llm_config["model"],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=_llm_config["temperature"],
                max_tokens=_llm_config["max_tokens"],
            )

            context = response.choices[0].message.content.strip()
            return context

        except Exception as e:
            logger.warning(
                "Failed to generate context, using metadata fallback",
                error=str(e),
            )
            return self._build_metadata_context(title, metadata)

    def generate_contexts(
        self,
        full_document: str,
        chunks: List[str],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ContextualizedChunk]:
        """
        Generate contexts for multiple chunks.

        Args:
            full_document: Full document text
            chunks: List of chunk texts
            title: Optional document title
            metadata: Optional metadata

        Returns:
            List of ContextualizedChunk objects
        """
        results = []

        for i, chunk_text in enumerate(chunks):
            logger.debug(
                "Generating context for chunk",
                chunk_index=i,
                total_chunks=len(chunks),
            )

            context = self.generate_context_for_chunk(
                full_document=full_document,
                chunk_text=chunk_text,
                title=title,
                metadata=metadata,
            )

            contextualized = ContextualizedChunk(
                original_text=chunk_text,
                context=context,
                contextualized_text=f"{context}{CONTEXT_DELIMITER}{chunk_text}",
                chunk_index=i,
            )

            results.append(contextualized)

        logger.info(
            "Generated contexts for all chunks",
            num_chunks=len(chunks),
            llm_available=is_contextual_retrieval_available(),
        )

        return results

    def _build_metadata_context(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a simple metadata-based context when LLM is unavailable.

        This is the fallback approach (similar to zotero-qdrant).
        """
        parts = []

        if title:
            parts.append(f"This content is from a document titled '{title}'.")

        if metadata:
            source = metadata.get("source")
            if source:
                parts.append(f"Source: {source}.")

            storage_type = metadata.get("storage_type")
            if storage_type:
                parts.append(f"Type: {storage_type}.")

        if not parts:
            return ""

        return " ".join(parts)


def generate_context_for_indexing(
    full_document: str,
    chunk_text: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    multilingual: bool = True,
) -> str:
    """
    Convenience function to generate context for a single chunk.

    Args:
        full_document: Full document text
        chunk_text: Text of the chunk
        title: Optional document title
        metadata: Optional metadata
        multilingual: Use multilingual prompt

    Returns:
        Contextualized text (context + chunk)
    """
    generator = ContextGenerator(multilingual=multilingual)
    context = generator.generate_context_for_chunk(
        full_document=full_document,
        chunk_text=chunk_text,
        title=title,
        metadata=metadata,
    )

    if context:
        return f"{context}{CONTEXT_DELIMITER}{chunk_text}"
    return chunk_text


# =============================================================================
# TAGGING & CHUNK CONTEXTUALIZATION (LLM-based, simple API)
# =============================================================================

_TAG_PROMPT = """\
You are a document tagging assistant. Given a document title and an excerpt, extract 3 to 7 concise keyword tags that best describe the document topic, geography, sector, and document type. Return ONLY a comma-separated list of lowercase tags with no explanation. Use hyphens for multi-word tags (e.g. key-populations, annual-report, mozambique).

Title: {title}

Excerpt:
{excerpt}

Tags:"""

_CHUNK_CONTEXT_PROMPT = """\
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

In one sentence, situate this chunk within the document to improve search retrieval. Start with "This chunk" and say nothing else."""


def auto_tag_document(
    title: str,
    excerpt: str,
    client=None,
    model: str = None,
) -> list:
    """Auto-generate keyword tags for a document using the LLM.

    Sends the document title + an excerpt and asks the model for 3-7
    lowercase, hyphenated tags covering topic, geography, sector, and
    document type.

    Args:
        title:   Document title
        excerpt: First ~500 chars of extracted text
        client:  OpenAI-compatible client. Uses the global LLM client
                 (init_llm_client) if None.
        model:   Model name. Uses the global config if None.

    Returns:
        List of lowercase tag strings (e.g. ["hiv", "mozambique"]).
        Returns [] if LLM is unavailable or returns nothing.
    """
    import re as _re

    llm = client or _llm_client
    if llm is None:
        logger.debug("auto_tag_document: LLM not available, returning empty tags")
        return []

    prompt = _TAG_PROMPT.format(
        title=title or "Untitled",
        excerpt=excerpt[:500] or "(no text)",
    )
    _model = model or _llm_config.get("model") or "default"

    try:
        response = llm.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("auto_tag_document: LLM call failed", error=str(e))
        return []

    tags = []
    for part in raw.split(","):
        tag = _re.sub(r"[^a-z0-9\-]", "", part.strip().lower().replace(" ", "-"))
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:7]


def contextualize_chunk(
    document: str,
    chunk: str,
    client=None,
    model: str = None,
) -> str:
    """Generate a one-sentence situational description for a chunk.

    Implements Anthropic's contextual retrieval technique. Returns a
    sentence starting with "This chunk" that situates the chunk within
    the full document. Falls back to empty string if LLM is unavailable.

    Args:
        document: Full document text (first 8000 chars used)
        chunk:    Chunk text to contextualize
        client:   OpenAI-compatible client. Uses global LLM client if None.
        model:    Model name. Uses the global config if None.

    Returns:
        Context string, or "" on failure/unavailability.
    """
    llm = client or _llm_client
    if llm is None:
        logger.debug("contextualize_chunk: LLM not available")
        return ""

    prompt = _CHUNK_CONTEXT_PROMPT.format(
        document=document[:8000],
        chunk=chunk,
    )
    _model = model or _llm_config.get("model") or "default"

    try:
        response = llm.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()
        for line in content.splitlines():
            line = line.strip()
            if line:
                return line
        return ""
    except Exception as e:
        logger.warning("contextualize_chunk: LLM call failed", error=str(e))
        return ""


__all__ = [
    # Fast approach (recommended)
    "build_metadata_context",
    "validate_metadata_quality",
    # High-level metadata enrichment (use at document creation)
    "ensure_document_metadata",
    # Metadata generation (LLM-based, only when needed)
    "generate_metadata_from_content",
    "generate_document_context",
    # LLM per-chunk approach (slow, optional)
    "ContextGenerator",
    "ContextualizedChunk",
    "init_llm_client",
    "get_llm_client",
    "is_contextual_retrieval_available",
    "generate_context_for_indexing",
    # Simple LLM tagging & contextualization
    "auto_tag_document",
    "contextualize_chunk",
    # Constants
    "CONTEXT_PROMPT_TEMPLATE",
    "CONTEXT_PROMPT_TEMPLATE_MULTILINGUAL",
    "CONTEXT_DELIMITER",
]
