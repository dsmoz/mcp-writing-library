"""
Evidence hallucination detection tools.

Provides claim extraction, corroboration search against Zotero and Cerebellum
knowledge bases, and evidence density scoring without external search.
"""
import os
import re
from pathlib import Path
from typing import List, Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Claim-detection regex patterns
# ---------------------------------------------------------------------------

_PATTERN_NUMBERS = re.compile(
    r'\b\d{2,}[\d,\.]*\s*(%|percent|per cent|pct)?\b',
    re.IGNORECASE,
)

_PATTERN_EPISTEMIC = re.compile(
    r'\b(shows? that|indicates? that|demonstrates? that|evidence suggests?'
    r'|data (reveals?|shows?)|according to'
    r'|research (shows?|finds?|suggests?))\b'
    # Portuguese epistemic verbs
    r'|os dados (mostram|indicam|revelam|sugerem)'
    r'|a (evidência|investigação|pesquisa) (sugere|indica|mostra|revela)'
    r'|segundo (os dados|o relatório|o MISAU|a OMS|a ONUSIDA|os estudos)'
    r'|de acordo com'
    r'|estudos (mostram|indicam|sugerem|revelam)',
    re.IGNORECASE,
)

_PATTERN_CITATION_APA = re.compile(
    r'\([A-Z][\w\s,\.]+\d{4}\)',
)

_PATTERN_CITATION_NUMERIC = re.compile(
    r'\[\d+\]',
)

_PATTERN_PREVALENCE = re.compile(
    r'\b(in Mozambique|in Angola|in SADC|among key populations?'
    r'|HIV prevalence|maternal mortality|under-five mortality'
    r'|adolescent|PLHIV'
    # Portuguese prevalence/country terms
    r'|prevalência do (VIH|SIDA|HIV|malária|tuberculose)'
    r'|mortalidade (materna|infantil|neonatal)'
    r'|entre (populações-chave|adolescentes|jovens|mulheres)'
    r'|em Moçambique|em Angola|na África Austral|na SADC)\b',
    re.IGNORECASE,
)

_ALL_CLAIM_PATTERNS = [
    _PATTERN_NUMBERS,
    _PATTERN_EPISTEMIC,
    _PATTERN_CITATION_APA,
    _PATTERN_CITATION_NUMERIC,
    _PATTERN_PREVALENCE,
]

_CITATION_PATTERNS = [_PATTERN_CITATION_APA, _PATTERN_CITATION_NUMERIC]

# ---------------------------------------------------------------------------
# Domain-specific claim-detection patterns
# ---------------------------------------------------------------------------

_DOMAIN_PATTERNS = {
    "finance": re.compile(
        r'\b(budget|cost|funding|expenditure|revenue|margin|deficit|surplus'
        r'|USD|EUR|MZN|GBP|million|billion|thousand'
        r'|orçamento|custo|financiamento|despesa|receita|défice|excedente)\b',
        re.IGNORECASE,
    ),
    "governance": re.compile(
        r'\b(corruption|accountability|transparency|participation|oversight'
        r'|governance|electoral|civil society|rule of law'
        r'|corrupção|responsabilização|transparência|participação|fiscalização'
        r'|governação|governança|sociedade civil|estado de direito)\b',
        re.IGNORECASE,
    ),
    "climate": re.compile(
        r'\b(emissions?|carbon|greenhouse|biodiversity|deforestation|rainfall'
        r'|temperature|climate|drought|flood|adaptation|mitigation'
        r'|emissões?|carbono|efeito estufa|biodiversidade|desmatamento|chuva'
        r'|temperatura|clima|seca|inundação|adaptação|mitigação)\b',
        re.IGNORECASE,
    ),
    "m-and-e": re.compile(
        r'\b(indicator|target|baseline|milestone|output|outcome|result'
        r'|monitoring|evaluation|assessment|KPI|benchmark'
        r'|indicador|meta|linha de base|marco|produto|resultado'
        r'|monitoramento|monitorização|avaliação|supervisão)\b',
        re.IGNORECASE,
    ),
    "org": re.compile(
        r'\b(staff turnover|employee satisfaction|organizational capacity'
        r'|governance structure|human resources'
        r'|rotatividade de pessoal|satisfação dos funcionários|capacidade organizacional'
        r'|estrutura de governação|recursos humanos)\b',
        re.IGNORECASE,
    ),
    "health": re.compile(
        r'\b(HIV|PLHIV|prevalence|incidence|mortality|morbidity|treatment'
        r'|antiretroviral|malaria|tuberculosis|TB|nutrition|maternal'
        r'|child|adolescent|SRHR|reproductive|sexual health'
        r'|VIH|SIDA|prevalência|incidência|mortalidade|tratamento'
        r'|antirretroviral|malária|tuberculose|nutrição|saúde materna'
        r'|criança|adolescente|saúde reprodutiva|saúde sexual)\b',
        re.IGNORECASE,
    ),
}


def _get_claim_patterns(domain: str) -> List[re.Pattern]:
    """Return the list of claim-detection patterns for the given domain.

    If domain is in _DOMAIN_PATTERNS, the domain-specific pattern is appended
    to the base patterns. Unknown domains fall back to the base patterns only.
    """
    if domain in _DOMAIN_PATTERNS:
        return _ALL_CLAIM_PATTERNS + [_DOMAIN_PATTERNS[domain]]
    return _ALL_CLAIM_PATTERNS


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering out very short fragments."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= 20]


def _is_claim_sentence(sentence: str, patterns: List[re.Pattern] = None) -> bool:
    """Return True if the sentence matches any claim-bearing pattern.

    Args:
        sentence: The sentence to evaluate.
        patterns: List of compiled regex patterns to use. Defaults to
                  _ALL_CLAIM_PATTERNS for backward compatibility.
    """
    if patterns is None:
        patterns = _ALL_CLAIM_PATTERNS
    return any(p.search(sentence) for p in patterns)


def _has_number(sentence: str) -> bool:
    """Return True if the sentence contains a number or percentage claim."""
    return bool(_PATTERN_NUMBERS.search(sentence))


def _has_citation(sentence: str) -> bool:
    """Return True if the sentence contains an explicit citation marker."""
    return any(p.search(sentence) for p in _CITATION_PATTERNS)


# ---------------------------------------------------------------------------
# Lazy cross-server imports with graceful degradation
# ---------------------------------------------------------------------------

def _search_zotero(query: str, top_k: int) -> List[dict]:
    """Search the Zotero MCP server via HTTP. Returns empty list on any failure."""
    base_url = os.getenv("ZOTERO_MCP_URL", "").rstrip("/")
    if not base_url:
        return []

    collection_name = os.getenv("ZOTERO_QDRANT_COLLECTION", "zotero_hybrid")
    code = (
        f"from src.qdrant.search import SemanticSearch\n"
        f"result = SemanticSearch(query={query!r}, top_k={top_k}, "
        f"collection_name={collection_name!r})\nresult"
    )
    headers = {"Content-Type": "application/json"}
    token = os.getenv("ZOTERO_MCP_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(f"{base_url}/run", json={"code": code}, headers=headers)
            response.raise_for_status()
            results = response.json()
    except Exception as exc:
        logger.warning("Zotero HTTP search failed — returning empty", error=str(exc))
        return []

    try:
        return [
            {
                "title": r.get("title", ""),
                "citekey": r.get("citekey"),
                "score": float(r.get("score", 0)),
                "source_type": "zotero",
                "excerpt": (r.get("text", "") or "")[:200],
            }
            for r in results
        ]
    except Exception as exc:
        logger.warning("Zotero result normalisation failed", error=str(exc))
        return []


def _search_cerebellum(query: str, top_k: int) -> List[dict]:
    """Search the Cerebellum MCP server via HTTP. Returns empty list on any failure."""
    base_url = os.getenv("CEREBELLUM_MCP_URL", "").rstrip("/")
    if not base_url:
        return []

    code = (
        f"from tools.search import global_search\n"
        f"result = global_search(query={query!r}, limit={top_k})\nresult"
    )
    headers = {"Content-Type": "application/json"}
    token = os.getenv("CEREBELLUM_MCP_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(f"{base_url}/run", json={"code": code}, headers=headers)
            response.raise_for_status()
            results = response.json()
    except Exception as exc:
        logger.warning("Cerebellum HTTP search failed — returning empty", error=str(exc))
        return []

    try:
        return [
            {
                "title": r.get("title", ""),
                "citekey": None,
                "score": float(r.get("score", 0)),
                "source_type": "cerebellum",
                "excerpt": (r.get("text", "") or "")[:200],
            }
            for r in results
        ]
    except Exception as exc:
        logger.warning("Cerebellum result normalisation failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Local research file helpers
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
_CHUNK_WORDS = 300


def _read_research_files(paths: List[str]) -> List[dict]:
    """Read local research files and return text chunks.

    Accepts a mix of file paths and directory paths. Directories are scanned
    one level deep for supported extensions (.md, .txt, .pdf).

    Returns a list of dicts: [{text, source_file}].
    """
    chunks = []
    for raw_path in paths:
        p = Path(raw_path).expanduser()
        if p.is_dir():
            candidates = [f for f in p.iterdir() if f.suffix.lower() in _SUPPORTED_EXTENSIONS]
        elif p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
            candidates = [p]
        else:
            logger.warning("research_paths: skipping unreadable path", path=str(p))
            continue

        for file_path in candidates:
            try:
                if file_path.suffix.lower() == ".pdf":
                    content = _read_pdf(file_path)
                else:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.warning("research_paths: failed to read file", path=str(file_path), error=str(exc))
                continue

            words = content.split()
            for i in range(0, max(1, len(words)), _CHUNK_WORDS):
                chunk_text = " ".join(words[i: i + _CHUNK_WORDS])
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "source_file": str(file_path)})

    return chunks


def _read_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using pypdf if available, else empty string."""
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        logger.warning("pypdf not installed — skipping PDF", path=str(file_path))
        return ""
    except Exception as exc:
        logger.warning("PDF read failed", path=str(file_path), error=str(exc))
        return ""


def _search_local_files(query: str, chunks: List[dict], top_k: int) -> List[dict]:
    """Score chunks against the query using keyword overlap (offline, no embedding).

    Returns normalised source dicts sorted by descending score, capped at top_k.
    """
    if not chunks:
        return []

    query_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
    if not query_tokens:
        return []

    scored = []
    for chunk in chunks:
        chunk_tokens = set(re.findall(r'\b\w{3,}\b', chunk["text"].lower()))
        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            continue
        score = round(overlap / len(query_tokens), 4)
        scored.append({
            "title": Path(chunk["source_file"]).name,
            "citekey": None,
            "score": score,
            "source_type": "local",
            "source_file": chunk["source_file"],
            "excerpt": chunk["text"][:200],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claims(
    text: str,
    domain: str = "general",
    top_k_per_claim: int = 3,
    corroboration_threshold: float = 0.65,
    research_paths: Optional[List[str]] = None,
) -> dict:
    """
    Extract claim-bearing sentences from text and verify each against local
    research files (optional), Zotero, and Cerebellum knowledge bases.

    Search order per claim: local files → Zotero → Cerebellum.
    A strong local match (score ≥ corroboration_threshold) short-circuits
    the remote searches for that claim.

    Args:
        text: The text to analyse for unsupported claims.
        domain: Thematic domain for domain-specific claim pattern augmentation.
                Valid values: "general", "finance", "governance", "climate",
                "m-and-e", "org", "health". Unknown values fall back to general patterns.
        top_k_per_claim: Number of sources to retrieve per claim sentence.
        corroboration_threshold: Minimum score from any source to mark a
                                 claim as "verified" (default 0.65).
        research_paths: Optional list of file paths or directory paths to local
                        research documents (.md, .txt, .pdf). Read at call time;
                        never indexed. Searched first, before Zotero/Cerebellum.

    Returns:
        dict with overall_evidence_score, verdict, per-claim results, and
        ghost_stat flags for unverified numeric claims.

        Possible verdict values:
          - "evidenced": >= 80% of claims are verified.
          - "mixed": 40–79% of claims are verified.
          - "unverified": < 40% of claims are verified.
          - "no_claims_detected": no claim-bearing sentences were found;
            no evidence verification was performed (overall_evidence_score
            is None in this case).
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    top_k_per_claim = min(top_k_per_claim, 10)
    claim_patterns = _get_claim_patterns(domain)

    sentences = _split_sentences(text)
    claim_sentences = [s for s in sentences if _is_claim_sentence(s, patterns=claim_patterns)]

    if not claim_sentences:
        return {
            "success": True,
            "overall_evidence_score": None,
            "verdict": "no_claims_detected",
            "total_claims": 0,
            "verified_count": 0,
            "claims": [],
            "domain": domain,
            "note": "No claim-bearing sentences detected. No evidence verification was performed.",
        }

    local_chunks = _read_research_files(research_paths) if research_paths else []

    claims_output = []
    verified_count = 0

    for sentence in claim_sentences:
        all_sources = []

        local_sources = _search_local_files(sentence, local_chunks, top_k=top_k_per_claim)
        all_sources.extend(local_sources)
        local_best = max((s["score"] for s in local_sources), default=0.0)

        if local_best < corroboration_threshold:
            zotero_sources = _search_zotero(sentence, top_k=top_k_per_claim)
            cerebellum_sources = _search_cerebellum(sentence, top_k=top_k_per_claim)
            all_sources.extend(zotero_sources)
            all_sources.extend(cerebellum_sources)

        best_score = max((s["score"] for s in all_sources), default=0.0)

        if best_score >= corroboration_threshold:
            verdict = "verified"
            ghost_stat = False
            verified_count += 1
        else:
            verdict = "unverified"
            ghost_stat = _has_number(sentence)

        claims_output.append({
            "sentence": sentence,
            "verdict": verdict,
            "ghost_stat": ghost_stat,
            "sources": all_sources,
        })

    total_claims = len(claim_sentences)
    overall_score = round(verified_count / total_claims, 4) if total_claims else 1.0

    if overall_score >= 0.8:
        overall_verdict = "evidenced"
    elif overall_score >= 0.4:
        overall_verdict = "mixed"
    else:
        overall_verdict = "unverified"

    return {
        "success": True,
        "overall_evidence_score": overall_score,
        "verdict": overall_verdict,
        "total_claims": total_claims,
        "verified_count": verified_count,
        "claims": claims_output,
        "domain": domain,
    }


def score_evidence_density(text: str, domain: str = "general") -> dict:
    """
    Analyse the ratio of evidenced sentences to total sentences without
    calling any external search APIs.

    Counts claim-bearing sentences (using the same regex patterns as
    verify_claims) and sentences with explicit citation markers.

    Args:
        text: The text to score.
        domain: Thematic domain for domain-specific claim pattern augmentation.
                Valid values: "general", "finance", "governance", "climate",
                "m-and-e", "org", "health". Unknown values fall back to general patterns.

    Returns:
        dict with total_sentences, claim_sentences, cited_sentences,
        evidence_density, verdict, domain, and a recommendation string.
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    sentences = _split_sentences(text)
    total = len(sentences)

    if total == 0:
        return {"success": False, "error": "No valid sentences found in text"}

    claim_patterns = _get_claim_patterns(domain)
    claim_count = sum(1 for s in sentences if _is_claim_sentence(s, patterns=claim_patterns))
    cited_count = sum(1 for s in sentences if _has_citation(s))

    density = round(cited_count / claim_count, 4) if claim_count > 0 else 0.0

    if density >= 0.6:
        verdict = "well-evidenced"
        recommendation = (
            "The text has a strong citation ratio. Review remaining uncited claims "
            "for accuracy but the overall evidential foundation is solid."
        )
    elif density >= 0.3:
        verdict = "partially-evidenced"
        recommendation = (
            "Some claims are cited but others lack source attribution. "
            "Identify the uncited claim sentences and add references or qualify the language."
        )
    else:
        verdict = "under-evidenced"
        if claim_count == 0:
            recommendation = (
                "No claim-bearing sentences detected. If the text makes empirical assertions, "
                "revisit the language to ensure claims are explicit and supported."
            )
        else:
            recommendation = (
                "Most claim-bearing sentences lack explicit citations. "
                "Add references for all statistics, prevalence figures, "
                "and causal statements before submission."
            )

    return {
        "success": True,
        "total_sentences": total,
        "claim_sentences": claim_count,
        "cited_sentences": cited_count,
        "evidence_density": density,
        "verdict": verdict,
        "domain": domain,
        "recommendation": recommendation,
    }
