"""Tests for evidence hallucination detection tools."""
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Claim extraction helpers
# ---------------------------------------------------------------------------

def test_is_claim_sentence_number_percentage():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence("HIV prevalence reached 12.5% among adults in 2023.")
    assert _is_claim_sentence("The programme covered 45 percent of the target population.")


def test_is_claim_sentence_epistemic_verb():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence("Evidence suggests that community-led interventions are more effective.")
    assert _is_claim_sentence("Research shows that peer support reduces dropout rates significantly.")


def test_is_claim_sentence_apa_citation():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence("Access to care improved substantially (Silva et al. 2022).")


def test_is_claim_sentence_numeric_citation():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence("The mortality rate declined by 30% over the period [1].")


def test_is_claim_sentence_prevalence_keyword():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence("In Mozambique, adolescent girls face significant barriers to healthcare.")
    assert _is_claim_sentence("PLHIV in SADC require sustained antiretroviral supply chains.")


def test_is_claim_sentence_plain_sentence_returns_false():
    from src.tools.evidence import _is_claim_sentence
    # No numbers, no epistemic verbs, no citations, no prevalence keywords
    assert not _is_claim_sentence("The programme was implemented by a dedicated team.")


def test_split_sentences_basic():
    from src.tools.evidence import _split_sentences
    text = "First sentence here. Second sentence here! Third sentence here?"
    result = _split_sentences(text)
    assert len(result) == 3


def test_split_sentences_filters_short():
    from src.tools.evidence import _split_sentences
    result = _split_sentences("Hi. This is a proper sentence with enough characters.")
    assert all(len(s) > 20 for s in result)


# ---------------------------------------------------------------------------
# verify_claims — verified result when Zotero returns high score
# ---------------------------------------------------------------------------

def test_verify_claims_verified_when_zotero_high_score():
    high_score_source = [{
        "title": "HIV in Mozambique 2023",
        "citekey": "silva-HIV-2023",
        "score": 0.85,
        "source_type": "zotero",
        "excerpt": "HIV prevalence reached 12.5% among adults.",
    }]

    with patch("src.tools.evidence._search_zotero", return_value=high_score_source), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "HIV prevalence reached 12.5% among adults in Mozambique in 2023."
        )

    assert result["success"] is True
    assert result["total_claims"] >= 1
    assert result["verified_count"] >= 1
    claim = result["claims"][0]
    assert claim["verdict"] == "verified"
    assert claim["ghost_stat"] is False
    assert any(s["source_type"] == "zotero" for s in claim["sources"])


# ---------------------------------------------------------------------------
# verify_claims — unverified + ghost_stat=True when number present but no match
# ---------------------------------------------------------------------------

def test_verify_claims_ghost_stat_when_number_unverified():
    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "Maternal mortality is 45% higher in rural districts with limited access."
        )

    assert result["success"] is True
    assert result["total_claims"] >= 1
    unverified_claims = [c for c in result["claims"] if c["verdict"] == "unverified"]
    assert len(unverified_claims) >= 1
    ghost_stats = [c for c in unverified_claims if c["ghost_stat"]]
    assert len(ghost_stats) >= 1


# ---------------------------------------------------------------------------
# verify_claims — graceful fallback when Zotero import fails
# ---------------------------------------------------------------------------

def test_verify_claims_graceful_when_zotero_fails():
    low_score_source = [{
        "title": "Some cerebellum doc",
        "citekey": None,
        "score": 0.3,
        "source_type": "cerebellum",
        "excerpt": "Some text snippet.",
    }]

    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=low_score_source):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "HIV prevalence reached 12.5% among adults in Mozambique in 2023."
        )

    assert result["success"] is True
    assert result["total_claims"] >= 1
    # With only a low-score cerebellum source, the claim should be unverified
    claim = result["claims"][0]
    assert claim["verdict"] == "unverified"


# ---------------------------------------------------------------------------
# verify_claims — graceful fallback when both sources fail
# ---------------------------------------------------------------------------

def test_verify_claims_graceful_when_both_sources_fail():
    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "Research shows that adolescent girls in Mozambique face high dropout rates."
        )

    assert result["success"] is True
    assert result["total_claims"] >= 1
    for claim in result["claims"]:
        assert claim["verdict"] == "unverified"
        assert claim["sources"] == []


# ---------------------------------------------------------------------------
# verify_claims — no claims found → overall_evidence_score = 1.0
# ---------------------------------------------------------------------------

def test_verify_claims_no_claims_returns_perfect_score():
    from src.tools.evidence import verify_claims
    # Plain prose with no claim patterns
    result = verify_claims(
        "The programme was designed and implemented by a dedicated team. "
        "Staff members worked collaboratively across multiple locations."
    )

    assert result["success"] is True
    assert result["total_claims"] == 0
    assert result["overall_evidence_score"] == 1.0
    assert result["verdict"] == "evidenced"
    assert result["claims"] == []


# ---------------------------------------------------------------------------
# verify_claims — empty text returns error
# ---------------------------------------------------------------------------

def test_verify_claims_empty_text():
    from src.tools.evidence import verify_claims
    result = verify_claims("")
    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# verify_claims — overall_evidence_score and verdict thresholds
# ---------------------------------------------------------------------------

def test_verify_claims_verdict_mixed():
    """50% verified → mixed verdict."""
    call_count = {"n": 0}

    def zotero_side_effect(query, top_k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First claim: high score → verified
            return [{"title": "Doc A", "citekey": "a-2023", "score": 0.9,
                     "source_type": "zotero", "excerpt": "excerpt A"}]
        # Second claim: no results → unverified
        return []

    with patch("src.tools.evidence._search_zotero", side_effect=zotero_side_effect), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "HIV prevalence reached 12.5% in the region. "
            "Maternal mortality is 45% higher in rural districts with limited access."
        )

    assert result["success"] is True
    assert result["total_claims"] == 2
    assert result["verified_count"] == 1
    assert result["overall_evidence_score"] == 0.5
    assert result["verdict"] == "mixed"


# ---------------------------------------------------------------------------
# score_evidence_density
# ---------------------------------------------------------------------------

def test_score_evidence_density_well_evidenced():
    from src.tools.evidence import score_evidence_density
    # Two claim sentences, both with APA citations
    text = (
        "HIV prevalence reached 12.5% among adults (UNAIDS 2023). "
        "Maternal mortality declined by 30% over the decade (WHO 2022). "
        "The programme was implemented by a committed team."
    )
    result = score_evidence_density(text)
    assert result["success"] is True
    assert result["claim_sentences"] >= 2
    assert result["cited_sentences"] >= 2
    assert result["verdict"] == "well-evidenced"


def test_score_evidence_density_under_evidenced():
    from src.tools.evidence import score_evidence_density
    # Three claim sentences with no citations
    text = (
        "HIV prevalence reached 12.5% among adults in the region. "
        "Research shows that maternal mortality is increasing in rural areas. "
        "In Mozambique, adolescent girls are particularly vulnerable to early marriage."
    )
    result = score_evidence_density(text)
    assert result["success"] is True
    assert result["claim_sentences"] >= 2
    assert result["cited_sentences"] == 0
    assert result["verdict"] == "under-evidenced"


def test_score_evidence_density_no_claims():
    from src.tools.evidence import score_evidence_density
    text = (
        "The programme was designed by a dedicated team of professionals. "
        "Staff members collaborated across multiple offices throughout the year."
    )
    result = score_evidence_density(text)
    assert result["success"] is True
    assert result["claim_sentences"] == 0
    assert result["evidence_density"] == 0.0


def test_score_evidence_density_empty_text():
    from src.tools.evidence import score_evidence_density
    result = score_evidence_density("")
    assert result["success"] is False
    assert "error" in result


def test_score_evidence_density_counts_numeric_citations():
    from src.tools.evidence import score_evidence_density
    text = (
        "Under-five mortality rates declined by 22% [1]. "
        "PLHIV in SADC reached 20 million in 2022 [2]. "
        "The team delivered training to 500 participants."
    )
    result = score_evidence_density(text)
    assert result["success"] is True
    assert result["cited_sentences"] >= 2
