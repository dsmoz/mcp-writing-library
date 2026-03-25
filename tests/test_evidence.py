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
# verify_claims — no claims found → distinct "no_claims_detected" verdict
# ---------------------------------------------------------------------------

def test_verify_claims_no_claims_returns_no_claims_detected():
    from src.tools.evidence import verify_claims
    # Plain prose with no claim patterns
    result = verify_claims(
        "The programme was designed and implemented by a dedicated team. "
        "Staff members worked collaboratively across multiple locations."
    )

    assert result["success"] is True
    assert result["total_claims"] == 0
    assert result["overall_evidence_score"] is None
    assert result["verdict"] == "no_claims_detected"
    assert result["claims"] == []
    assert "note" in result


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


# ---------------------------------------------------------------------------
# Portuguese pattern detection
# ---------------------------------------------------------------------------

def test_is_claim_sentence_portuguese_epistemic():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence(
        "Os dados mostram que a prevalência do VIH aumentou em Moçambique."
    )
    assert _is_claim_sentence(
        "Segundo o MISAU, a mortalidade materna continua elevada nas zonas rurais."
    )
    assert _is_claim_sentence(
        "A investigação sugere que as intervenções comunitárias são mais eficazes."
    )
    assert _is_claim_sentence(
        "De acordo com os estudos recentes, os jovens são mais vulneráveis."
    )


def test_is_claim_sentence_portuguese_prevalence():
    from src.tools.evidence import _is_claim_sentence
    assert _is_claim_sentence(
        "A prevalência do VIH entre populações-chave excede 10% em Moçambique."
    )
    assert _is_claim_sentence(
        "A mortalidade materna em Moçambique permanece entre as mais altas da região."
    )
    assert _is_claim_sentence(
        "Na África Austral, a prevalência do HIV entre adolescentes preocupa os especialistas."
    )


# ---------------------------------------------------------------------------
# ghost_stat for bare large numbers (Issue 3)
# ---------------------------------------------------------------------------

def test_has_number_matches_bare_large_numbers():
    from src.tools.evidence import _has_number
    # 4+ digit numbers
    assert _has_number("The programme reached 35000 beneficiaries across the district.")
    assert _has_number("There were 1,234 participants enrolled in the study.")
    # 3-digit numbers
    assert _has_number("Some 150 communities participated in the baseline survey.")
    # Still matches percentages
    assert _has_number("HIV prevalence reached 12.5% among adults.")


def test_has_number_does_not_flag_single_digits():
    from src.tools.evidence import _has_number
    # Single digits should not trigger (too common, not a "stat")
    assert not _has_number("There are 5 key partners involved in the project.")
    assert not _has_number("The team has 3 staff members.")


def test_ghost_stat_flagged_for_bare_number():
    """Unverified claim with a bare large number should set ghost_stat=True."""
    from unittest.mock import patch
    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "The programme reached 35000 beneficiaries in the northern provinces."
        )

    assert result["success"] is True
    unverified = [c for c in result["claims"] if c["verdict"] == "unverified"]
    assert len(unverified) >= 1
    assert any(c["ghost_stat"] for c in unverified)


# ---------------------------------------------------------------------------
# Domain-aware claim pattern tests
# ---------------------------------------------------------------------------

def test_domain_health_detects_hiv_prevalence_sentence():
    """With domain='health', 'HIV prevalence' sentence is still detected as a claim."""
    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "HIV prevalence among adolescent girls has remained persistently high.",
            domain="health",
        )

    assert result["success"] is True
    assert result["total_claims"] >= 1


def test_domain_finance_detects_budget_sentence():
    """With domain='finance', 'budget allocation of USD 2 million' is detected as a claim."""
    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims
        result = verify_claims(
            "The budget allocation of USD 2 million was approved for the next fiscal year.",
            domain="finance",
        )

    assert result["success"] is True
    assert result["total_claims"] >= 1


def test_domain_m_and_e_detects_indicator_sentence_no_number():
    """With domain='m-and-e', an indicator sentence (no number, no epistemic verb) IS a claim.
    With domain='general' the same sentence is NOT a claim.
    """
    from src.tools.evidence import _is_claim_sentence, _get_claim_patterns

    sentence = "The indicator was not met during the reporting period."

    # domain=general: should NOT detect as claim
    general_patterns = _get_claim_patterns("general")
    assert not _is_claim_sentence(sentence, patterns=general_patterns)

    # domain=m-and-e: SHOULD detect as claim
    mne_patterns = _get_claim_patterns("m-and-e")
    assert _is_claim_sentence(sentence, patterns=mne_patterns)


def test_verify_claims_returns_domain_key():
    """Return dict from verify_claims() must include a 'domain' key."""
    with patch("src.tools.evidence._search_zotero", return_value=[]), \
         patch("src.tools.evidence._search_cerebellum", return_value=[]):
        from src.tools.evidence import verify_claims

        result_with_claims = verify_claims(
            "HIV prevalence reached 12.5% in the region.",
            domain="health",
        )
        assert "domain" in result_with_claims
        assert result_with_claims["domain"] == "health"

        result_no_claims = verify_claims(
            "The programme was implemented by a dedicated team.",
            domain="governance",
        )
        assert "domain" in result_no_claims
        assert result_no_claims["domain"] == "governance"


def test_domain_unknown_falls_back_to_general_patterns():
    """Unknown domain values use only the base _ALL_CLAIM_PATTERNS."""
    from src.tools.evidence import _get_claim_patterns, _ALL_CLAIM_PATTERNS
    patterns = _get_claim_patterns("unknown-domain")
    assert patterns == _ALL_CLAIM_PATTERNS
