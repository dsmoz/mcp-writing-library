"""
Tests for the AI writing pattern scorer.

All tests are pure Python — no external services or mocking needed.
"""
import pytest
from src.tools.ai_patterns import score_ai_patterns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLEAN_TEXT = """
The programme reached 3,400 key population members in 2024, exceeding the annual target by 12%.
Community health workers delivered peer-led interventions across five districts.
Follow-up data collected at 90 days showed a 34% increase in voluntary testing uptake.

These results reflect the strength of the community-based model.
Peer educators received 40 hours of training and were supervised monthly.
The data collection system proved reliable across all sites.

Several gaps remain. HIV stigma continues to reduce service uptake in rural areas.
Supply chain disruptions in Q3 affected condom distribution for six weeks.
The programme will address both issues in the 2025 work plan.
"""

HEAVY_AI_TEXT = """
Furthermore, it is important to note that the programme has achieved remarkable results.
Moreover, the fundamental insight here is that community engagement is not mere rhetoric.
Additionally, these outcomes are not mere statistical artefacts.

Firstly, the data shows strong uptake. Secondly, the community responded well.
Thirdly, the supervision model proved effective. Finally, all targets were met.

In conclusion, this report has shown that the programme delivered on its commitments.
To summarise the above, the evidence demonstrates consistent impact across all indicators.
"""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_text_returns_error():
    result = score_ai_patterns("")
    assert result["success"] is False
    assert "error" in result


def test_whitespace_only_returns_error():
    result = score_ai_patterns("   \n\n  ")
    assert result["success"] is False


def test_invalid_language_returns_error():
    result = score_ai_patterns("Some text.", language="fr")
    assert result["success"] is False
    assert "language" in result["error"].lower()


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

def test_clean_varied_text_returns_low_score():
    result = score_ai_patterns(CLEAN_TEXT)
    assert result["success"] is True
    assert result["verdict"] in ("clean", "review")
    assert result["overall_score"] < 0.55


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------

def test_connector_repetition_detected():
    text = (
        "Furthermore, the programme achieved its targets. "
        "Furthermore, the data confirms this. "
        "Furthermore, the community benefited greatly."
    )
    result = score_ai_patterns(text, language="en")
    assert result["success"] is True
    cat = result["categories"]["connector_repetition"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_hollow_intensifiers_detected():
    text = (
        "It is important to note that the results are strong. "
        "The programme delivered well in 2024. "
        "It should be noted that gaps remain in rural areas."
    )
    result = score_ai_patterns(text, language="en")
    assert result["success"] is True
    cat = result["categories"]["hollow_intensifiers"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_grandiose_opener_en_detected():
    text = (
        "Against this backdrop of progress, the programme faces new challenges.\n\n"
        "The fundamental insight here is that community trust takes time to build.\n\n"
        "The evidence is unequivocal: peer-led models outperform facility-based ones."
    )
    result = score_ai_patterns(text, language="en")
    assert result["success"] is True
    cat = result["categories"]["grandiose_openers"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_grandiose_opener_pt_detected():
    text = (
        "Contra este pano de fundo, o programa enfrenta novos desafios.\n\n"
        "A percepção fundamental aqui é que a confiança da comunidade leva tempo.\n\n"
        "Os dados revelam um padrão consistente em todos os distritos."
    )
    result = score_ai_patterns(text, language="pt")
    assert result["success"] is True
    cat = result["categories"]["grandiose_openers"]
    assert cat["score"] > 0


def test_em_dash_intercalation_detected():
    text = (
        "The data — though fragmented — points to consistent patterns.\n"
        "Community workers — all trained in 2023 — delivered 80% of sessions.\n"
        "Results were strong across all districts."
    )
    result = score_ai_patterns(text)
    assert result["success"] is True
    cat = result["categories"]["em_dash_intercalation"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_sentence_monotony_detected():
    # 5 sentences all approximately 10 words
    text = (
        "The programme reached its annual targets in all districts. "
        "Community workers delivered sessions across five provinces. "
        "Testing uptake increased by thirty percent during the period. "
        "Supply chain issues were resolved in the third quarter. "
        "All indicators met their planned targets for the year."
    )
    result = score_ai_patterns(text)
    assert result["success"] is True
    cat = result["categories"]["sentence_monotony"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_passive_voice_density_detected():
    text = (
        "The targets were exceeded by the programme team. "
        "Data was collected by community workers at each site. "
        "Sessions were delivered to over 3,000 participants. "
        "Reports were compiled and reviewed by supervisors. "
        "All findings were validated by the M&E team."
    )
    result = score_ai_patterns(text, language="en")
    assert result["success"] is True
    cat = result["categories"]["passive_voice"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_paragraph_length_detected():
    # One paragraph with 6 sentences
    text = (
        "The programme exceeded its targets in 2024. "
        "Community workers delivered peer-led sessions. "
        "Testing uptake rose by 34 percent. "
        "Supply disruptions were resolved by Q4. "
        "The supervision model proved effective. "
        "All six districts reported positive outcomes."
    )
    result = score_ai_patterns(text)
    assert result["success"] is True
    cat = result["categories"]["paragraph_length"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_discursive_deficit_detected():
    # ~300 words of plain factual text with no discursive expressions
    plain = (
        "The programme reached 3,400 people in 2024. Testing uptake rose 34%. "
        "Community workers delivered sessions in five districts. Supply issues "
        "affected Q3. The supervision model was monthly. All targets were met. " * 6
    )
    result = score_ai_patterns(plain)
    assert result["success"] is True
    cat = result["categories"]["discursive_deficit"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_mechanical_listing_detected():
    text = (
        "Firstly, the programme achieved its targets.\n\n"
        "Secondly, community workers performed well.\n\n"
        "Thirdly, the supervision model was effective.\n\n"
        "Finally, all districts reported positive outcomes."
    )
    result = score_ai_patterns(text, language="en")
    assert result["success"] is True
    cat = result["categories"]["mechanical_listing"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_generic_closing_detected():
    text = (
        "The programme delivered strong results across all indicators.\n\n"
        "In conclusion, this report has shown that community-based models work.\n"
        "To summarise the above, the evidence supports continued investment."
    )
    result = score_ai_patterns(text, language="en")
    assert result["success"] is True
    cat = result["categories"]["generic_closings"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def test_language_auto_detect_en():
    result = score_ai_patterns("The programme achieved its targets in all districts.", language="auto")
    assert result["success"] is True
    assert result["language"] == "en"


def test_language_auto_detect_pt():
    result = score_ai_patterns(
        "O programa atingiu as metas em todos os distritos. "
        "Os trabalhadores comunitários entregaram sessões com sucesso. "
        "Os dados são consistentes com as tendências regionais.",
        language="auto",
    )
    assert result["success"] is True
    assert result["language"] == "pt"


# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------

def test_verdict_thresholds():
    # Heavy AI text should score high
    heavy = score_ai_patterns(HEAVY_AI_TEXT)
    assert heavy["success"] is True
    assert heavy["overall_score"] >= 0.0  # score is non-negative

    # Clean text should score lower than heavy AI text
    clean = score_ai_patterns(CLEAN_TEXT)
    assert clean["success"] is True
    assert clean["overall_score"] < heavy["overall_score"]


def test_verdict_field_present():
    result = score_ai_patterns(CLEAN_TEXT)
    assert result["success"] is True
    assert result["verdict"] in ("clean", "review", "ai-sounding")


def test_return_structure_complete():
    result = score_ai_patterns(CLEAN_TEXT)
    assert result["success"] is True
    required_keys = {"language", "overall_score", "verdict", "threshold", "doc_type", "categories", "summary", "word_count", "page_equivalent"}
    assert required_keys.issubset(result.keys())
    # All 10 categories present
    expected_categories = {
        "connector_repetition", "hollow_intensifiers", "grandiose_openers",
        "em_dash_intercalation", "sentence_monotony", "passive_voice",
        "paragraph_length", "discursive_deficit", "mechanical_listing", "generic_closings",
    }
    assert expected_categories == set(result["categories"].keys())


# ---------------------------------------------------------------------------
# doc_type threshold calibration
# ---------------------------------------------------------------------------

def test_financial_report_doc_type_no_discursive_deficit():
    # Plain factual text with no discursive expressions — financial-report should score 0.0
    plain = (
        "The programme reached 3,400 people in 2024. Testing uptake rose 34%. "
        "Community workers delivered sessions in five districts. Supply issues "
        "affected Q3. The supervision model was monthly. All targets were met. " * 6
    )
    result = score_ai_patterns(plain, doc_type="financial-report")
    assert result["success"] is True
    assert result["doc_type"] == "financial-report"
    cat = result["categories"]["discursive_deficit"]
    assert cat["score"] == 0.0
    assert cat["findings"] == []


def test_monitoring_report_allows_seven_sentence_paragraphs():
    # One paragraph with exactly 7 sentences — should NOT be flagged for monitoring-report (limit=7)
    text = (
        "The programme exceeded its targets in 2024. "
        "Community workers delivered peer-led sessions. "
        "Testing uptake rose by 34 percent. "
        "Supply disruptions were resolved by Q4. "
        "The supervision model proved effective. "
        "All six districts reported positive outcomes. "
        "Data quality checks passed at all sites."
    )
    result = score_ai_patterns(text, doc_type="monitoring-report")
    assert result["success"] is True
    assert result["doc_type"] == "monitoring-report"
    cat = result["categories"]["paragraph_length"]
    assert cat["score"] == 0.0
    assert cat["findings"] == []


def test_concept_note_flags_five_sentence_paragraph():
    # One paragraph with 5 sentences — should be flagged for concept-note (limit=4)
    text = (
        "The programme exceeded its targets in 2024. "
        "Community workers delivered peer-led sessions. "
        "Testing uptake rose by 34 percent. "
        "Supply disruptions were resolved by Q4. "
        "All six districts reported positive outcomes."
    )
    result = score_ai_patterns(text, doc_type="concept-note")
    assert result["success"] is True
    assert result["doc_type"] == "concept-note"
    cat = result["categories"]["paragraph_length"]
    assert cat["score"] > 0
    assert len(cat["findings"]) > 0


def test_invalid_doc_type_returns_error():
    result = score_ai_patterns("Some text.", doc_type="brochure")
    assert result["success"] is False
    assert "doc_type" in result["error"].lower() or "brochure" in result["error"]
