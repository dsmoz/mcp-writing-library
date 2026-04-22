"""Tests for Portuguese-specific AI-writing forensic detectors."""
from src.tools.pt_forensic import score_pt_forensic


JURIDIQUES_TEXT = (
    "Outrossim, doravante haja vista o supracitado, consoante o disposto nos "
    "termos da lei, vem respeitosamente o hodierno conquanto ad hoc. "
    "Destarte, à guisa de consideração, mutatis mutandis ipso facto."
)

SYNTHETIC_PASSIVE_TEXT = (
    "O relatório foi elaborado pela equipa. As metas foram alcançadas pelos "
    "beneficiários. Os resultados foram apresentados aos doadores. O plano foi "
    "aprovado pela direcção. A avaliação foi conduzida pelo consultor. Os "
    "indicadores foram validados pelo comité."
)

NOMINALISATION_TEXT = (
    "A realização da implementação da monitorização teve como consequência a "
    "obtenção de resultados de qualidade e sustentabilidade com elevada "
    "capacidade de transformação. Esta situação, pela implementação do "
    "planeamento, resulta na avaliação do desenvolvimento da capacitação e do "
    "fortalecimento institucional."
)

CLEAN_PT_TEXT = (
    "A equipa visitou as comunidades em Maputo. Ouvimos histórias difíceis. "
    "Falámos com jovens, mães e enfermeiros. O que aprendemos mudou o plano "
    "para o ano seguinte."
)


def test_score_returns_success_and_required_keys():
    r = score_pt_forensic(CLEAN_PT_TEXT)
    assert r["success"]
    for k in ["overall_score", "verdict", "categories", "summary", "word_count", "page_equivalent"]:
        assert k in r
    assert set(r["categories"].keys()) == {
        "juridiques_density", "synthetic_passive", "nominalisation_density",
    }


def test_juridiques_flags_bureaucratic_jargon():
    r = score_pt_forensic(JURIDIQUES_TEXT)
    assert r["success"]
    assert r["categories"]["juridiques_density"]["score"] >= 0.5


def test_synthetic_passive_flags_foi_realizado_family():
    r = score_pt_forensic(SYNTHETIC_PASSIVE_TEXT)
    assert r["success"]
    assert r["categories"]["synthetic_passive"]["score"] > 0.3


def test_nominalisation_flags_suffix_heavy_prose():
    r = score_pt_forensic(NOMINALISATION_TEXT)
    assert r["success"]
    assert r["categories"]["nominalisation_density"]["score"] > 0.3


def test_clean_text_scores_low():
    r = score_pt_forensic(CLEAN_PT_TEXT)
    assert r["success"]
    assert r["verdict"] == "clean"
    assert r["overall_score"] < 0.25


def test_empty_text_rejected():
    r = score_pt_forensic("")
    assert not r["success"]


def test_invalid_threshold_rejected():
    r = score_pt_forensic("alguma coisa qualquer aqui escrita", threshold=2.5)
    assert not r["success"]
