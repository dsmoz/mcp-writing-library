#!/usr/bin/env python3
"""
Seed writing_terms collection from existing copywriter reference files.

Sources:
    ~/.claude/agents/references/copywriter/data/overstated-language-alternatives.md
    (AI slop section of) natural-writing-quick-reference.md

Run:
    uv run python scripts/seed_from_markdown.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
if kbase_core_path.exists():
    sys.path.insert(0, str(kbase_core_path))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.tools.terms import add_term
from src.tools.passages import add_passage

HOME = Path.home()
COPYWRITER_REF = HOME / '.claude/agents/references/copywriter/data'

# ----------------------------------------------------------------
# Terminology seed data
# ----------------------------------------------------------------
# Extracted from overstated-language-alternatives.md + AI slop blocklist
TERMS = [
    # Overstated language (consultancy context)
    dict(preferred="notable", avoid="unprecedented", domain="general", language="en",
         why="Superlative; undermines credibility in evidence-based writing",
         example_bad="This unprecedented initiative...", example_good="This notable initiative..."),
    dict(preferred="favourable", avoid="optimal", domain="general", language="en",
         why="Overstated; use measured language",
         example_bad="optimal conditions", example_good="favourable conditions"),
    dict(preferred="a priority", avoid="paramount", domain="general", language="en",
         why="Hyperbolic; plain language preferred",
         example_bad="It is paramount that...", example_good="It is a priority that..."),
    dict(preferred="significant", avoid="transformative", domain="general", language="en",
         why="Overpromising; use measured language",
         example_bad="transformative change", example_good="significant change"),
    dict(preferred="new", avoid="groundbreaking", domain="general", language="en",
         why="Hyperbole; UNDP standard prohibits superlatives",
         example_bad="groundbreaking research", example_good="new research"),
    dict(preferred="inconsistent", avoid="paradoxical", domain="general", language="en",
         why="Too abstract for evidence-based writing; suggests philosophical impossibility",
         example_bad="a paradoxical approach to rights", example_good="an inconsistent approach to rights"),
    # AI slop verbs
    dict(preferred="examine", avoid="delve into", domain="general", language="en",
         why="AI slop — never appears in sector documents",
         example_bad="Let us delve into the data...", example_good="The data reveals..."),
    dict(preferred="use", avoid="leverage", domain="general", language="en",
         why="Corporate jargon; plain language preferred",
         example_bad="leverage innovative approaches", example_good="apply community-led approaches"),
    dict(preferred="use", avoid="utilize", domain="general", language="en",
         why="Verbose; 'use' is always sufficient",
         example_bad="utilize available resources", example_good="use available resources"),
    dict(preferred="coordinate", avoid="synergize", domain="general", language="en",
         why="Management jargon; avoid in sector documents",
         example_bad="synergize stakeholder outcomes", example_good="coordinate stakeholder inputs"),
    dict(preferred="explain", avoid="unpack", domain="general", language="en",
         why="AI slop — informal; not in development sector writing",
         example_bad="Let us unpack this finding.", example_good="This finding reveals..."),
    # People-first language (SRHR)
    dict(preferred="people living with HIV", avoid="HIV victims", domain="srhr", language="en",
         why="Deficit framing undermines agency. UNDP/UNAIDS standard terminology.",
         example_bad="HIV victims need treatment.", example_good="People living with HIV require access to treatment."),
    dict(preferred="key populations", avoid="high-risk groups", domain="srhr", language="en",
         why="UNAIDS standard; 'high-risk' stigmatises rather than describes structural barriers",
         example_bad="high-risk groups face barriers", example_good="key populations face structural barriers"),
    dict(preferred="sex worker", avoid="prostitute", domain="srhr", language="en",
         why="Rights-based terminology; recognises labour rights",
         example_bad="prostitutes in the study area", example_good="sex workers in the study area"),
    dict(preferred="person who uses drugs", avoid="drug addict", domain="srhr", language="en",
         why="People-first language; avoids criminalising framing",
         example_bad="drug addicts in the programme", example_good="people who use drugs in the programme"),
    dict(preferred="people with disabilities", avoid="the disabled", domain="general", language="en",
         why="People-first language — CRPD standard",
         example_bad="services for the disabled", example_good="services for people with disabilities"),
    # Padding phrases
    dict(preferred="[state the point directly]", avoid="It is important to note that",
         domain="general", language="en",
         why="Padding — adds no meaning; direct statement preferred",
         example_bad="It is important to note that data shows...", example_good="Data shows..."),
    dict(preferred="[state the point directly]", avoid="It goes without saying",
         domain="general", language="en",
         why="If it goes without saying, don't say it",
         example_bad="It goes without saying that results matter.", example_good="Results matter."),
    # PT equivalents
    dict(preferred="pessoas vivendo com HIV", avoid="vítimas do HIV", domain="srhr", language="pt",
         why="Linguagem baseada em direitos. Padrão ONUSIDA/CNCS.",
         example_bad="As vítimas do HIV necessitam...", example_good="As pessoas vivendo com HIV necessitam..."),
    dict(preferred="populações-chave", avoid="grupos de alto risco", domain="srhr", language="pt",
         why="Terminologia padrão ONUSIDA; 'alto risco' estigmatiza em vez de descrever barreiras estruturais",
         example_bad="grupos de alto risco enfrentam barreiras", example_good="populações-chave enfrentam barreiras estruturais"),
]

# ----------------------------------------------------------------
# Passage seed data (before/after models from natural-writing-quick-reference.md)
# ----------------------------------------------------------------
PASSAGES = [
    dict(
        text="The assessment reveals a familiar pattern: progress coexists with persistent gaps. Service delivery has expanded, yet implementation challenges continue to undermine outcomes—particularly in coordination across sectors. What emerges clearly is the need for sustained investment, not as an option but as a prerequisite for consolidating gains.",
        doc_type="executive-summary", language="en", domain="general",
        quality_notes="Classic before/after rewrite. Replaces connector-only sequence with discursive opener and argumentative momentum. Shows contrast without forced 'However'.",
        tags=["discursive", "contrast", "argumentative-momentum"],
        source="natural-writing-quick-reference",
    ),
    dict(
        text="The legal framework has strengthened—reforms enacted since 2019 represent genuine progress. The challenge lies elsewhere: in the persistent gap between policy and practice. Capacity constraints explain part of this disconnect, but enforcement inconsistency points to deeper institutional factors. The question is not whether reforms are needed, but how to translate existing commitments into operational reality.",
        doc_type="report", language="en", domain="governance",
        quality_notes="Findings paragraph with argumentative flow. Short-medium-long rhythm. Uses em-dash for tight transition. Ends with a direct question that frames the analysis.",
        tags=["findings", "argumentative-flow", "rhythm-variation", "policy-practice-gap"],
        source="natural-writing-quick-reference",
    ),
    dict(
        text="O que a análise revela é um quadro de progressos reais mas frágeis. O enquadramento legal fortaleceu-se, os indicadores melhoraram—e no entanto, desafios estruturais persistem. O padrão é consistente: avanços formais que esbarram em limitações de implementação. A questão central não é se existem ganhos, mas se são sustentáveis sem intervenção adicional.",
        doc_type="report", language="pt", domain="governance",
        quality_notes="PT equivalent of the findings before/after. Uses discursive opener 'O que a análise revela'. Avoids em-dash intercalations (uses 'e no entanto' instead). Ends with a direct question.",
        tags=["findings", "discursive", "PT", "policy-practice-gap"],
        source="natural-writing-quick-reference",
    ),
    dict(
        text="Angola's legal framework is evolving. Recent reforms signal political will. However, implementation gaps persist, particularly in rural areas where service infrastructure remains underdeveloped. This urban-rural divide undermines national health targets.",
        doc_type="report", language="en", domain="governance",
        quality_notes="Demonstrates rhythm variation: 5-word emphasis sentence, 7-word sentence, 15-word medium, 8-word conclusion. Mix of short-medium sentences for impact.",
        tags=["rhythm-variation", "short-sentences", "urban-rural", "health"],
        source="natural-writing-quick-reference",
    ),
]


def seed_terms():
    print(f"\nSeeding {len(TERMS)} terminology entries...")
    ok, fail = 0, 0
    for term in TERMS:
        result = add_term(**term)
        if result["success"]:
            ok += 1
            print(f"  ✅ {term['preferred']} ({term['language']})")
        else:
            fail += 1
            print(f"  ❌ {term['preferred']}: {result.get('error')}")
    print(f"Terms: {ok} added, {fail} failed")


def seed_passages():
    print(f"\nSeeding {len(PASSAGES)} exemplary passages...")
    ok, fail = 0, 0
    for passage in PASSAGES:
        result = add_passage(**passage)
        if result["success"]:
            ok += 1
            print(f"  ✅ [{passage['doc_type']} | {passage['language']}] {passage['text'][:50]}...")
        else:
            fail += 1
            print(f"  ❌ {result.get('error')}")
    print(f"Passages: {ok} added, {fail} failed")


# ----------------------------------------------------------------
# Style-tagged passage seed data
# ----------------------------------------------------------------
STYLE_SEED_PASSAGES = [
    # structural: narrative
    dict(
        text="In 2022, a community health worker in Nampula described her daily rounds as 'walking between two worlds': one where health data was recorded dutifully in logbooks, another where those records never reached the district office. Her observation captures a structural failure that national surveys have since confirmed at scale.",
        doc_type="report", language="en", domain="srhr",
        quality_notes="Opens with an individual story, scales to systemic observation. Classic narrative structure.",
        tags=["story-led", "systemic-observation", "community-voice"],
        source="manual",
        style=["narrative", "conversational"],
    ),
    # structural: data-driven
    dict(
        text="Coverage rates for antenatal care reached 87% nationally in 2023, up from 71% in 2018. Yet facility-based delivery rates stagnated at 54%, revealing a persistent dropout between first ANC contact and skilled birth attendance. This gap is not random: it tracks closely with distance to the nearest health facility, a factor that accounts for 63% of the variance in delivery outcomes across districts.",
        doc_type="report", language="en", domain="srhr",
        quality_notes="Leads with data, explains causation. Each sentence adds interpretive depth. No hedging.",
        tags=["statistics", "causation", "health-data"],
        source="manual",
        style=["data-driven", "formal"],
    ),
    # structural: argumentative
    dict(
        text="The funding gap in community-led HIV responses is not a technical problem. It is a political one. Donors have consistently underfunded grassroots organisations relative to their epidemiological footprint, a pattern that cannot be explained by capacity concerns alone. If the goal is to reach key populations, then the current allocation logic is working against it.",
        doc_type="policy-brief", language="en", domain="srhr",
        quality_notes="States thesis in first sentence. Builds case with evidence. Final sentence returns to the argument.",
        tags=["thesis-driven", "funding", "advocacy"],
        source="manual",
        style=["argumentative", "advocacy"],
    ),
    # structural: minimalist
    dict(
        text="Three findings stand out. First, service access has improved. Second, quality has not. Third, the gap between the two is widening. Any strategy that addresses access without addressing quality will not close the outcome gap.",
        doc_type="executive-summary", language="en", domain="general",
        quality_notes="High information density. Short, parallel structure. No filler. Final sentence is the implication.",
        tags=["crisp", "parallel-structure", "executive"],
        source="manual",
        style=["minimalist", "formal"],
    ),
    # tonal: donor-facing
    dict(
        text="By the end of Year 2, the project will have directly reached 12,400 rights-holders across four provinces, with 78% of participants demonstrating improved knowledge of SRHR services. The sustainability plan — embedded in the district health system from Month 6 — ensures continued service delivery beyond the grant period without additional donor financing.",
        doc_type="concept-note", language="en", domain="srhr",
        quality_notes="Results chain is explicit. Indicators anchored to timeline. Sustainability language reassures donors.",
        tags=["results-chain", "indicators", "sustainability"],
        source="manual",
        style=["donor-facing", "formal"],
    ),
    # tonal: advocacy
    dict(
        text="Criminalisation does not protect communities. It protects the status quo. When same-sex relationships remain illegal, health workers cannot ask the right questions, organisations cannot operate openly, and people cannot seek care without fear. The evidence is consistent across every context where these laws have been studied. The question is not whether decriminalisation improves health outcomes. It does. The question is whether policymakers are willing to act on that evidence.",
        doc_type="policy-brief", language="en", domain="srhr",
        quality_notes="Rights-based framing, urgency without hyperbole. Rhetorical build. Evidence-anchored conclusion.",
        tags=["rights-based", "criminalisation", "LGBTQIA"],
        source="manual",
        style=["advocacy", "argumentative"],
    ),
    # source: undp
    dict(
        text="Human development progress in Southern Africa has been uneven, reflecting both the gains of the post-2000 period and the persistent structural constraints that limit their sustainability. Where capabilities have expanded — in education, in health access, in civic participation — the gains are real but remain concentrated among those already closest to opportunity. Addressing this distribution challenge requires not only continued investment but a reconsideration of how development resources are allocated across population groups.",
        doc_type="report", language="en", domain="governance",
        quality_notes="UNDP HDR register. Capability framing. Discursive transitions. No em-dashes. Measured language.",
        tags=["capability", "HDR-register", "distribution"],
        source="undp-hdr",
        style=["undp", "formal", "data-driven"],
    ),
    # source: danilo-voice
    dict(
        text="The organisations doing the hardest work in this sector are rarely the ones receiving the largest grants. That is not an accident of funding systems — it is a feature of them. Community-based organisations working with key populations in Mozambique and Angola operate in a context where visibility is risk, where legal ambiguity discourages formal registration, and where donor due diligence processes were designed for a different kind of organisation entirely. The question practitioners should be asking is not how to help these organisations fit the system. It is how to change the system so it stops excluding them.",
        doc_type="general", language="en", domain="srhr",
        quality_notes="Direct, analytical, Southern African context. Rhythm variation. Ends with a reframe. Personal voice.",
        tags=["personal-voice", "system-critique", "SADC", "funding"],
        source="manual",
        style=["danilo-voice", "argumentative", "conversational"],
    ),
    # anti-pattern: ai-sounding
    dict(
        text="In today's rapidly evolving landscape, it is crucial to leverage synergies between stakeholders to harness transformative change. Furthermore, by delving into the complexities of the ecosystem, we can unlock unprecedented opportunities for impact. It goes without saying that utilising best practices will be paramount to achieving optimal outcomes.",
        doc_type="general", language="en", domain="general",
        quality_notes="NEGATIVE EXAMPLE. Contains: em-dashes absent but multiple AI slop markers — 'leverage', 'harness', 'delve', 'unprecedented', 'paramount', 'optimal', 'Furthermore', 'It goes without saying', 'rapidly evolving landscape'.",
        tags=["anti-pattern", "AI-slop", "do-not-use"],
        source="manual",
        style=["ai-sounding"],
    ),
    # anti-pattern: bureaucratic
    dict(
        text="The implementation of the aforementioned programmatic interventions shall be conducted in accordance with the established operational frameworks and procedural guidelines, ensuring the maximisation of resource utilisation efficiencies whilst simultaneously ensuring compliance with all applicable regulatory requirements and donor reporting obligations.",
        doc_type="report", language="en", domain="general",
        quality_notes="NEGATIVE EXAMPLE. Dense nominalisations, passive voice, stacked abstractions, zero concrete information.",
        tags=["anti-pattern", "nominalisation", "passive-voice", "do-not-use"],
        source="manual",
        style=["bureaucratic"],
    ),
    # tonal: conversational (LinkedIn)
    dict(
        text="Most grant applications fail before the first reviewer reads a word. Not because the project is weak — because the opening paragraph loses them. I've reviewed hundreds of proposals across Mozambique and Angola. The ones that work all do the same thing: they start with the problem, not the organisation. One sentence. The problem. That's the hook.",
        doc_type="general", language="en", domain="general",
        quality_notes="LinkedIn/conversational register. Short paragraphs. Direct address. 'I've' is intentional. Ends with punchy takeaway.",
        tags=["LinkedIn", "conversational", "grant-writing", "hook"],
        source="manual",
        style=["conversational", "danilo-voice", "narrative"],
    ),
    # structural: narrative (Portuguese)
    dict(
        text="Em 2021, uma organização de base em Maputo perdeu o seu financiamento principal três semanas antes do final do ano fiscal. O coordenador de programas descreveu aquela semana como 'o momento em que percebemos que éramos invisíveis para o sistema'. Não era uma crise de competência — era uma crise de visibilidade. A organização sabia o que fazer. O problema era que ninguém a via a fazê-lo.",
        doc_type="report", language="pt", domain="srhr",
        quality_notes="PT narrative voice. Story-led, scales to systemic observation. Natural Portuguese rhythm. Ends with a distinction that reframes the problem.",
        tags=["story-led", "PT", "funding-crisis", "visibility"],
        source="manual",
        style=["narrative", "danilo-voice"],
    ),
]


def seed_style_passages():
    print(f"\nSeeding {len(STYLE_SEED_PASSAGES)} style-tagged passages...")
    ok, fail = 0, 0
    for passage in STYLE_SEED_PASSAGES:
        result = add_passage(**passage)
        if result["success"]:
            ok += 1
            styles_str = ", ".join(passage.get("style", []))
            print(f"  ✅ [{styles_str}] {passage['text'][:50]}...")
            if result.get("warnings"):
                for w in result["warnings"]:
                    print(f"     ⚠️  {w}")
        else:
            fail += 1
            print(f"  ❌ {result.get('error')}")
    print(f"Style passages: {ok} added, {fail} failed")


if __name__ == "__main__":
    seed_terms()
    seed_passages()
    seed_style_passages()
    print("\nDone. Run setup_collections.py stats to verify counts.")
