"""Seed the canonical pattern JSON files in data/patterns/.

Defaults are inlined here so the script keeps working after ai_patterns.py is
refactored to load from JSON. Re-run with --force to regenerate; by default,
existing files are left untouched.
"""
import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "patterns"


DEFAULTS: dict[str, dict] = {
    "connectors_en": {
        "description": "EN connectors that read as AI-generated when repeated",
        "items": [
            "furthermore", "additionally", "moreover", "in conclusion", "in summary",
            "to summarise", "to summarize", "firstly", "secondly", "thirdly",
            "lastly", "in addition", "it is worth noting",
        ],
    },
    "connectors_pt": {
        "description": "PT connectors that read as AI-generated when repeated",
        "items": [
            "além disso", "adicionalmente", "ademais", "em conclusão", "em resumo",
            "em primeiro lugar", "em segundo lugar", "em terceiro lugar",
            "por último", "igualmente", "do mesmo modo",
        ],
    },
    "hollow_intensifiers": {
        "description": "Hollow intensifier regex patterns (EN)",
        "items": [
            r"it is important to note that",
            r"it is crucial that",
            r"it is essential to recognise",
            r"it is essential to recognize",
            r"it should be noted that",
            r"it is worth noting that",
            r"it is important to highlight",
            r"it bears emphasising",
            r"it bears emphasizing",
            r"it must be acknowledged",
            r"needless to say",
        ],
    },
    "grandiose_openers_en": {
        "description": "Grandiose paragraph opener regex patterns (EN)",
        "items": [
            r"against this backdrop",
            r"the fundamental insight here is that",
            r"what emerges from this analysis",
            r"the picture that emerges",
            r"these .{0,30} are not mere",
            r"\w+ deserve(?:s)? special attention",
            r"in this context of",
            r"at the heart of this",
            r"this is a pivotal moment",
            r"this represents a watershed",
            r"the evidence is unequivocal",
            r"the data paints? a (?:clear|stark|compelling)",
        ],
    },
    "grandiose_openers_pt": {
        "description": "Grandiose paragraph opener regex patterns (PT)",
        "items": [
            r"contra este pano de fundo",
            r"a percep[çc][ãa]o fundamental aqui [eé] que",
            r"o quadro que emerge",
            r"estas? .{0,30} n[ãa]o s[ãa]o meros?",
            r"\w+ merece(?:m)? destaque",
            r"neste contexto de",
            r"no cerne desta",
            r"este [eé] um momento fulcral",
            r"os dados revelam",
            r"a evid[eê]ncia [eé] inequ[íi]voca",
        ],
    },
    "generic_closings": {
        "description": "Generic AI closing phrase regex patterns (EN)",
        "items": [
            r"in conclusion,? this (?:report|document|paper|analysis) has shown",
            r"to summaris[e] the above",
            r"to summarize the above",
            r"as (?:has been )?demonstrated above",
            r"as (?:has been )?shown above",
            r"in summary,? this (?:report|document|analysis)",
            r"the foregoing analysis (?:has shown|demonstrates)",
            r"as outlined (?:above|in this report)",
        ],
    },
    "discursive_expressions": {
        "description": "Discursive stance marker regex patterns (EN + PT); their absence is the AI signal",
        "items": [
            r"what (?:this|the analysis|the evidence) (?:reveals?|shows?|suggests?|indicates?)",
            r"what emerges from",
            r"the key (?:insight|finding|implication) (?:here )?is",
            r"what makes this (?:particularly )?significant",
            r"this raises (?:a )?(?:crucial|important|key) question",
            r"the implications extend beyond",
            r"what this means in practice",
            r"to understand why",
            r"consider what this suggests",
            r"the challenge,? then,? is",
            r"building on this",
            r"this is evident in",
            r"o que (?:esta|a análise|os dados) (?:revela?|mostra?|sugere?)",
            r"o que emerge de",
            r"a (?:conclusão|lição|implicação) central (?:aqui )?é",
            r"o que isto significa na prática",
            r"para compreender (?:por que|porquê)",
            r"esta (?:questão|realidade) é evidente em",
            r"construindo sobre isto",
            r"o desafio,? portanto,? é",
        ],
    },
    "passive_patterns": {
        "description": "EN passive-voice regex patterns",
        "items": [
            r"\b(?:was|were|is|are|has been|have been|had been|being)\s+\w+ed\b",
            r"\b(?:was|were|is|are|has been|have been|had been|being)\s+\w+en\b",
        ],
    },
    "pt_function_words": {
        "description": "Portuguese function words used by the language detector",
        "items": [
            "que", "uma", "para", "com", "por", "são", "também", "mais", "sobre",
            "como", "mas", "dos", "das", "nos", "nas", "quando", "porque", "entre",
            "seus", "suas", "este", "esta", "estes", "estas", "isso", "essa",
        ],
    },
    "para_limits": {
        "description": "Max sentences per paragraph before flagging, keyed by doc_type",
        "values": {
            "concept-note": 4, "full-proposal": 4, "eoi": 4, "executive-summary": 3,
            "general": 5, "annual-report": 6, "monitoring-report": 7,
            "financial-report": 8, "assessment": 7, "tor": 6, "governance-review": 6,
            "facebook-post": 2, "linkedin-post": 3, "instagram-caption": 1,
            "haiku": 1, "sonnet": 4, "free-verse": 6, "villanelle": 6, "spoken-word": 8,
            "pop-song": 4, "ballad": 6, "rap-verse": 8, "hymn": 4, "jingle": 2,
            "novel-chapter": 8, "short-story": 7, "flash-fiction": 4,
            "screenplay": 3, "creative-nonfiction": 6,
        },
    },
    "discursive_targets": {
        "description": "Required discursive expressions per 300-word page, keyed by doc_type",
        "values": {
            "concept-note": 2.0, "full-proposal": 2.0, "eoi": 1.5,
            "executive-summary": 2.0, "general": 1.0, "annual-report": 1.0,
            "monitoring-report": 0.5, "financial-report": 0.0, "assessment": 1.0,
            "tor": 0.5, "governance-review": 1.0,
            "facebook-post": 0.0, "linkedin-post": 0.5, "instagram-caption": 0.0,
            "haiku": 0.0, "sonnet": 0.0, "free-verse": 0.0, "villanelle": 0.0,
            "spoken-word": 0.0,
            "pop-song": 0.0, "ballad": 0.0, "rap-verse": 0.0, "hymn": 0.0, "jingle": 0.0,
            "novel-chapter": 0.0, "short-story": 0.0, "flash-fiction": 0.0,
            "screenplay": 0.0, "creative-nonfiction": 0.3,
        },
    },
    "hedging_words_en": {
        "description": "EN epistemic hedges; AI strips these out — their absence is the signal",
        "items": [
            "arguably", "perhaps", "seemingly", "apparently", "presumably", "plausibly",
            "possibly", "tentatively", "conceivably", "tends to", "appears to",
            "may suggest", "might indicate", "in many cases", "to some extent",
            "for the most part", "it could be argued", "one might argue", "this suggests",
        ],
    },
    "hedging_words_pt": {
        "description": "PT epistemic hedges; AI strips these out — their absence is the signal",
        "items": [
            "possivelmente", "provavelmente", "aparentemente", "presumivelmente",
            "talvez", "porventura", "tende a", "parece", "pode sugerir",
            "poderia indicar", "em muitos casos", "em certa medida", "em grande parte",
            "pode-se argumentar", "isto sugere", "ao que tudo indica",
        ],
    },
    "hedging_targets": {
        "description": "Required hedge density per 300-word page, keyed by doc_type (0.0 = not expected)",
        "values": {
            "concept-note": 1.5, "full-proposal": 1.5, "eoi": 1.0,
            "executive-summary": 1.5, "general": 1.0, "annual-report": 1.0,
            "monitoring-report": 0.5, "financial-report": 0.0, "assessment": 1.5,
            "tor": 0.5, "governance-review": 1.5,
            "facebook-post": 0.0, "linkedin-post": 0.0, "instagram-caption": 0.0,
            "haiku": 0.0, "sonnet": 0.0, "free-verse": 0.0, "villanelle": 0.0,
            "spoken-word": 0.0,
            "pop-song": 0.0, "ballad": 0.0, "rap-verse": 0.0, "hymn": 0.0, "jingle": 0.0,
            "novel-chapter": 0.0, "short-story": 0.0, "flash-fiction": 0.0,
            "screenplay": 0.0, "creative-nonfiction": 0.0,
        },
    },
    "juridiques_terms": {
        "description": "Portuguese juridiquês / bureaucratic terms AI over-uses",
        "items": [
            "outrossim", "hodierno", "mister", "doravante", "dessarte", "destarte",
            "consoante", "supracitado", "infracitado", "porquanto", "conquanto",
            "deveras", "haja vista", "ipso facto", "ex vi", "prima facie",
            "mutatis mutandis", "salvo melhor juízo", "nos termos da lei",
            "em conformidade com o disposto", "vem respeitosamente", "à guisa de",
            "ad hoc",
        ],
    },
    "pt_passive_patterns": {
        "description": "PT synthetic-passive regex patterns (foi realizado, são implementadas, etc.)",
        "items": [
            r"\b(?:é|são|foi|foram|será|serão|seria|seriam|sendo|sido)\s+\w+(?:ado|ido|ada|ida|ados|idas)\b",
        ],
    },
    "nominalisation_suffixes": {
        "description": "PT nominalisation suffix regex alternation (high ratio = abstract, verb-poor prose)",
        "items": [
            r"ção|ções|mento|mentos|dade|dades|eza|ezas|ismo|ismos|tura|turas|ncia|ncias",
        ],
    },
    "config": {
        "description": "Scalar thresholds used by detectors",
        "values": {
            "burstiness_cov_threshold": 0.45,
            "juridiques_hits_per_page_max": 2.0,
            "nominalisation_low": 0.15,
            "nominalisation_high": 0.40,
        },
    },
}


def write_files(force: bool = False, dry_run: bool = False) -> tuple[int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for name, body in DEFAULTS.items():
        path = OUT_DIR / f"{name}.json"
        if path.exists() and not force:
            print(f"skip  {path.relative_to(REPO_ROOT)} (exists; use --force to overwrite)")
            skipped += 1
            continue
        payload = {"description": body["description"]}
        if "items" in body:
            payload["items"] = body["items"]
        if "values" in body:
            payload["values"] = body["values"]
        if dry_run:
            print(f"WOULD write {path.relative_to(REPO_ROOT)} ({len(json.dumps(payload))} bytes)")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed data/patterns/*.json from inline defaults")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written")
    args = parser.parse_args()
    written, skipped = write_files(force=args.force, dry_run=args.dry_run)
    print(f"\ndone: {written} written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
