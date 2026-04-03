"""
Seed script: populate writing_thesaurus collection.

Pipeline:
  1. Load curated word list from scripts/data/thesaurus_wordlist.py
  2. For EN entries: enrich definitions from Wordnik API (free, no key required)
  3. For PT entries: enrich synonyms from Dicionário Aberto XML (downloaded on first run)
  4. Import all entries via add_thesaurus_entry()

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_thesaurus.py

    # Dry run (no Qdrant writes):
    python scripts/seed_thesaurus.py --dry-run

    # Skip API enrichment (use wordlist as-is):
    python scripts/seed_thesaurus.py --no-enrich

    # Only import PT words:
    python scripts/seed_thesaurus.py --language pt
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests
import structlog

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tools.collections import setup_collections
from src.tools.thesaurus import add_thesaurus_entry

logger = structlog.get_logger(__name__)

WORDNIK_DEFINITIONS_URL = "https://api.wordnik.com/v4/word.json/{word}/definitions"
WORDNIK_API_KEY = os.getenv("WORDNIK_API_KEY", "")  # Optional; raises rate limit without key

DICIONARIO_ABERTO_URL = "https://raw.githubusercontent.com/ambs/Dicionario-Aberto/master/dic.xml"
DICIONARIO_ABERTO_CACHE = Path(__file__).parent / "data" / "dicionario_aberto.xml"


# ── Wordnik enrichment ────────────────────────────────────────────────────────

def _wordnik_definitions(word: str) -> list:
    """Fetch definitions from Wordnik API. Returns list of definition dicts or empty list."""
    url = WORDNIK_DEFINITIONS_URL.format(word=quote(word))
    params = {"limit": 3, "sourceDictionaries": "ahd-5,wordnet"}
    if WORDNIK_API_KEY:
        params["api_key"] = WORDNIK_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json() if isinstance(resp.json(), list) else []
        logger.warning("Wordnik non-200", status=resp.status_code, word=word)
    except Exception as e:
        logger.warning("Wordnik request failed", error=str(e), word=word)
    return []


def enrich_en(entry: dict) -> dict:
    """Enrich an EN entry with Wordnik definitions."""
    headword = entry["headword"]
    # Skip multi-word phrases for Wordnik lookup
    if " " in headword or "-" in headword:
        return entry

    data = _wordnik_definitions(headword)
    if data:
        definitions = [d.get("text", "") for d in data if d.get("text")]
        if definitions and not entry.get("definition"):
            entry["definition"] = definitions[0]
        if not entry.get("part_of_speech") and data[0].get("partOfSpeech"):
            pos = data[0]["partOfSpeech"].split("-")[0]  # "verb-transitive" → "verb"
            entry["part_of_speech"] = pos

    time.sleep(0.3)  # Wordnik rate limit: ~100 req/min without key
    return entry


# ── Dicionário Aberto enrichment ──────────────────────────────────────────────

def _download_dicionario_aberto() -> Path:
    """Download Dicionário Aberto XML to local cache if not present."""
    if DICIONARIO_ABERTO_CACHE.exists():
        print(f"  Using cached Dicionário Aberto: {DICIONARIO_ABERTO_CACHE}")
        return DICIONARIO_ABERTO_CACHE

    print("  Downloading Dicionário Aberto (~30MB)...")
    DICIONARIO_ABERTO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DICIONARIO_ABERTO_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(DICIONARIO_ABERTO_CACHE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded: {DICIONARIO_ABERTO_CACHE}")
    return DICIONARIO_ABERTO_CACHE


def _build_da_index(xml_path: Path) -> dict:
    """
    Parse Dicionário Aberto TEI XML and build a headword → {definition, synonyms} index.

    The TEI structure is:
      <entry>
        <form><orth>headword</orth></form>
        <sense><def>definition text</def></sense>
        <sense><usg type="syn">synonym1, synonym2</usg></sense>
      </entry>
    """
    print("  Parsing Dicionário Aberto XML (this may take 30-60s)...")
    index = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Strip namespace if present
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for entry in root.iter(f"{ns}entry"):
            orth = entry.find(f".//{ns}orth")
            if orth is None or not orth.text:
                continue
            headword = orth.text.strip().lower()

            definition = ""
            def_el = entry.find(f".//{ns}def")
            if def_el is not None and def_el.text:
                definition = def_el.text.strip()

            synonyms = []
            for usg in entry.findall(f".//{ns}usg"):
                if usg.get("type") == "syn" and usg.text:
                    synonyms.extend([s.strip() for s in usg.text.split(",")])

            index[headword] = {"definition": definition, "synonyms": synonyms}

    except ET.ParseError as e:
        print(f"  Warning: XML parse error — {e}. PT enrichment will be skipped.")
        return {}

    print(f"  Indexed {len(index):,} entries from Dicionário Aberto.")
    return index


def enrich_pt(entry: dict, da_index: dict) -> dict:
    """Enrich a PT entry using Dicionário Aberto index."""
    headword = entry["headword"].lower()
    da_entry = da_index.get(headword)
    if not da_entry:
        return entry

    if not entry.get("definition") and da_entry.get("definition"):
        entry["definition"] = da_entry["definition"]

    existing_words = {a["word"].lower() for a in entry.get("alternatives", [])}
    for syn in da_entry.get("synonyms", [])[:5]:
        if syn.lower() not in existing_words and syn.lower() != headword:
            entry.setdefault("alternatives", []).append({
                "word": syn,
                "meaning_nuance": "Sinónimo (Dicionário Aberto)",
                "register": "neutral",
                "when_to_use": "Consulte contexto — sinónimo de fonte lexicográfica",
            })
            existing_words.add(syn.lower())

    return entry


# ── Main seeding pipeline ─────────────────────────────────────────────────────

def seed(dry_run: bool = False, no_enrich: bool = False, language_filter: str | None = None):
    from scripts.data.thesaurus_wordlist import EN_WORDS, PT_WORDS

    print("Setting up collections...")
    if not dry_run:
        result = setup_collections()
        thesaurus_status = result.get("thesaurus", {}).get("status", "unknown")
        print(f"  writing_thesaurus: {thesaurus_status}")

    # Build Dicionário Aberto index (PT enrichment)
    da_index = {}
    if not no_enrich and (language_filter is None or language_filter == "pt"):
        try:
            xml_path = _download_dicionario_aberto()
            da_index = _build_da_index(xml_path)
        except Exception as e:
            print(f"  Warning: Dicionário Aberto download/parse failed — {e}. PT entries will use wordlist only.")

    all_entries = []
    if language_filter is None or language_filter == "en":
        for entry in EN_WORDS:
            e = {**entry, "language": "en", "source": "wordnik" if not no_enrich else "manual"}
            if not no_enrich:
                e = enrich_en(e)
            all_entries.append(e)

    if language_filter is None or language_filter == "pt":
        for entry in PT_WORDS:
            e = {**entry, "language": "pt", "source": "dicionario-aberto" if da_index else "manual"}
            if not no_enrich and da_index:
                e = enrich_pt(e, da_index)
            all_entries.append(e)

    print(f"\nSeeding {len(all_entries)} entries (dry_run={dry_run})...")
    succeeded = 0
    failed = 0
    skipped = 0

    for entry in all_entries:
        label = f"{entry['headword']} ({entry['language']})"
        if dry_run:
            print(f"  [DRY RUN] Would insert: {label}")
            succeeded += 1
            continue

        result = add_thesaurus_entry(
            headword=entry["headword"],
            language=entry["language"],
            domain=entry.get("domain", "general"),
            definition=entry.get("definition", ""),
            part_of_speech=entry.get("part_of_speech", "verb"),
            register=entry.get("register", "neutral"),
            alternatives=entry.get("alternatives", []),
            collocations=entry.get("collocations", []),
            why_avoid=entry.get("why_avoid", ""),
            example_bad=entry.get("example_bad", ""),
            example_good=entry.get("example_good", ""),
            source=entry.get("source", "manual"),
        )

        if result.get("success"):
            print(f"  ✓ {label}")
            succeeded += 1
        elif "already exists" in result.get("error", ""):
            print(f"  ~ {label} (already exists — skipped)")
            skipped += 1
        else:
            print(f"  ✗ {label}: {result.get('error')}")
            failed += 1

    print(f"\nDone. succeeded={succeeded} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed writing_thesaurus collection")
    parser.add_argument("--dry-run", action="store_true", help="Print entries without inserting")
    parser.add_argument("--no-enrich", action="store_true", help="Skip Wordnik and Dicionário Aberto enrichment")
    parser.add_argument("--language", choices=["en", "pt"], help="Only seed one language")
    args = parser.parse_args()
    seed(dry_run=args.dry_run, no_enrich=args.no_enrich, language_filter=args.language)
