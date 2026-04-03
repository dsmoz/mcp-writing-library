"""
Seed script: populate writing_rubrics collection with poetry evaluation criteria.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_poetry_rubrics.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tools.collections import setup_collections
from src.tools.rubrics import add_rubric_criterion


def seed():
    print("Setting up collections...")
    setup_result = setup_collections()
    rubric_status = setup_result.get("rubrics", {}).get("status", "unknown")
    print(f"  writing_rubrics: {rubric_status}")

    criteria = [
        # --- Imagery ---
        dict(
            framework="poetry",
            section="imagery",
            criterion="Concrete sensory images anchor abstract meaning; abstractions are earned through prior specific detail.",
            weight=2.0,
            red_flags=["abstract without grounding", "vague imagery", "no sensory detail"],
        ),
        dict(
            framework="poetry",
            section="imagery",
            criterion="Imagery avoids the stock (moon, heart, tears alone); images are specific, unexpected, and earned.",
            weight=1.5,
            red_flags=["clichéd imagery", "overused metaphor", "stock images"],
        ),
        dict(
            framework="poetry",
            section="imagery",
            criterion="The central metaphor or image is extended or transformed rather than dropped after first use.",
            weight=1.5,
            red_flags=["mixed metaphor", "abandoned metaphor", "inconsistent imagery"],
        ),
        dict(
            framework="poetry",
            section="imagery",
            criterion="Visual, auditory, tactile, and olfactory senses are engaged — not only sight.",
            weight=1.0,
            red_flags=["sight only", "no sensory variety"],
        ),
        # --- Rhythm ---
        dict(
            framework="poetry",
            section="rhythm",
            criterion="Line breaks serve meaning and breath; enjambment creates tension or surprise rather than arbitrary line splits.",
            weight=2.0,
            red_flags=["arbitrary line breaks", "end-stopped every line", "no enjambment effect"],
        ),
        dict(
            framework="poetry",
            section="rhythm",
            criterion="For formal poems (sonnet, villanelle): meter is consistent or deliberately varied for expressive effect.",
            weight=1.5,
            red_flags=["inconsistent meter without effect", "broken meter mid-stanza"],
        ),
        dict(
            framework="poetry",
            section="rhythm",
            criterion="Sound devices (alliteration, assonance, consonance) reinforce meaning rather than feeling decorative.",
            weight=1.0,
            red_flags=["forced alliteration", "decorative sound without meaning"],
        ),
        # --- Thematic coherence ---
        dict(
            framework="poetry",
            section="thematic-coherence",
            criterion="Opening and closing images relate; the poem turns or pivots between them — movement is visible.",
            weight=2.0,
            red_flags=["no development", "static poem", "no turn", "no volta"],
        ),
        dict(
            framework="poetry",
            section="thematic-coherence",
            criterion="The poem's subject (what it is about) and its emotional stakes (why it matters) are distinct but connected.",
            weight=1.5,
            red_flags=["subject not clear", "no emotional stakes", "flat"],
        ),
        dict(
            framework="poetry",
            section="thematic-coherence",
            criterion="The poem does not over-explain its meaning; the reader is trusted to make the connection.",
            weight=1.5,
            red_flags=["over-explained", "didactic", "tells the reader what to feel"],
        ),
        # --- Voice ---
        dict(
            framework="poetry",
            section="voice",
            criterion="The speaker's relationship to the subject is clear and sustained throughout the poem.",
            weight=1.5,
            red_flags=["inconsistent speaker", "unclear POV", "voice shift without purpose"],
        ),
        dict(
            framework="poetry",
            section="voice",
            criterion="The poem has a distinctive register — it does not sound interchangeable with any other poem.",
            weight=1.0,
            red_flags=["generic voice", "interchangeable", "no distinctiveness"],
        ),
        # --- Form ---
        dict(
            framework="poetry",
            section="form",
            criterion="For haiku: three images in 5-7-5 syllable structure; a seasonal or nature kigo is present or strongly implied.",
            weight=2.0,
            red_flags=["wrong syllable count", "no kigo", "no juxtaposition"],
        ),
        dict(
            framework="poetry",
            section="form",
            criterion="For sonnet: a volta (turn) is present between octave and sestet or in the closing couplet; rhyme scheme is respected.",
            weight=2.0,
            red_flags=["no volta", "broken rhyme scheme", "no turn"],
        ),
        dict(
            framework="poetry",
            section="form",
            criterion="For villanelle: both refrains (A1 and A2) recur in the correct positions; final quatrain brings them together.",
            weight=2.0,
            red_flags=["missing refrain", "incorrect refrain position", "broken villanelle structure"],
        ),
        # --- Compression ---
        dict(
            framework="poetry",
            section="compression",
            criterion="No line is padding; each word earns its place — the poem could not be shortened without loss.",
            weight=2.0,
            red_flags=["padding", "filler lines", "could be shorter", "unnecessary words"],
        ),
        dict(
            framework="poetry",
            section="compression",
            criterion="Pronouns, articles, and prepositions are stripped where ambiguity serves the poem; nothing is redundant.",
            weight=1.0,
            red_flags=["redundant phrases", "over-explained"],
        ),
    ]

    print(f"\nSeeding {len(criteria)} poetry rubric criteria...")
    succeeded = 0
    failed = 0

    for c in criteria:
        result = add_rubric_criterion(**c)
        section = c["section"]
        snippet = c["criterion"][:60]
        if result.get("success"):
            print(f"  [OK] [poetry | {section}] {snippet}...")
            succeeded += 1
        else:
            print(f"  [FAIL] [poetry | {section}] {snippet}... ERROR: {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    seed()
