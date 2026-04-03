"""
Seed script: populate writing_templates collection with creative writing structural templates.

Covers: sonnet, villanelle, pop-song, short-story, flash-fiction, screenplay.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_creative_templates.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tools.collections import setup_collections
from src.tools.templates import add_template


def seed():
    print("Setting up collections...")
    setup_result = setup_collections()
    template_status = setup_result.get("templates", {}).get("status", "unknown")
    print(f"  writing_templates: {template_status}")

    templates = [
        # --- Poetry: Sonnet ---
        dict(
            framework="poetry-classic",
            doc_type="sonnet",
            sections=[
                {
                    "name": "Octave",
                    "description": "Lines 1–8: establish the problem, tension, or opening observation. Two quatrains with ABAB CDCD rhyme.",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Volta",
                    "description": "The turn or pivot between octave and sestet — the 'but', 'yet', or 'however' moment that shifts perspective.",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Sestet",
                    "description": "Lines 9–14: resolution, complication, or deepening of the octave's tension. EFEF GG rhyme.",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Closing Couplet",
                    "description": "Lines 13–14: the epigrammatic resolution or twist. AA rhyme. Should not merely summarise.",
                    "required": True,
                    "order": 4,
                },
            ],
        ),
        # --- Poetry: Villanelle ---
        dict(
            framework="poetry-classic",
            doc_type="villanelle",
            sections=[
                {
                    "name": "Opening Tercet",
                    "description": "Three lines establishing A1 (line 1) and A2 (line 3) refrains with ABA rhyme. Sets the poem's emotional stakes.",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Development Tercets",
                    "description": "Four tercets (stanzas 2–5) each ending with alternating A1 and A2 refrains. Each tercet deepens or complicates the theme.",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Closing Quatrain",
                    "description": "Final four lines bringing both A1 and A2 refrains together (ABAA). The refrains must earn new meaning by their final appearance.",
                    "required": True,
                    "order": 3,
                },
            ],
        ),
        # --- Songwriting: Pop Song ---
        dict(
            framework="songwriting",
            doc_type="pop-song",
            sections=[
                {
                    "name": "Verse 1",
                    "description": "Establishes the scene, character, or situation. Sets up the emotional context for the chorus. Specific and concrete.",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Pre-Chorus",
                    "description": "Builds tension toward the chorus. Raises the emotional stakes. Optional but powerful for pop structure.",
                    "required": False,
                    "order": 2,
                },
                {
                    "name": "Chorus",
                    "description": "The emotional and melodic peak. Contains the hook. Broad enough to be universal, specific enough to feel true.",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Verse 2",
                    "description": "Advances the story or deepens the perspective from Verse 1. Does not repeat the same information.",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "Bridge",
                    "description": "Contrast section — different melody, rhythm, perspective, or emotional register. Offers revelation or shift before final chorus.",
                    "required": False,
                    "order": 5,
                },
                {
                    "name": "Final Chorus / Outro",
                    "description": "Returns to the chorus with added emotional weight. May extend, vary, or resolve the hook. Outro winds down or intensifies.",
                    "required": True,
                    "order": 6,
                },
            ],
        ),
        # --- Prose Fiction: Short Story ---
        dict(
            framework="fiction",
            doc_type="short-story",
            sections=[
                {
                    "name": "Opening Hook",
                    "description": "Establishes voice, situation, or tension in the first paragraph. Orients the reader in time, place, and POV without backstory dumps.",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Rising Action",
                    "description": "Develops character, deepens conflict, and builds stakes. Each scene moves the story forward — no filler scenes.",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Complication / Turning Point",
                    "description": "The moment the central problem intensifies or a revelation changes the character's situation. The story cannot go back.",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Climax",
                    "description": "The highest point of tension: the confrontation, decision, or revelation the story has been building toward.",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "Resolution / Closing Image",
                    "description": "Resolves or deliberately leaves unresolved the central tension. A resonant closing image or line that echoes the opening.",
                    "required": True,
                    "order": 5,
                },
            ],
        ),
        # --- Prose Fiction: Flash Fiction ---
        dict(
            framework="fiction",
            doc_type="flash-fiction",
            sections=[
                {
                    "name": "Inciting Moment",
                    "description": "The single event or image that sets the story in motion. In flash, this often IS the opening line.",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Central Revelation",
                    "description": "The emotional or narrative core: a shift in understanding, a decision, or a collision of opposing forces.",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Resonant Closing",
                    "description": "The final image or line that expands the story beyond its word count. Should recontextualise the opening.",
                    "required": True,
                    "order": 3,
                },
            ],
        ),
        # --- Prose Fiction: Screenplay ---
        dict(
            framework="fiction",
            doc_type="screenplay",
            sections=[
                {
                    "name": "Act 1 — Setup",
                    "description": "Introduces protagonist, world, and ordinary life. Ends with the inciting incident that launches the story.",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "First Plot Point",
                    "description": "The point of no return: protagonist commits to the central journey. Story world expands or changes irrevocably.",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Act 2 — Confrontation",
                    "description": "Protagonist faces escalating obstacles. Each attempt to solve the problem reveals a deeper problem.",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Midpoint",
                    "description": "False victory or false defeat. Stakes double. Protagonist is changed by the midpoint event.",
                    "required": False,
                    "order": 4,
                },
                {
                    "name": "Second Plot Point",
                    "description": "All seems lost. Protagonist must face the central flaw or fear to move forward. Darkest moment.",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Act 3 — Resolution",
                    "description": "Climax and resolution. Protagonist applies what they have learned. Thematic statement is embodied in action.",
                    "required": True,
                    "order": 6,
                },
            ],
        ),
    ]

    print(f"\nSeeding {len(templates)} creative writing templates...")
    succeeded = 0
    failed = 0

    for t in templates:
        result = add_template(**t)
        label = f"{t['framework']} / {t['doc_type']}"
        if result.get("success"):
            print(f"  [OK] {label}")
            succeeded += 1
        else:
            print(f"  [FAIL] {label} — ERROR: {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    seed()
