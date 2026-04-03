"""
Seed script: populate writing_rubrics collection with prose fiction evaluation criteria.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_fiction_rubrics.py
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
        # --- Show vs tell ---
        dict(
            framework="fiction",
            section="show-vs-tell",
            criterion="Emotional states are rendered through physical action, dialogue, or sensory detail rather than direct labelling.",
            weight=2.0,
            red_flags=["she felt sad", "he was angry", "direct emotion label", "telling not showing"],
        ),
        dict(
            framework="fiction",
            section="show-vs-tell",
            criterion="The narrator does not explain what the scene already demonstrates; trust the reader to make the inference.",
            weight=1.5,
            red_flags=["over-explained", "authorial intrusion", "meaning spelled out"],
        ),
        dict(
            framework="fiction",
            section="show-vs-tell",
            criterion="Filter words (she noticed, he saw, she felt) are minimised; the reader experiences events directly.",
            weight=1.5,
            red_flags=["filter words", "she noticed", "he saw that", "she felt that"],
        ),
        # --- Pacing ---
        dict(
            framework="fiction",
            section="pacing",
            criterion="Scene length is proportionate to narrative importance; no scene overstays its purpose.",
            weight=1.5,
            red_flags=["scene too long", "overstays purpose", "dragging", "no narrative urgency"],
        ),
        dict(
            framework="fiction",
            section="pacing",
            criterion="Transitions between scenes earn their compression; jumps in time or place are signalled clearly.",
            weight=1.0,
            red_flags=["jarring transition", "unexplained jump", "disorienting cut"],
        ),
        dict(
            framework="fiction",
            section="pacing",
            criterion="Dialogue scenes alternate with action or interiority to vary rhythm; no scene is all dialogue or all description.",
            weight=1.0,
            red_flags=["wall of dialogue", "wall of description", "no rhythm variation"],
        ),
        # --- Character voice ---
        dict(
            framework="fiction",
            section="character-voice",
            criterion="Each character's dialogue is syntactically and lexically distinct — no two characters sound interchangeable.",
            weight=2.0,
            red_flags=["all characters sound alike", "no voice distinction", "interchangeable dialogue"],
        ),
        dict(
            framework="fiction",
            section="character-voice",
            criterion="The POV character's interiority is consistent with their established voice, knowledge, and emotional context.",
            weight=1.5,
            red_flags=["inconsistent POV voice", "knows too much", "out of character"],
        ),
        dict(
            framework="fiction",
            section="character-voice",
            criterion="Character vocabulary and syntax reflect their background, education, and emotional state in the scene.",
            weight=1.0,
            red_flags=["anachronistic vocabulary", "wrong register for character", "inconsistent education level"],
        ),
        # --- Dialogue ---
        dict(
            framework="fiction",
            section="dialogue",
            criterion="Dialogue serves dual purpose: character revelation AND plot or scene function simultaneously.",
            weight=2.0,
            red_flags=["dialogue only expository", "no character revelation", "flat dialogue"],
        ),
        dict(
            framework="fiction",
            section="dialogue",
            criterion="Attribution is lean; action beats replace said-bookisms where possible ('he said' over 'he exclaimed breathlessly').",
            weight=1.5,
            red_flags=["said-bookisms", "excessive adverbs on tags", "theatrical attribution"],
        ),
        dict(
            framework="fiction",
            section="dialogue",
            criterion="Subtext is present: characters do not always say what they mean; tension lives beneath the spoken words.",
            weight=1.5,
            red_flags=["on-the-nose dialogue", "no subtext", "characters explain everything"],
        ),
        # --- Narrative distance ---
        dict(
            framework="fiction",
            section="narrative-distance",
            criterion="POV is consistent within a scene; head-hopping between characters is intentional or absent.",
            weight=2.0,
            red_flags=["head-hopping", "POV shift mid-scene", "inconsistent POV"],
        ),
        dict(
            framework="fiction",
            section="narrative-distance",
            criterion="Narrative summary is used purposefully to compress time; it does not substitute for scene-level showing.",
            weight=1.0,
            red_flags=["summary as substitute for scene", "told not shown at scene level"],
        ),
        # --- Opening ---
        dict(
            framework="fiction",
            section="opening",
            criterion="The first paragraph establishes voice, situation, or tension without over-explaining backstory.",
            weight=2.0,
            red_flags=["starts with backstory", "info-dump opening", "no hook", "starts too early"],
        ),
        dict(
            framework="fiction",
            section="opening",
            criterion="The reader is oriented in time, place, and POV within the first page without an explicit inventory of details.",
            weight=1.5,
            red_flags=["disorienting opening", "no grounding", "unclear when/where", "unclear POV"],
        ),
    ]

    print(f"\nSeeding {len(criteria)} fiction rubric criteria...")
    succeeded = 0
    failed = 0

    for c in criteria:
        result = add_rubric_criterion(**c)
        section = c["section"]
        snippet = c["criterion"][:60]
        if result.get("success"):
            print(f"  [OK] [fiction | {section}] {snippet}...")
            succeeded += 1
        else:
            print(f"  [FAIL] [fiction | {section}] {snippet}... ERROR: {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    seed()
