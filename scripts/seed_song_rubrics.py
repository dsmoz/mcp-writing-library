"""
Seed script: populate writing_rubrics collection with songwriting evaluation criteria.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_song_rubrics.py
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
        # --- Hook ---
        dict(
            framework="songwriting",
            section="hook",
            criterion="The hook is memorable as a standalone phrase; it encapsulates the song's core emotion or theme.",
            weight=2.0,
            red_flags=["forgettable hook", "no hook", "hook buried", "hook unclear"],
        ),
        dict(
            framework="songwriting",
            section="hook",
            criterion="The hook is placed at the most emotionally charged point of the song (typically chorus peak).",
            weight=1.5,
            red_flags=["hook not at peak", "hook buried in verse", "weak placement"],
        ),
        dict(
            framework="songwriting",
            section="hook",
            criterion="The hook phrase is singable: short enough to be remembered, rhythmically satisfying, and phonetically clean.",
            weight=1.5,
            red_flags=["too long to sing", "awkward phrasing", "hard to pronounce", "forced rhyme"],
        ),
        # --- Lyric clarity ---
        dict(
            framework="songwriting",
            section="lyric-clarity",
            criterion="The lyric's central metaphor or story is clear after one listen; no ambiguity is unintentional.",
            weight=2.0,
            red_flags=["too abstract", "unclear metaphor", "confusing narrative"],
        ),
        dict(
            framework="songwriting",
            section="lyric-clarity",
            criterion="Concrete nouns ground the emotional journey; the lyric does not rely exclusively on abstract language.",
            weight=1.5,
            red_flags=["all abstract", "no imagery", "no concrete details"],
        ),
        dict(
            framework="songwriting",
            section="lyric-clarity",
            criterion="Verses advance the story or deepen the perspective; they do not repeat information already established.",
            weight=1.0,
            red_flags=["verse repetition", "no narrative progression", "verses redundant"],
        ),
        # --- Emotional arc ---
        dict(
            framework="songwriting",
            section="emotional-arc",
            criterion="The song moves from an initial emotional state to a resolved, transformed, or deepened one by the end.",
            weight=2.0,
            red_flags=["no arc", "flat emotion", "same feeling throughout", "no resolution"],
        ),
        dict(
            framework="songwriting",
            section="emotional-arc",
            criterion="The bridge offers contrast in perspective, register, or emotional intensity relative to verses and chorus.",
            weight=1.5,
            red_flags=["bridge identical to verse", "no bridge contrast", "bridge adds nothing"],
        ),
        dict(
            framework="songwriting",
            section="emotional-arc",
            criterion="The final chorus or outro carries added emotional weight — it is not identical to the opening chorus.",
            weight=1.0,
            red_flags=["outro identical to intro", "no emotional growth", "flat ending"],
        ),
        # --- Structure ---
        dict(
            framework="songwriting",
            section="structure",
            criterion="Verse-chorus relationship is clear: verses set up the emotional context, chorus delivers the payoff.",
            weight=1.5,
            red_flags=["verse and chorus feel interchangeable", "unclear structure", "no payoff"],
        ),
        dict(
            framework="songwriting",
            section="structure",
            criterion="Repetition is intentional and cumulative: repeated sections feel more resonant each time, not merely recycled.",
            weight=1.0,
            red_flags=["repetition feels mechanical", "no cumulative effect", "padded"],
        ),
        # --- Singability ---
        dict(
            framework="songwriting",
            section="singability",
            criterion="Syllable stress aligns with natural speech rhythm; phrasing is breathable without awkward breaks.",
            weight=2.0,
            red_flags=["stress mismatch", "awkward phrasing", "not singable", "forced syllable count"],
        ),
        dict(
            framework="songwriting",
            section="singability",
            criterion="Lines that will be sung to the same melody have comparable syllable counts (±2 syllables).",
            weight=1.5,
            red_flags=["inconsistent syllable count", "melody mismatch", "too long", "too short"],
        ),
        dict(
            framework="songwriting",
            section="singability",
            criterion="End words in rhyming positions use true or near-rhymes; forced rhymes that distort natural speech are avoided.",
            weight=1.5,
            red_flags=["forced rhyme", "inverted syntax for rhyme", "eye-rhyme only", "no rhyme where expected"],
        ),
    ]

    print(f"\nSeeding {len(criteria)} songwriting rubric criteria...")
    succeeded = 0
    failed = 0

    for c in criteria:
        result = add_rubric_criterion(**c)
        section = c["section"]
        snippet = c["criterion"][:60]
        if result.get("success"):
            print(f"  [OK] [songwriting | {section}] {snippet}...")
            succeeded += 1
        else:
            print(f"  [FAIL] [songwriting | {section}] {snippet}... ERROR: {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    seed()
