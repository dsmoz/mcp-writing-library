"""Seed EN epistemic hedges into the writing_thesaurus collection.

Hedges are NOT words-to-avoid — they are words AI strips out. Stored in the
thesaurus so writers can look up nuance, register, and when-to-use guidance.
Detection of their absence lives in hedging_words_en.json (ai_patterns.py).

Usage:
    python scripts/seed_hedging_thesaurus.py            # write
    python scripts/seed_hedging_thesaurus.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.tools.thesaurus import add_thesaurus_entry  # noqa: E402


HEDGES: list[dict] = [
    {
        "headword": "arguably",
        "definition": "Signals the writer knows a claim is contestable and invites counter-evidence.",
        "part_of_speech": "adverb", "register": "academic",
        "when_to_use": "Introducing a strong-but-defeasible interpretation.",
        "example_good": "The programme is arguably the most effective SRHR investment in the province.",
        "example_bad": "The programme is the most effective SRHR investment in the province.",
        "why_keep": "Marks the claim as argued, not declared — reads as human analysis.",
    },
    {
        "headword": "perhaps",
        "definition": "Low-confidence marker that softens speculative claims.",
        "part_of_speech": "adverb", "register": "neutral",
        "when_to_use": "Raising a possibility without committing to it.",
        "example_good": "Perhaps the donor's silence reflects a shift in priorities.",
        "example_bad": "The donor's silence reflects a shift in priorities.",
        "why_keep": "Signals interpretive restraint; its absence flags AI-flavoured certainty.",
    },
    {
        "headword": "seemingly",
        "definition": "Marks appearance without endorsing reality.",
        "part_of_speech": "adverb", "register": "neutral",
        "when_to_use": "When evidence is visible but interpretation is uncertain.",
        "example_good": "The seemingly steady uptake masks heavy attrition at month three.",
    },
    {
        "headword": "tends to",
        "definition": "Indicates a pattern, not a universal rule.",
        "part_of_speech": "verb-phrase", "register": "neutral",
        "when_to_use": "Describing tendencies across heterogeneous data.",
        "example_good": "Adolescent girls tend to disengage once their partners object.",
    },
    {
        "headword": "appears to",
        "definition": "Observational marker: what the data surfaces, not what is true.",
        "part_of_speech": "verb-phrase", "register": "academic",
        "when_to_use": "Reporting what the evidence shows while preserving epistemic distance.",
        "example_good": "Uptake appears to plateau around week eight, though the sample is small.",
    },
    {
        "headword": "may suggest",
        "definition": "Tentative inferential frame.",
        "part_of_speech": "verb-phrase", "register": "academic",
        "when_to_use": "When interpretation is plausible but not established.",
        "example_good": "The drop-off may suggest stigma at the facility level rather than service failure.",
    },
    {
        "headword": "might indicate",
        "definition": "Alternative tentative inferential frame.",
        "part_of_speech": "verb-phrase", "register": "academic",
        "when_to_use": "Offering a reading without claiming certainty.",
        "example_good": "The seasonal dip might indicate harvest-period labour competing with clinic visits.",
    },
    {
        "headword": "presumably",
        "definition": "Assumption flagged as an assumption.",
        "part_of_speech": "adverb", "register": "neutral",
        "when_to_use": "When a step in reasoning rests on a plausible but unverified premise.",
        "example_good": "Presumably the stockout propagated downstream before the audit caught it.",
    },
    {
        "headword": "plausibly",
        "definition": "A reading that fits the evidence but is not the only possible reading.",
        "part_of_speech": "adverb", "register": "academic",
        "when_to_use": "Offering a defensible interpretation without foreclosing alternatives.",
        "example_good": "Plausibly, the funding cut accelerated attrition already underway.",
    },
    {
        "headword": "tentatively",
        "definition": "Provisional framing; invites revision.",
        "part_of_speech": "adverb", "register": "academic",
        "when_to_use": "First-look findings or pilot results.",
        "example_good": "We tentatively attribute the uplift to the peer-navigator roll-out.",
    },
    {
        "headword": "in many cases",
        "definition": "Asserts breadth without claiming universality.",
        "part_of_speech": "adverb-phrase", "register": "neutral",
        "when_to_use": "Generalising from varied data.",
        "example_good": "In many cases, the first barrier is distance, not stigma.",
    },
    {
        "headword": "to some extent",
        "definition": "Partial endorsement; avoids binary framings.",
        "part_of_speech": "adverb-phrase", "register": "neutral",
        "when_to_use": "When something is true in degree, not absolutely.",
        "example_good": "The training closed the gap, to some extent.",
    },
    {
        "headword": "for the most part",
        "definition": "Covers majority behaviour while leaving room for exceptions.",
        "part_of_speech": "adverb-phrase", "register": "neutral",
        "when_to_use": "Summarising uneven patterns.",
        "example_good": "For the most part, clinics reopened within a week of the floods.",
    },
    {
        "headword": "it could be argued",
        "definition": "Invites the reader into the argument as a participant, not a recipient.",
        "part_of_speech": "verb-phrase", "register": "academic",
        "when_to_use": "Surfacing an interpretation the writer is willing to defend but not assert flatly.",
        "example_good": "It could be argued that the framework privileges measurement over relationship.",
    },
    {
        "headword": "one might argue",
        "definition": "Third-person variant of 'it could be argued'.",
        "part_of_speech": "verb-phrase", "register": "academic",
        "when_to_use": "Attributing a view to a general reader or interlocutor.",
        "example_good": "One might argue that the scoring rubric flattens what the narrative reveals.",
    },
    {
        "headword": "this suggests",
        "definition": "Soft inferential bridge between data and interpretation.",
        "part_of_speech": "verb-phrase", "register": "academic",
        "when_to_use": "Connecting evidence to a reading without overstating the link.",
        "example_good": "This suggests a supply-side constraint, not a demand problem.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Qdrant")
    args = parser.parse_args()

    added = skipped = errored = 0
    for h in HEDGES:
        label = f"{h['headword']} [en]"
        if args.dry_run:
            print(f"WOULD add: {label}  ({h['register']})")
            added += 1
            continue
        result = add_thesaurus_entry(
            headword=h["headword"],
            language="en",
            domain="general",
            definition=h.get("definition", ""),
            part_of_speech=h.get("part_of_speech", "phrase"),
            register=h.get("register", "neutral"),
            alternatives=[],  # hedges stand alone — no substitutions
            collocations=[],
            why_avoid="",  # hedges are NOT to avoid; see why_keep in notes
            example_bad=h.get("example_bad", ""),
            example_good=h.get("example_good", ""),
            source="hedging-seed-2026",
        )
        status = result.get("status") or ("success" if result.get("success") else "error")
        if status == "exists" or result.get("exists"):
            print(f"skip   {label} (already in thesaurus)")
            skipped += 1
        elif result.get("success"):
            print(f"wrote  {label}")
            added += 1
        else:
            print(f"ERROR  {label}: {result.get('error')}")
            errored += 1

    print(f"\ndone: {added} added, {skipped} skipped, {errored} errored")
    return 0 if errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
