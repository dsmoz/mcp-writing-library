"""
Seed script: populate writing_rubrics collection with initial criteria for 4 frameworks.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_rubrics.py
"""
import sys
import os

# Ensure project root is on path
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
        # USAID — technical-approach
        dict(
            framework="usaid",
            section="technical-approach",
            criterion="The proposal clearly articulates a theory of change linking activities to outputs, outcomes, and impact.",
            weight=1.5,
            red_flags=["vague", "unclear linkage", "no theory of change"],
        ),
        dict(
            framework="usaid",
            section="technical-approach",
            criterion="Past performance evidence is cited with specific results, dates, and implementing partners.",
            weight=1.5,
            red_flags=["no prior experience", "unverified claims"],
        ),
        dict(
            framework="usaid",
            section="technical-approach",
            criterion="The technical approach addresses the specific evaluation criteria from the RFP.",
            weight=2.0,
            red_flags=["generic", "copy-paste", "not responsive to RFP"],
        ),
        # UNDP — results-framework
        dict(
            framework="undp",
            section="results-framework",
            criterion="SMART indicators are defined with baselines, targets, data sources, and collection frequency.",
            weight=2.0,
            red_flags=["no baseline", "unmeasurable indicator", "vague target"],
        ),
        dict(
            framework="undp",
            section="results-framework",
            criterion="The results chain is logical: activities → outputs → outcomes → impact.",
            weight=1.5,
            red_flags=["missing outputs", "outcomes not linked to activities"],
        ),
        dict(
            framework="undp",
            section="results-framework",
            criterion="Risks and assumptions are explicitly identified with mitigation strategies.",
            weight=1.0,
            red_flags=["no risks identified", "no mitigation"],
        ),
        # Global Fund — community-led
        dict(
            framework="global-fund",
            section="community-led",
            criterion="Communities most affected by the disease are meaningfully involved in design and governance.",
            weight=2.0,
            red_flags=["top-down", "no community voice", "tokenistic"],
        ),
        dict(
            framework="global-fund",
            section="community-led",
            criterion="Sustainability plan demonstrates how community structures will continue post-grant.",
            weight=1.5,
            red_flags=["no sustainability", "framework-dependent", "unclear exit strategy"],
        ),
        dict(
            framework="global-fund",
            section="community-led",
            criterion="M&E plan includes community-based monitoring with disaggregated data by key population.",
            weight=1.5,
            red_flags=["no disaggregated data", "no KP data"],
        ),
        # EU — relevance
        dict(
            framework="eu",
            section="relevance",
            criterion="The intervention is aligned with EU development priorities and the partner country's national strategy.",
            weight=1.5,
            red_flags=["no alignment", "contradicts national strategy"],
        ),
        dict(
            framework="eu",
            section="relevance",
            criterion="Evidence base justifying the problem statement is current (within 5 years) and from credible sources.",
            weight=1.0,
            red_flags=["outdated data", "no citations", "anecdotal"],
        ),
        dict(
            framework="eu",
            section="relevance",
            criterion="The proposal demonstrates added value over existing interventions — no duplication.",
            weight=1.0,
            red_flags=["duplication", "no gap analysis"],
        ),
        # M&E Report — quality criteria (framework="general", section="m-and-e-report")
        dict(
            framework="general",
            section="m-and-e-report",
            criterion="Indicator data is disaggregated by sex, age, geography, and key population group as applicable.",
            weight=1.5,
            red_flags=["aggregate only", "no disaggregation", "no sex-disaggregation"],
        ),
        dict(
            framework="general",
            section="m-and-e-report",
            criterion="Actual results are compared to targets with a clear explanation of variances (exceeded, met, not met).",
            weight=2.0,
            red_flags=["no variance explanation", "missing targets", "results without targets"],
        ),
        dict(
            framework="general",
            section="m-and-e-report",
            criterion="Data quality limitations are disclosed, including collection methodology and potential biases.",
            weight=1.0,
            red_flags=["no data quality note", "unqualified data"],
        ),
        # Governance Review — criteria (framework="general", section="governance-review")
        dict(
            framework="general",
            section="governance-review",
            criterion="Board composition and decision-making authority are clearly documented and separation of powers is evident.",
            weight=1.5,
            red_flags=["no board documentation", "unclear authority", "no separation of powers"],
        ),
        dict(
            framework="general",
            section="governance-review",
            criterion="Financial oversight mechanisms include internal audit, external audit, and approval thresholds.",
            weight=2.0,
            red_flags=["no audit", "no financial controls", "unlimited spending authority"],
        ),
        dict(
            framework="general",
            section="governance-review",
            criterion="Conflict of interest policy is documented and there is evidence of its application.",
            weight=1.0,
            red_flags=["no conflict of interest policy", "no evidence of enforcement"],
        ),
        # Financial Report — criteria (framework="general", section="financial-report")
        dict(
            framework="general",
            section="financial-report",
            criterion="Expenditure is reported against approved budget lines with percentage utilisation rates.",
            weight=2.0,
            red_flags=["no budget comparison", "missing utilisation rates"],
        ),
        dict(
            framework="general",
            section="financial-report",
            criterion="Variances above 10% between budget and actuals are explained with justification and remediation plan.",
            weight=1.5,
            red_flags=["unexplained variance", "no justification for overspend"],
        ),
        dict(
            framework="general",
            section="financial-report",
            criterion="Supporting documentation is referenced for all major expenditure categories.",
            weight=1.0,
            red_flags=["no receipts referenced", "no supporting documentation"],
        ),
    ]

    print(f"\nSeeding {len(criteria)} criteria (proposals + M&E reports + governance + financial)...")
    succeeded = 0
    failed = 0

    for c in criteria:
        result = add_rubric_criterion(**c)
        framework = c["framework"].upper()
        section = c["section"]
        snippet = c["criterion"][:60]
        if result.get("success"):
            print(f"  [OK] [{framework} | {section}] {snippet}...")
            succeeded += 1
        else:
            print(f"  [FAIL] [{framework} | {section}] {snippet}... ERROR: {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    seed()
