"""
Seed script: populate writing_templates collection with initial templates for 4 donors.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_templates.py
"""
import sys
import os

# Ensure project root is on path
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
        # UNDP — concept-note
        dict(
            donor="undp",
            doc_type="concept-note",
            sections=[
                {
                    "name": "Executive Summary",
                    "description": "Brief overview of the intervention, its objectives, expected results, and budget",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Problem Statement",
                    "description": "Evidence-based analysis of the development problem, root causes, and affected populations",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Proposed Intervention",
                    "description": "Description of activities, approaches, and methodologies to address the problem",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Expected Results",
                    "description": "Outputs and outcomes with SMART indicators linked to the results chain",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "Implementation Arrangements",
                    "description": "Organisational structure, roles, responsibilities, and partnerships",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Budget Overview",
                    "description": "Summary budget with cost categories and funding sources",
                    "required": True,
                    "order": 6,
                },
            ],
        ),
        # Global Fund — concept-note
        dict(
            donor="global-fund",
            doc_type="concept-note",
            sections=[
                {
                    "name": "Disease Context",
                    "description": "Epidemiological data on the target disease burden and key populations affected",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Key Populations",
                    "description": "Evidence on barriers faced by key populations and community-led response capacity",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Program Description",
                    "description": "Detailed description of interventions, service delivery models, and target coverage",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Community Leadership",
                    "description": "How communities most affected lead and govern the program",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "M&E Framework",
                    "description": "Monitoring and evaluation plan with indicators, data sources, and reporting frequency",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Sustainability",
                    "description": "Plan for financial and programmatic sustainability beyond the grant period",
                    "required": True,
                    "order": 6,
                },
                {
                    "name": "Budget",
                    "description": "Detailed budget with cost per output and efficiency justification",
                    "required": True,
                    "order": 7,
                },
            ],
        ),
        # USAID — full-proposal
        dict(
            donor="usaid",
            doc_type="full-proposal",
            sections=[
                {
                    "name": "Technical Approach",
                    "description": "Detailed methodology, theory of change, and evidence base for the proposed approach",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Organizational Capacity",
                    "description": "Past performance, organizational structure, and relevant experience",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Personnel",
                    "description": "Key personnel qualifications, roles, and percentage of effort",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Management Plan",
                    "description": "Project management structure, risk management, and quality assurance",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "Monitoring and Evaluation",
                    "description": "M&E framework, data collection methods, and learning agenda",
                    "required": True,
                    "order": 5,
                },
            ],
        ),
        # EU — eoi
        dict(
            donor="eu",
            doc_type="eoi",
            sections=[
                {
                    "name": "Relevance",
                    "description": "Alignment with EU priorities, national strategies, and evidence of need",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Methodology",
                    "description": "Proposed approach, activities, and implementation timeline",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Expected Results",
                    "description": "Outputs, outcomes, and sustainability beyond the project period",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Organisational Profile",
                    "description": "Organisation description, experience, and relevant partnerships",
                    "required": True,
                    "order": 4,
                },
            ],
        ),
    ]

    succeeded = 0
    failed = 0

    for tmpl in templates:
        donor = tmpl["donor"]
        doc_type = tmpl["doc_type"]
        sections = tmpl["sections"]
        result = add_template(donor=donor, doc_type=doc_type, sections=sections)
        if result.get("success"):
            print(f"  [OK] {donor} / {doc_type} — {result['section_count']} sections")
            succeeded += 1
        else:
            print(f"  [FAIL] {donor} / {doc_type} — {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} templates added, {failed} failed.")


if __name__ == "__main__":
    seed()
