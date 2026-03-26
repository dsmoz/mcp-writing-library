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
            framework="undp",
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
            framework="global-fund",
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
            framework="usaid",
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
            framework="eu",
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
        # General — monitoring-report
        dict(
            framework="general",
            doc_type="monitoring-report",
            sections=[
                {
                    "name": "Executive Summary",
                    "description": "Brief overview of reporting period, headline results, and key issues",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Context Update",
                    "description": "Changes in the operating environment, risks materialised, and adaptations made",
                    "required": False,
                    "order": 2,
                },
                {
                    "name": "Progress Against Results",
                    "description": "Indicator-by-indicator progress table with actuals vs targets, disaggregated data, and variance explanations",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Activities Delivered",
                    "description": "Summary of activities completed, beneficiaries reached, and implementation quality",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "Financial Summary",
                    "description": "Budget vs expenditure by cost category with utilisation rates and explanations",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Lessons Learned",
                    "description": "What worked, what did not, and adaptations for the next period",
                    "required": False,
                    "order": 6,
                },
                {
                    "name": "Next Period Plan",
                    "description": "Planned activities and targets for the upcoming reporting period",
                    "required": True,
                    "order": 7,
                },
            ],
        ),
        # General — assessment
        dict(
            framework="general",
            doc_type="assessment",
            sections=[
                {
                    "name": "Introduction and Scope",
                    "description": "Assessment objectives, methodology, data sources, and scope of work",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Context",
                    "description": "Background information on the organisation, programme, or policy being assessed",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Findings",
                    "description": "Evidence-based findings organised by assessment dimension or thematic area",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Analysis",
                    "description": "Interpretation of findings against assessment criteria or standards",
                    "required": True,
                    "order": 4,
                },
                {
                    "name": "Conclusions",
                    "description": "Overall assessment verdict and key conclusions drawn from evidence",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Recommendations",
                    "description": "Actionable recommendations with responsible parties, timelines, and priority levels",
                    "required": True,
                    "order": 6,
                },
            ],
        ),
        # General — tor (Terms of Reference)
        dict(
            framework="general",
            doc_type="tor",
            sections=[
                {
                    "name": "Background",
                    "description": "Organisation or project context that makes this consultancy necessary",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Objectives",
                    "description": "Specific objectives the consultancy is intended to achieve",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Scope of Work",
                    "description": "Tasks, deliverables, and boundaries of the consultancy",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Methodology",
                    "description": "Expected approach, data collection methods, and quality standards",
                    "required": False,
                    "order": 4,
                },
                {
                    "name": "Deliverables",
                    "description": "Specific outputs with format, quality standards, and submission deadlines",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Qualifications",
                    "description": "Required expertise, experience, and competencies for the consultant",
                    "required": True,
                    "order": 6,
                },
                {
                    "name": "Administrative Details",
                    "description": "Duration, location, reporting line, budget envelope, and application instructions",
                    "required": True,
                    "order": 7,
                },
            ],
        ),
        # General — governance-review
        dict(
            framework="general",
            doc_type="governance-review",
            sections=[
                {
                    "name": "Scope and Methodology",
                    "description": "What governance dimensions were reviewed, how evidence was collected, and limitations",
                    "required": True,
                    "order": 1,
                },
                {
                    "name": "Governance Structure",
                    "description": "Board composition, decision-making authority, committees, and reporting lines",
                    "required": True,
                    "order": 2,
                },
                {
                    "name": "Financial Management",
                    "description": "Financial controls, internal audit, external audit, and budget oversight mechanisms",
                    "required": True,
                    "order": 3,
                },
                {
                    "name": "Human Resources",
                    "description": "HR policies, staff structure, performance management, and safeguarding mechanisms",
                    "required": False,
                    "order": 4,
                },
                {
                    "name": "Risk Management",
                    "description": "Risk register, risk appetite statement, and evidence of risk monitoring",
                    "required": True,
                    "order": 5,
                },
                {
                    "name": "Compliance",
                    "description": "Legal registration, regulatory compliance, and policy adherence findings",
                    "required": True,
                    "order": 6,
                },
                {
                    "name": "Recommendations",
                    "description": "Prioritised governance improvement actions with responsible parties and timelines",
                    "required": True,
                    "order": 7,
                },
            ],
        ),
    ]

    succeeded = 0
    failed = 0

    for tmpl in templates:
        framework = tmpl["framework"]
        doc_type = tmpl["doc_type"]
        sections = tmpl["sections"]
        result = add_template(framework=framework, doc_type=doc_type, sections=sections)
        if result.get("success"):
            print(f"  [OK] {framework} / {doc_type} — {result['section_count']} sections")
            succeeded += 1
        else:
            print(f"  [FAIL] {framework} / {doc_type} — {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} templates added, {failed} failed.")


if __name__ == "__main__":
    seed()
