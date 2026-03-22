"""
Configuration for synthetic email generation pipeline.
Defines email categories and LLM prompts for the
planner → generator → validator stages.
"""

# ── Category Taxonomy ──────────────────────────────────────────────

CATEGORIES = [
    "todo_tasks",
    "meeting_coordination",
    "newsletter_information",
    "important_alert",
    "general",
    "transactional",
    "jobs",
]

CATEGORY_DESCRIPTIONS = {
    "todo_tasks": "Direct asks to do something — tasks, deliverables, approvals",
    "meeting_coordination": "Scheduling, rescheduling, agenda sharing, meeting follow-ups",
    "newsletter_information": "Internal or external newsletters, digests, roundups",
    "important_alert": "Notifications related to immediate actions, ie security codes, urgent checks for new logins etc, password resets",
    "general": "Personal emails landing in work inbox — family, friends, side projects",
    "transactional": "Receipts, order confirmations, subscription notices",
    "jobs": "Job related emails, updates, offers notifications",
}

# ── Target distribution per category ──────────────────────────────

TARGET_PER_CATEGORY = {cat: 100 for cat in CATEGORIES}

# ── LLM Prompts ───────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are an expert at designing realistic, diverse email scenarios for \
a synthetic dataset. Your job is to produce scenario specifications \
that a separate generator model will turn into full email texts.

Each scenario spec must include:
- sender_role: who is sending (e.g. "Engineering Manager", "AWS CloudWatch", "LinkedIn")
- recipient_context: who is receiving and their role
- key_details: 2-3 specific details to include (names, dates, numbers)
- tone: formal / casual / automated / urgent

Output valid JSON — an array of scenario objects. No commentary outside the JSON."""

PLANNER_USER = """\
Generate {n} diverse scenario specifications for emails that belong to:
- Category: {category} ({category_desc})

Return a JSON array of {n} scenario specification objects."""

GENERATOR_SYSTEM = """\
You are an expert at writing realistic emails. Given a scenario \
specification, produce a single realistic email with:
- Subject line
- From (name and email address)
- Date (use dates in 2024-2025)
- Body (realistic length and formatting for the scenario)

Output the email in this exact format:

Subject: <subject line>
From: <sender name> <sender@example.com>
Date: <date>

<email body>

Write ONLY the email. No commentary, no labels, no wrapping."""

GENERATOR_USER = """\
Write a realistic email based on this scenario:

Category: {category} ({category_desc})
Scenario: {scenario_json}

Write ONLY the email text."""

VALIDATOR_SYSTEM = """\
You are an email classifier. Given an email, classify it into \
exactly one of these categories:

todo_tasks — Direct asks to do something — tasks, deliverables, approvals
meeting_coordination — Scheduling, rescheduling, agenda sharing, meeting follow-ups
newsletter_information — Internal or external newsletters, digests, roundups
important_alert — Notifications related to immediate actions (security codes, login alerts, password resets)
general — Personal emails landing in work inbox — family, friends, side projects
transactional — Receipts, order confirmations, subscription notices
jobs — Job related emails, updates, offers, application notifications

Think step-by-step about the email's purpose and content, then output your classification.

Output format (exactly):
<category>todo_tasks</category>
<rationale>Your reasoning here</rationale>

Respond with ONLY the XML tags above."""

VALIDATOR_USER = """\
Classify this email's category:

{email_text}"""

# ── Model Configuration ───────────────────────────────────────────

MODEL_PRIORITY = [
    "qwen/qwen3-32b",
]

API_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
