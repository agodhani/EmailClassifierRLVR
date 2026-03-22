"""
Configuration for synthetic email generation pipeline.
Defines priority tiers, email categories, and LLM prompts for the
planner → generator → validator stages.
"""

# ── Priority Taxonomy ──────────────────────────────────────────────

PRIORITIES = {
    "P1": {
        "label": "P1 — Urgent / Action Required Now",
        "description": (
            "Requires immediate action within hours. Blocking others, "
            "time-sensitive deadlines, critical incidents, or urgent "
            "requests from leadership/key clients."
        ),
        "examples": [
            "Production outage alert",
            "Client threatening to cancel contract today",
            "CEO asking for board deck by EOD",
            "Security breach notification",
        ],
    },
    "P2": {
        "label": "P2 — Important / Action Required Soon",
        "description": (
            "Needs a response within 1-2 business days. Important but not "
            "immediately blocking. Project updates needing decisions, "
            "meeting requests from stakeholders, meaningful work requests."
        ),
        "examples": [
            "Manager asking for project status update by Friday",
            "Client requesting a feature discussion next week",
            "Code review request on a key PR",
            "Interview scheduling for a candidate",
        ],
    },
    "P3": {
        "label": "P3 — Low Priority / Can Wait",
        "description": (
            "Can be addressed within a week or deferred. FYI messages, "
            "non-urgent internal updates, optional meeting invites, "
            "informational newsletters relevant to your work."
        ),
        "examples": [
            "Team lunch poll",
            "Internal blog post about company values",
            "Optional training webinar invitation",
            "Weekly digest from a project you follow",
        ],
    },
    "P4": {
        "label": "P4 — Noise / No Action Needed",
        "description": (
            "No action required. Automated notifications, marketing emails, "
            "spam, mass CCs, out-of-office replies, social media alerts, "
            "subscription confirmations."
        ),
        "examples": [
            "GitHub bot: dependabot PR merged",
            "Marketing newsletter from a SaaS tool",
            "LinkedIn connection request notification",
            "Automated calendar reminder for a recurring meeting",
        ],
    },
}

# ── Email Categories (for fan-out diversity) ───────────────────────

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

# ── Target distribution per priority ──────────────────────────────
# Maps priority → how many (category, priority) cells to fill and
# how many scenarios per cell. Adjusted so totals ≈ 1000.

TARGET_PER_PRIORITY = {
    "P1": 250,
    "P2": 250,
    "P3": 250,
    "P4": 250,
}

# ── LLM Prompts ───────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are an expert at designing realistic, diverse email scenarios for \
a synthetic dataset. Your job is to produce scenario specifications \
that a separate generator model will turn into full email texts.

Each scenario spec must include:
- sender_role: who is sending (e.g. "Engineering Manager", "AWS CloudWatch", "LinkedIn")
- recipient_context: who is receiving and their role
- urgency_context: why this email has its assigned priority level
- key_details: 2-3 specific details to include (names, dates, numbers)
- tone: formal / casual / automated / urgent

Output valid JSON — an array of scenario objects. No commentary outside the JSON."""

PLANNER_USER = """\
Generate {n} diverse scenario specifications for emails that belong to:
- Category: {category} ({category_desc})
- Priority: {priority} ({priority_desc})

Examples of this priority level: {priority_examples}

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

Category: {category}
Priority: {priority}
Scenario: {scenario_json}

Write ONLY the email text."""

VALIDATOR_SYSTEM = """\
You are an email priority classifier. Given an email, classify it into \
exactly one priority tier:

P1 — Urgent / Action Required Now (blocking, time-critical, incidents)
P2 — Important / Action Required Soon (needs response in 1-2 days)
P3 — Low Priority / Can Wait (FYI, optional, can defer a week)
P4 — Noise / No Action Needed (automated, marketing, no action)

Think step-by-step about the urgency, sender importance, and required \
action, then output your classification.

Output format (exactly):
<priority>P1</priority>
<rationale>Your reasoning here</rationale>

Respond with ONLY the XML tags above."""

VALIDATOR_USER = """\
Classify this email's priority:

{email_text}"""

# ── Model Configuration ───────────────────────────────────────────

MODEL_PRIORITY = [
    "qwen/qwen3-32b",
]

API_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
