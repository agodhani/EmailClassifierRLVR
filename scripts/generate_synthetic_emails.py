
#!/usr/bin/env python3
"""
Synthetic email generation pipeline.

Three-stage LLM pipeline:
  1. Planner  — generates diverse scenario specs per (category × priority)
  2. Generator — turns each spec into a realistic email
  3. Validator — independently classifies; keeps only agreement examples

Can ingest real Gmail exports from email-data-raw/ as seed examples that
are included in the output and used as few-shots for more realistic generation.

Usage:
  python scripts/generate_synthetic_emails.py --num-per-cell 5 --dry-run
  python scripts/generate_synthetic_emails.py --num-per-cell 8
  python scripts/generate_synthetic_emails.py --seed-from-gmail           # include real emails
  python scripts/generate_synthetic_emails.py --seed-from-gmail --gmail-only  # real emails only
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

from email_config import (
    API_BASE_URL,
    API_KEY_ENV,
    CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    GENERATOR_SYSTEM,
    GENERATOR_USER,
    MODEL_PRIORITY,
    PLANNER_SYSTEM,
    PLANNER_USER,
    PRIORITIES,
    VALIDATOR_SYSTEM,
    VALIDATOR_USER,
)

# ── Globals ────────────────────────────────────────────────────────

MAX_CONCURRENT = 2  # semaphore limit for API calls (Groq free tier: ~30 RPM)
MAX_RETRIES = 3
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
GMAIL_RAW_DIR = Path(__file__).resolve().parent.parent / "email-data-raw"

# Folder name → priority mapping
GMAIL_FOLDER_PRIORITY = {
    "important_emails": "P1",
    "job_emails": "P2",
    "receipts_emails": "P3",
    "noise_emails": "P4",
}


# ── Gmail Ingestion ───────────────────────────────────────────────

def format_gmail_to_text(email: dict) -> str:
    """Format a raw Gmail JSON object into plain email text."""
    parts = []
    if email.get("subject"):
        parts.append(f"Subject: {email['subject']}")
    if email.get("from"):
        parts.append(f"From: {email['from']}")
    if email.get("date"):
        parts.append(f"Date: {email['date']}")
    parts.append("")  # blank line before body

    body = email.get("body", "").strip()
    if not body:
        body = email.get("snippet", "").strip()
    parts.append(body)

    return "\n".join(parts)


def load_gmail_seeds() -> list[dict]:
    """Load real Gmail exports from email-data-raw/ and format them."""
    if not GMAIL_RAW_DIR.exists():
        print(f"Warning: {GMAIL_RAW_DIR} not found, skipping Gmail seeds")
        return []

    results = []
    for folder_name, priority in GMAIL_FOLDER_PRIORITY.items():
        folder = GMAIL_RAW_DIR / folder_name
        if not folder.exists():
            continue
        for json_file in folder.glob("*.json"):
            with open(json_file) as f:
                emails = json.load(f)
            for email in emails:
                email_text = format_gmail_to_text(email)
                if len(email_text.strip()) < 20:
                    continue  # skip empty/trivial
                results.append({
                    "email_text": email_text,
                    "priority": priority,
                    "category": "gmail_real",
                    "rationale": f"Real Gmail sample from {folder_name}/",
                    "scenario_spec": {"source": "gmail", "id": email.get("id", "")},
                })

    print(f"Loaded {len(results)} real Gmail emails:")
    pri_counts = {}
    for r in results:
        pri_counts[r["priority"]] = pri_counts.get(r["priority"], 0) + 1
    for p in sorted(pri_counts):
        print(f"  {p}: {pri_counts[p]}")

    return results


def get_fewshot_examples(gmail_seeds: list[dict], priority: str, n: int = 2) -> str:
    """Pick n real email examples for a priority to use as few-shots."""
    matching = [s for s in gmail_seeds if s["priority"] == priority]
    if not matching:
        return ""
    samples = random.sample(matching, min(n, len(matching)))
    lines = ["\n\nHere are real email examples at this priority level for reference:"]
    for i, s in enumerate(samples, 1):
        # Truncate to avoid huge prompts and token limits
        text = s["email_text"][:300]
        lines.append(f"\n--- Example {i} ---\n{text}")
    return "\n".join(lines)


def _resolve_api_key() -> str:
    """Get API key from env var."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key
    print(f"Error: {API_KEY_ENV} environment variable not set.")
    sys.exit(1)


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=API_BASE_URL, api_key=_resolve_api_key())


async def call_llm(
    client: AsyncOpenAI,
    system: str,
    user: str,
    sem: asyncio.Semaphore,
    model_idx: int = 0,
) -> str | None:
    """Call LLM with retry and rate-limit backoff."""
    model = MODEL_PRIORITY[model_idx]
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.9,
                    max_tokens=4096,
                )
            content = resp.choices[0].message.content
            if content:
                # Strip Qwen thinking traces
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                content = content.strip()
                if content:
                    return content
        except Exception as e:
            err = str(e)
            if "429" in err:
                # Rate limited — wait longer, then retry same model
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                await asyncio.sleep(wait)
            elif ("402" in err or "503" in err) and model_idx + 1 < len(MODEL_PRIORITY):
                return await call_llm(client, system, user, sem, model_idx + 1)
            else:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1}/{MAX_RETRIES} for {model} ({err[:80]}), waiting {wait}s")
                await asyncio.sleep(wait)
    return None


# ── Stage 1: Planner ──────────────────────────────────────────────

async def plan_scenarios(
    client: AsyncOpenAI,
    category: str,
    priority: str,
    n: int,
    sem: asyncio.Semaphore,
    fewshot_block: str = "",
) -> list[dict]:
    """Generate n scenario specs for a (category, priority) cell."""
    p = PRIORITIES[priority]
    user_msg = PLANNER_USER.format(
        n=n,
        category=category,
        category_desc=CATEGORY_DESCRIPTIONS[category],
        priority=p["label"],
        priority_desc=p["description"],
        priority_examples=", ".join(p["examples"]),
    )
    if fewshot_block:
        user_msg += fewshot_block
    raw = await call_llm(client, PLANNER_SYSTEM, user_msg, sem)
    if not raw:
        return []

    # Extract JSON array from response (handle markdown fences)
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.strip().rstrip("`")
    try:
        scenarios = json.loads(raw)
        if isinstance(scenarios, list):
            return scenarios[:n]
    except json.JSONDecodeError:
        # Try to find array in the response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())[:n]
            except json.JSONDecodeError:
                pass
    print(f"  Warning: could not parse planner output for {category}/{priority}")
    return []


# ── Stage 2: Generator ────────────────────────────────────────────

async def generate_email(
    client: AsyncOpenAI,
    category: str,
    priority: str,
    scenario: dict,
    sem: asyncio.Semaphore,
) -> str | None:
    """Generate a single email from a scenario spec."""
    user_msg = GENERATOR_USER.format(
        category=category,
        priority=priority,
        scenario_json=json.dumps(scenario, indent=2),
    )
    return await call_llm(client, GENERATOR_SYSTEM, user_msg, sem)


# ── Stage 3: Validator ────────────────────────────────────────────

def parse_priority_xml(text: str) -> tuple[str | None, str | None]:
    """Extract priority and rationale from validator XML output."""
    pri_match = re.search(r"<priority>\s*(P[1-4])\s*</priority>", text)
    rat_match = re.search(r"<rationale>(.*?)</rationale>", text, re.DOTALL)
    priority = pri_match.group(1) if pri_match else None
    rationale = rat_match.group(1).strip() if rat_match else None
    return priority, rationale


async def validate_email(
    client: AsyncOpenAI,
    email_text: str,
    sem: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Independently classify an email and return (priority, rationale)."""
    user_msg = VALIDATOR_USER.format(email_text=email_text)
    raw = await call_llm(client, VALIDATOR_SYSTEM, user_msg, sem)
    if not raw:
        return None, None
    return parse_priority_xml(raw)


# ── Pipeline ──────────────────────────────────────────────────────

async def process_cell(
    client: AsyncOpenAI,
    category: str,
    priority: str,
    num_scenarios: int,
    sem: asyncio.Semaphore,
    dry_run: bool = False,
    fewshot_block: str = "",
) -> list[dict]:
    """Process one (category, priority) cell through all 3 stages."""
    print(f"  Planning {category} × {priority} ({num_scenarios} scenarios)...")
    scenarios = await plan_scenarios(
        client, category, priority, num_scenarios, sem, fewshot_block
    )
    if not scenarios:
        return []
    print(f"    Got {len(scenarios)} scenarios")

    if dry_run:
        return [
            {
                "email_text": f"[DRY RUN] Scenario: {json.dumps(s)[:100]}",
                "priority": priority,
                "category": category,
                "rationale": "dry run",
                "scenario_spec": s,
            }
            for s in scenarios
        ]

    results = []

    # Generate emails concurrently
    gen_tasks = [
        generate_email(client, category, priority, s, sem) for s in scenarios
    ]
    emails = await asyncio.gather(*gen_tasks)

    # Validate concurrently
    valid_pairs = [(s, e) for s, e in zip(scenarios, emails) if e]
    if not valid_pairs:
        return []

    val_tasks = [validate_email(client, e, sem) for _, e in valid_pairs]
    validations = await asyncio.gather(*val_tasks)

    for (scenario, email_text), (val_priority, val_rationale) in zip(
        valid_pairs, validations
    ):
        if val_priority == priority:
            results.append(
                {
                    "email_text": email_text,
                    "priority": priority,
                    "category": category,
                    "rationale": val_rationale or "",
                    "scenario_spec": scenario,
                }
            )
        else:
            print(
                f"    Mismatch: expected {priority}, validator said {val_priority} — skipping"
            )

    return results


async def run_pipeline(
    num_per_cell: int,
    dry_run: bool = False,
    seed_from_gmail: bool = False,
    gmail_only: bool = False,
):
    """Run the full generation pipeline."""
    all_results = []
    gmail_seeds = []
    start = time.time()

    # ── Gmail ingestion ───────────────────────────────────────────
    if seed_from_gmail or gmail_only:
        gmail_seeds = load_gmail_seeds()
        all_results.extend(gmail_seeds)
        if gmail_only:
            # Write output and exit — no synthetic generation
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / "synthetic_emails.jsonl"
            with open(out_path, "w") as f:
                for item in all_results:
                    f.write(json.dumps(item) + "\n")
            print(f"\nGmail-only mode: {len(all_results)} examples written to {out_path}")
            return

    # ── Synthetic generation ──────────────────────────────────────
    client = get_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Pre-compute few-shot blocks per priority
    fewshot_blocks = {}
    if gmail_seeds:
        for pri in PRIORITIES:
            fewshot_blocks[pri] = get_fewshot_examples(gmail_seeds, pri)

    total_cells = len(CATEGORIES) * len(PRIORITIES)
    done = 0

    print(f"Starting pipeline: {total_cells} cells, {num_per_cell} scenarios each")
    print(f"Model priority: {MODEL_PRIORITY}")
    if gmail_seeds:
        print(f"Using {len(gmail_seeds)} real Gmail emails as seeds + few-shots")
    if dry_run:
        print("DRY RUN — no emails will be generated\n")

    cells = [
        (cat, pri)
        for pri in PRIORITIES
        for cat in CATEGORIES
    ]

    # Process one cell at a time to stay under Groq rate limits
    for i, (cat, pri) in enumerate(cells):
        results = await process_cell(
            client, cat, pri, num_per_cell, sem, dry_run,
            fewshot_block=fewshot_blocks.get(pri, ""),
        )
        all_results.extend(results)
        done += 1
        print(
            f"  Progress: {done}/{total_cells} cells, "
            f"{len(all_results)} examples so far"
        )
        # Pause between cells to stay under Groq rate limits
        if i + 1 < len(cells):
            await asyncio.sleep(15)

    elapsed = time.time() - start

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "synthetic_emails.jsonl"
    with open(out_path, "w") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    # Print summary
    gmail_count = len(gmail_seeds)
    synth_count = len(all_results) - gmail_count
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total examples: {len(all_results)} ({gmail_count} real + {synth_count} synthetic)")
    pri_counts = {}
    for item in all_results:
        pri_counts[item["priority"]] = pri_counts.get(item["priority"], 0) + 1
    for p in sorted(pri_counts):
        print(f"  {p}: {pri_counts[p]}")
    print(f"Output: {out_path}")

    # Validation rate (synthetic only)
    expected = len(cells) * num_per_cell
    rate = synth_count / expected * 100 if expected else 0
    print(f"Validation agreement rate: {rate:.1f}% ({synth_count}/{expected})")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic email dataset")
    parser.add_argument(
        "--num-per-cell",
        type=int,
        default=8,
        help="Number of scenarios per (category × priority) cell (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run planner only, skip generation and validation",
    )
    parser.add_argument(
        "--seed-from-gmail",
        action="store_true",
        help="Include real Gmail emails from email-data-raw/ and use as few-shots",
    )
    parser.add_argument(
        "--gmail-only",
        action="store_true",
        help="Only ingest Gmail data, skip synthetic generation",
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(
        args.num_per_cell, args.dry_run, args.seed_from_gmail, args.gmail_only,
    ))


if __name__ == "__main__":
    main()
