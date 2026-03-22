#!/usr/bin/env python3
"""
Synthetic email generation pipeline.

Two modes:
  --gmail-only  : Run the validator LLM on every real Gmail email to assign
                  a category label, then write the labeled dataset.
  (default)     : Three-stage LLM pipeline:
                    1. Planner   — generates diverse scenario specs per category
                    2. Generator — turns each spec into a realistic email
                    3. Validator — independently classifies; keeps only agreement examples

Usage:
  python scripts/generate_synthetic_emails.py --num-per-cell 5 --dry-run
  python scripts/generate_synthetic_emails.py --num-per-cell 8
  python scripts/generate_synthetic_emails.py --seed-from-gmail           # include real emails
  python scripts/generate_synthetic_emails.py --gmail-only                # label real emails only
  python scripts/generate_synthetic_emails.py --gmail-only --preview      # preview labels, no write
"""

import argparse
import asyncio
import json
import os
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
    VALIDATOR_SYSTEM,
    VALIDATOR_USER,
)

# ── Globals ────────────────────────────────────────────────────────

MAX_CONCURRENT = 2  # semaphore limit for API calls (Groq free tier: ~30 RPM)
MAX_RETRIES = 3
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
GMAIL_RAW_DIR = Path(__file__).resolve().parent.parent / "email-data-raw"

# Folder name → expected category hints (informational only; validator assigns the final label)
GMAIL_FOLDER_CATEGORY = {
    "important_emails": ["important_alert", "todo_tasks"],
    "job_emails": ["jobs"],
    "receipts_emails": ["transactional"],
    "noise_emails": ["newsletter_information", "transactional", "general"],
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


def load_gmail_raw() -> list[dict]:
    """Load raw Gmail emails from email-data-raw/, returning (email_text, folder, email_id)."""
    if not GMAIL_RAW_DIR.exists():
        print(f"Warning: {GMAIL_RAW_DIR} not found, skipping Gmail data")
        return []

    results = []
    for folder_name in GMAIL_FOLDER_CATEGORY:
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
                    "folder": folder_name,
                    "email_id": email.get("id", ""),
                    "subject": email.get("subject", "(no subject)"),
                })

    print(f"Loaded {len(results)} raw Gmail emails from {len(GMAIL_FOLDER_CATEGORY)} folders")
    return results


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
    n: int,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Generate n scenario specs for a category."""
    user_msg = PLANNER_USER.format(
        n=n,
        category=category,
        category_desc=CATEGORY_DESCRIPTIONS[category],
    )
    raw = await call_llm(client, PLANNER_SYSTEM, user_msg, sem)
    if not raw:
        return []

    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.strip().rstrip("`")
    try:
        scenarios = json.loads(raw)
        if isinstance(scenarios, list):
            return scenarios[:n]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())[:n]
            except json.JSONDecodeError:
                pass
    print(f"  Warning: could not parse planner output for {category}")
    return []


# ── Stage 2: Generator ────────────────────────────────────────────

async def generate_email(
    client: AsyncOpenAI,
    category: str,
    scenario: dict,
    sem: asyncio.Semaphore,
) -> str | None:
    """Generate a single email from a scenario spec."""
    user_msg = GENERATOR_USER.format(
        category=category,
        category_desc=CATEGORY_DESCRIPTIONS[category],
        scenario_json=json.dumps(scenario, indent=2),
    )
    return await call_llm(client, GENERATOR_SYSTEM, user_msg, sem)


# ── Stage 3: Validator ────────────────────────────────────────────

def parse_category_xml(text: str) -> tuple[str | None, str | None]:
    """Extract category and rationale from validator XML output."""
    valid = set(CATEGORIES)
    cat_match = re.search(r"<category>\s*(\w+)\s*</category>", text)
    rat_match = re.search(r"<rationale>(.*?)</rationale>", text, re.DOTALL)
    category = cat_match.group(1).strip() if cat_match else None
    if category and category not in valid:
        category = None  # reject unknown categories
    rationale = rat_match.group(1).strip() if rat_match else None
    return category, rationale


async def validate_email(
    client: AsyncOpenAI,
    email_text: str,
    sem: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Classify an email into a category, returning (category, rationale)."""
    user_msg = VALIDATOR_USER.format(email_text=email_text)
    raw = await call_llm(client, VALIDATOR_SYSTEM, user_msg, sem)
    if not raw:
        return None, None
    return parse_category_xml(raw)


# ── Gmail labeling mode ────────────────────────────────────────────

async def label_gmail_emails(preview: bool = False) -> list[dict]:
    """Run validator LLM on every real Gmail email to assign category labels."""
    raw_emails = load_gmail_raw()
    if not raw_emails:
        return []

    client = get_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"\nClassifying {len(raw_emails)} emails into categories...")
    if preview:
        print("PREVIEW MODE — no file will be written\n")

    # Validate all emails concurrently (within semaphore limit)
    tasks = [validate_email(client, e["email_text"], sem) for e in raw_emails]
    validations = await asyncio.gather(*tasks)

    results = []
    category_counts: dict[str, int] = {}
    skipped = 0

    for raw, (category, rationale) in zip(raw_emails, validations):
        if not category:
            print(f"  SKIP (no valid category): {raw['subject'][:60]}")
            skipped += 1
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
        if preview:
            hint = GMAIL_FOLDER_CATEGORY.get(raw["folder"], [])
            hint_str = f" (folder hint: {hint})" if hint else ""
            print(f"  [{category}]{hint_str}  {raw['subject'][:70]}")
        results.append({
            "email_text": raw["email_text"],
            "category": category,
            "rationale": rationale or "",
            "scenario_spec": {"source": "gmail", "id": raw["email_id"]},
        })

    print(f"\nLabeled {len(results)} emails ({skipped} skipped — no valid category)")
    print("Category distribution:")
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        print(f"  {cat}: {count}")

    return results


# ── Synthetic generation pipeline ─────────────────────────────────

async def process_category(
    client: AsyncOpenAI,
    category: str,
    num_scenarios: int,
    sem: asyncio.Semaphore,
    dry_run: bool = False,
) -> list[dict]:
    """Process one category through all 3 stages."""
    print(f"  Planning {category} ({num_scenarios} scenarios)...")
    scenarios = await plan_scenarios(client, category, num_scenarios, sem)
    if not scenarios:
        return []
    print(f"    Got {len(scenarios)} scenarios")

    if dry_run:
        return [
            {
                "email_text": f"[DRY RUN] Scenario: {json.dumps(s)[:100]}",
                "category": category,
                "rationale": "dry run",
                "scenario_spec": s,
            }
            for s in scenarios
        ]

    results = []

    gen_tasks = [generate_email(client, category, s, sem) for s in scenarios]
    emails = await asyncio.gather(*gen_tasks)

    valid_pairs = [(s, e) for s, e in zip(scenarios, emails) if e]
    if not valid_pairs:
        return []

    val_tasks = [validate_email(client, e, sem) for _, e in valid_pairs]
    validations = await asyncio.gather(*val_tasks)

    for (scenario, email_text), (val_category, val_rationale) in zip(valid_pairs, validations):
        if val_category == category:
            results.append({
                "email_text": email_text,
                "category": category,
                "rationale": val_rationale or "",
                "scenario_spec": scenario,
            })
        else:
            print(f"    Mismatch: expected {category}, validator said {val_category} — skipping")

    return results


async def run_pipeline(
    num_per_cell: int,
    dry_run: bool = False,
    seed_from_gmail: bool = False,
    gmail_only: bool = False,
    preview: bool = False,
):
    """Run the full generation pipeline."""
    all_results = []
    start = time.time()

    # ── Gmail labeling mode ───────────────────────────────────────
    if gmail_only:
        labeled = await label_gmail_emails(preview=preview)
        if preview:
            return
        all_results.extend(labeled)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "synthetic_emails.jsonl"
        with open(out_path, "w") as f:
            for item in all_results:
                f.write(json.dumps(item) + "\n")
        print(f"\nGmail-only mode: {len(all_results)} examples written to {out_path}")
        return

    # ── Seed real emails if requested ────────────────────────────
    if seed_from_gmail:
        labeled = await label_gmail_emails()
        all_results.extend(labeled)
        gmail_count = len(labeled)
    else:
        gmail_count = 0

    # ── Synthetic generation ──────────────────────────────────────
    client = get_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"Starting pipeline: {len(CATEGORIES)} categories, {num_per_cell} scenarios each")
    print(f"Model: {MODEL_PRIORITY}")
    if dry_run:
        print("DRY RUN — no emails will be generated\n")

    for i, category in enumerate(CATEGORIES):
        results = await process_category(client, category, num_per_cell, sem, dry_run)
        all_results.extend(results)
        print(f"  Progress: {i+1}/{len(CATEGORIES)} categories, {len(all_results)} examples so far")
        if i + 1 < len(CATEGORIES):
            await asyncio.sleep(15)

    elapsed = time.time() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "synthetic_emails.jsonl"
    with open(out_path, "w") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    synth_count = len(all_results) - gmail_count
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total examples: {len(all_results)} ({gmail_count} real + {synth_count} synthetic)")
    cat_counts: dict[str, int] = {}
    for item in all_results:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts.get(cat, 0)}")
    print(f"Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic email dataset")
    parser.add_argument(
        "--num-per-cell",
        type=int,
        default=8,
        help="Number of scenarios per category (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run planner only, skip generation and validation",
    )
    parser.add_argument(
        "--seed-from-gmail",
        action="store_true",
        help="Include real Gmail emails (validator-labeled) and use them as seeds",
    )
    parser.add_argument(
        "--gmail-only",
        action="store_true",
        help="Only label Gmail data with the validator, skip synthetic generation",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="With --gmail-only: print subject → category assignments without writing output",
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(
        args.num_per_cell,
        args.dry_run,
        args.seed_from_gmail,
        args.gmail_only,
        args.preview,
    ))


if __name__ == "__main__":
    main()
