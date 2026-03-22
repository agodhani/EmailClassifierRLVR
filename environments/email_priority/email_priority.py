"""
Email Category Classification environment for verifiers.

Single-turn: model receives an email, outputs a category classification
with rationale in XML format.
"""

import json
from pathlib import Path

import sacrebleu as sb
import verifiers as vf
from datasets import Dataset
from rouge_score import rouge_scorer as rouge_lib

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_emails.jsonl"

SYSTEM_PROMPT = """\
You are an email classifier. Given an email, classify it into \
exactly one of these categories:

todo_tasks — Direct asks to do something — tasks, deliverables, approvals
meeting_coordination — Scheduling, rescheduling, agenda sharing, meeting follow-ups
newsletter_information — Internal or external newsletters, digests, roundups
important_alert — Notifications requiring immediate action (security codes, login alerts, password resets)
general — Personal emails landing in work inbox — family, friends, side projects
transactional — Receipts, order confirmations, subscription notices
jobs — Job related emails, updates, offers, application notifications

Think carefully about the email's purpose, sender, and required action. \
Then respond with EXACTLY this XML format:

<category>todo_tasks</category>
<rationale>Your reasoning here</rationale>

Respond with ONLY the XML tags above. No other text."""


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_environment(
    split: str = "train",
    num_examples: int = -1,
    data_path: str | None = None,
    train_ratio: float = 0.8,
) -> vf.Environment:
    """Load the email category classification environment.

    Args:
        split: "train" or "eval"
        num_examples: number of examples to use (-1 for all)
        data_path: override path to JSONL data file
        train_ratio: fraction of data for training (rest is eval)
    """
    path = Path(data_path) if data_path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python scripts/generate_synthetic_emails.py --gmail-only` first."
        )

    raw = _load_jsonl(path)

    # Split into train/eval
    split_idx = int(len(raw) * train_ratio)
    if split == "train":
        items = raw[:split_idx]
    else:
        items = raw[split_idx:]

    if num_examples > 0:
        items = items[:num_examples]

    # Build dataset
    dataset = Dataset.from_list(
        [
            {
                "question": item["email_text"],
                "answer": item["category"],
            }
            for item in items
        ]
    )

    # Build eval dataset (always from the eval split)
    eval_items = raw[split_idx:]
    eval_dataset = Dataset.from_list(
        [
            {
                "question": item["email_text"],
                "answer": item["category"],
            }
            for item in eval_items
        ]
    )

    # Reward: exact match on category
    parser = vf.XMLParser(fields=["category", "rationale"])

    def _get_text(completion) -> str:
        if isinstance(completion, list):
            return next(
                (m["content"] for m in reversed(completion) if m.get("role") == "assistant"),
                "",
            )
        return completion or ""

    async def exact_match(completion, answer, parser) -> float:
        parsed = parser.parse(_get_text(completion))
        if parsed.category and parsed.category.strip().lower() == answer.strip().lower():
            return 1.0
        return 0.0

    async def rationale_quality(completion, parser) -> float:
        parsed = parser.parse(_get_text(completion))
        rationale = parsed.rationale or ""
        word_count = len(rationale.split())
        if word_count >= 10:
            return 1.0
        elif word_count >= 5:
            return 0.5
        return 0.0

    async def format_quality(completion, parser) -> float:
        parsed = parser.parse(_get_text(completion))
        if parsed.category and parsed.rationale:
            return 2.0
        return 0.0
    
    async def rouge_rationale(completion, question, parser) -> float:
        rationale = parser.parse(_get_text(completion)).rationale or ""
        if not rationale:
            return 0.0
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(question, rationale)["rougeL"].fmeasure)

    async def bleu_rationale(completion, question, parser) -> float:
        rationale = parser.parse(_get_text(completion)).rationale or ""
        if not rationale:
            return 0.0
        score = sb.sentence_bleu(rationale, [question])
        return float(score.score) / 100.0  # normalize 0–100 → 0–1

    rubric = vf.Rubric(funcs=[exact_match, rationale_quality, format_quality], weights=[10.0, 1.0, 2.0], parser=parser)
    rubric.add_metric(rouge_rationale)
    rubric.add_metric(bleu_rationale)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        parser=parser,
    )
