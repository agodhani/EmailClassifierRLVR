"""
Email Priority Classification environment for verifiers.

Single-turn: model receives an email, outputs a priority classification
(P1-P4) with rationale in XML format.
"""

import json
from pathlib import Path

import verifiers as vf
from datasets import Dataset

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic_emails.jsonl"

SYSTEM_PROMPT = """\
You are an email priority classifier. Given an email, classify it into \
exactly one priority tier:

P1 — Urgent / Action Required Now (blocking, time-critical, incidents, \
urgent requests from leadership or key clients)
P2 — Important / Action Required Soon (needs response in 1-2 business days, \
meaningful work requests, stakeholder asks)
P3 — Low Priority / Can Wait (FYI, optional meetings, informational, \
can defer a week)
P4 — Noise / No Action Needed (automated notifications, marketing, spam, \
mass CCs, no action required)

Think carefully about the urgency, who sent it, and what action is needed. \
Then respond with EXACTLY this XML format:

<priority>P1</priority>
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
    """Load the email priority classification environment.

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
            "Run `python scripts/generate_synthetic_emails.py` first."
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
                "answer": item["priority"],
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
                "answer": item["priority"],
            }
            for item in eval_items
        ]
    )

    # Reward: exact match on priority
    parser = vf.XMLParser(fields=["priority", "rationale"])

    async def exact_match(completion, answer, parser) -> float:
        parsed = parser.parse(completion)
        if parsed.priority and parsed.priority.strip().upper() == answer.strip().upper():
            return 1.0
        return 0.0

    async def rationale_quality(completion, parser) -> float:
        parsed = parser.parse(completion)
        rationale = parsed.rationale or ""
        # Simple heuristic: reward having a non-trivial rationale
        word_count = len(rationale.split())
        if word_count >= 10:
            return 1.0
        elif word_count >= 5:
            return 0.5
        return 0.0

    rubric = vf.Rubric(funcs=[exact_match], weights=[1.0], parser=parser)
    rubric.add_metric(rationale_quality)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        parser=parser,
    )
