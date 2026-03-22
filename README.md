# EmailClassifierRLVR

Train an LLM to classify emails into P1–P4 priority tiers using RLVR.

---

## Quickstart

```bash
# 1. Build dataset from Gmail data already in email-data-raw/
cd scripts && uv run python generate_synthetic_emails.py --gmail-only

#1.5 (if updates)
prime env push --path ./environments/email_priority                                                                             

# 2. Install the environment
prime env install email-priority


# 3. Baseline survey (free — uses training infra, see note below) (just dont run this it doesnt work)
prime rl run configs/rl/email-priority-baseline.toml

# 4. Train
prime rl run configs/rl/email-priority.toml

# 5. Eval trained model (paid — requires Prime inference credits)
prime eval run email-priority -m openai/gpt-4.1-mini -n 50 -r 3
```

> **Free baseline via training:** `prime eval run` uses paid inference. Instead, use
> `configs/rl/email-priority-baseline.toml` which sets `batch_size=1` and includes all
> emails. Each training step processes one email, so the per-step reward logged to W&B
> is effectively per-email accuracy — giving you a full baseline sweep for free.
> Check W&B or `prime train tui` for the reward curve.

---

## Priority Tiers

- **P1** — Urgent, action required now (incidents, blocking, leadership requests)
- **P2** — Important, action within 1–2 days (stakeholder asks, project work)
- **P3** — Low priority, can wait a week (FYI, optional, informational)
- **P4** — Noise, no action needed (automated alerts, marketing, spam)

---

## Fetching More Gmail Data

Edit the variables at the top of `scripts/fetch_emails.py`, then run:

```bash
python scripts/fetch_emails.py
```

The generator maps these folders to priority tiers:

```
email-data-raw/important_emails/   → P1
email-data-raw/job_emails/         → P2
email-data-raw/receipts_emails/    → P3
email-data-raw/noise_emails/       → P4
```

---

## Generating a Larger Synthetic Dataset

```bash
# Synthetic only (requires PRIME_API_KEY or prime login)
uv run python generate_synthetic_emails.py --num-per-cell 8

# Synthetic + real Gmail as few-shot seeds
uv run python generate_synthetic_emails.py --num-per-cell 8 --seed-from-gmail
```

---

## Project Structure

```
configs/rl/          RL training configs
configs/eval/        Eval suite configs
configs/endpoints.toml  Model registry
data/                Generated dataset (not committed)
email-data-raw/      Raw Gmail exports by label
environments/email_priority/  Verifiers environment
scripts/             Data fetching + generation tools
```
