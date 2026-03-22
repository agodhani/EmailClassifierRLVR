"""
Gmail label importer — pulls emails by label and saves them as JSON.

Usage:
    python scripts/fetch_emails.py

On first run a browser window opens for Google OAuth2 consent.
Subsequent runs reuse the saved token.
"""

# ---------------------------------------------------------------------------
# CONFIGURATION — change these variables to control what gets fetched / saved
# ---------------------------------------------------------------------------

GMAIL_LABEL      = "reciepts"                  # Gmail label to fetch
OUTPUT_FOLDER    = "email-data-raw/receipts_emails"  # Folder to write JSON into
MAX_EMAILS       = 500           # Max emails to fetch per run (set to None for all)
CREDENTIALS_FILE = "credentials.json"  # OAuth2 Desktop App credentials from Google Cloud
TOKEN_FILE       = "token.json"        # Saved token — auto-created on first auth

# ---------------------------------------------------------------------------

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _get_credentials():
    """Load or refresh OAuth2 credentials, prompting the user if needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "Missing dependencies. Run:\n"
            "  uv add google-api-python-client google-auth-httplib2 google-auth-oauthlib\n"
            "or:\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        sys.exit(1)

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(
                    f"credentials.json not found at '{CREDENTIALS_FILE}'.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return creds


def _decode_body(payload) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")

    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    if mime_type.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _decode_body(part)
            if text:
                return text

    return ""


def _header(headers: list, name: str) -> str:
    """Return the value of a header by name (case-insensitive)."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def fetch_emails(service, label: str, max_emails):
    """Fetch all messages with *label* and return a list of structured dicts."""
    messages = []
    page_token = None
    query = f"label:{label}"

    while True:
        kwargs = {"userId": "me", "q": query, "maxResults": 500}
        if page_token:
            kwargs["pageToken"] = page_token

        response = service.users().messages().list(**kwargs).execute()
        batch = response.get("messages", [])
        messages.extend(batch)

        if max_emails and len(messages) >= max_emails:
            messages = messages[:max_emails]
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(messages)} messages with label '{label}'. Fetching details...")

    emails = []
    for i, msg_ref in enumerate(messages, 1):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_ref["id"], format="full")
            .execute()
        )

        headers = msg.get("payload", {}).get("headers", [])
        body = _decode_body(msg.get("payload", {}))

        emails.append(
            {
                "id": msg["id"],
                "thread_id": msg.get("threadId", ""),
                "subject": _header(headers, "subject"),
                "from": _header(headers, "from"),
                "to": _header(headers, "to"),
                "date": _header(headers, "date"),
                "labels": msg.get("labelIds", []),
                "snippet": msg.get("snippet", ""),
                "body": body,
            }
        )

        if i % 50 == 0 or i == len(messages):
            print(f"  Processed {i}/{len(messages)}")

    return emails


def main():
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "Missing dependencies. Run:\n"
            "  uv add google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        sys.exit(1)

    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    emails = fetch_emails(service, GMAIL_LABEL, MAX_EMAILS)

    # Write output
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_label = GMAIL_LABEL.replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"{safe_label}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(emails)} emails → {output_path}")


if __name__ == "__main__":
    main()
