#!/usr/bin/env python3
"""
LiveKit Voice Bot Call Processing Pipeline

Ported from the ElevenLabs version of this pipeline. The core downstream
logic (BigQuery institution lookup, call_notes upserts, email alerts) is
unchanged — only the data source changed, since we now own the call data
directly instead of polling a third-party API for it.

Flow per call (sourced from our own `live_kit_bot_calls` Firestore collection,
written directly by va_bot.py at the end of every call):

  1. Query live_kit_bot_calls where pipeline_processed == False
  2. Mirror into calls_va_answered (source="livekit"), Category = null always
  3. If AMD already flagged this as a machine (voicemail/IVR/unavailable),
     auto-classify as "Non-Engagement" without spending a GPT call — no
     conversation occurred, so there's nothing for GPT to usefully read.
  4. Otherwise, skip deeper analysis if call duration < 40s
  5. Otherwise: AI extraction (GPT-4.1-mini) from the full transcript
  6. BigQuery institution lookup
  7. Skip call_notes upserts if no institution found
  8. Upsert call_notes_latest  (key: phone)
  9. Upsert call_notes_2       (key: doc_id)
 10. Send email alert if category is Complaint / Negotiation / PTP / Paid /
     Callback Request / Request to Speak to Agent
 11. Mark the source doc pipeline_processed=True

Idempotent — safe to run as often as you like. Each call is only ever
processed once (tracked via pipeline_processed on the source doc), unlike
the original ElevenLabs version's rolling time-window re-scan, which could
re-send the same email alert twice if runs overlapped.

Run manually or via cron:
    */15 * * * * /path/to/venv/bin/python /path/to/livekit_pipeline.py >> /var/log/livekit_pipeline.log 2>&1
"""

import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.cloud import bigquery, firestore
from google.oauth2 import service_account
from openai import OpenAI

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Environment ──────────────────────────────────────────────────────────
OPENAI_API_KEY     = os.environ["OPENAI_API_KEY"]
GOOGLE_SA_FILE      = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "service_account.json")
GMAIL_SENDER        = os.environ["GMAIL_SENDER"]
GMAIL_APP_PASSWORD  = os.environ["GMAIL_APP_PASSWORD"]
ALERT_RECIPIENT     = os.getenv("ALERT_RECIPIENT", "payments@fsldigital.com")

BQ_PROJECT        = "fssspark"
FS_PROJECT        = "fssspark"
SOURCE_COLLECTION = "live_kit_bot_calls"
RAW_LOG_COLLECTION = "calls_va_answered"
MIN_DURATION_SECS = 40
LAGOS_TZ = ZoneInfo("Africa/Lagos")

# AMD categories that mean "a machine answered, not a person" — see
# va_bot.py's AMDCategory. No conversation happens in these cases (the bot
# hangs up immediately), so there's nothing for GPT to usefully extract.
AMD_MACHINE_CATEGORIES = {"machine-vm", "machine-unavailable", "machine-ivr"}

ALERT_CATEGORIES = {
    "Complaint",
    "Negotiation",
    "Request to Speak to Agent",
    "PTP",
    "Callback Request",
    "Paid",
}

RECIPIENT_MAP = {
    0: "gbolade@fsldigital.com",
    1: "dami.adeyanju@fsldigital.com",
    2: "jedidah@fsldigital.com",
    3: "kesta@fsldigital.com",
    4: "emmanuella@fsldigital.com",
    5: "itoro@fsldigital.com",
}


# ── Prompt / schema ──────────────────────────────────────────────────────
# Same classification categories and extraction fields as the ElevenLabs
# version — this is about the conversation's content, not which platform
# ran the call. Only the "you will be given" framing changed, since we send
# the full transcript directly rather than a first-message + AI summary.
SYSTEM_PROMPT = """\
You are a call analysis engine for a loan recovery company.

You will be given the full transcript of a completed voice bot call (every
customer and agent turn, in order), the call duration, and the answering-
machine-detection (AMD) result. Your job is to:
1. Extract structured information from the transcript
2. Classify the conversation into exactly one category

CLASSIFICATION CATEGORIES:
- Paid — Customer confirms they have already made a payment or settlement
- PTP — Customer makes a clear, specific promise to pay on a stated date or within a stated timeframe. A vague "I'll pay soon" is NOT a PTP.
- Negotiation — Customer is discussing repayment terms, requesting a reduced amount, an extension, or proposing an alternative arrangement
- Complaint — Customer is expressing dissatisfaction, disputing the loan, the charges, or the recovery process
- System Error — Call dropped unexpectedly, transcript is empty or garbled, duration is abnormally short with no meaningful exchange, or termination reason indicates a technical failure
- Non-Engagement — Customer was reached but refused to engage, gave no meaningful response, or repeatedly deflected without committing to anything
- Unresolved — Conversation ended without a clear outcome — no payment, no PTP, no complaint, no refusal. Customer may have been unreachable or call ended ambiguously.

CLASSIFICATION RULES:
- If transcript is empty, missing, or under 2 meaningful exchanges → System Error
- If customer was reached but said nothing useful → Non-Engagement
- A specific date or amount mentioned with intent to pay → PTP, not Negotiation
- Customer saying "I already paid" → Paid, not Complaint, even if they sound frustrated
- If two categories seem to apply, pick the dominant intent of the conversation

---

EXTRACTION FIELDS:

Amount_Promised: Exact amount customer committed to paying (blank if none)
Payment_Method: Only if customer claims prior payment not yet reflected. One of: Branch payment, Zenith bank, Credit officer, Paystack, Union Bank, Leader, Providus Bank, Sales Rep, Unknown
Promise_to_Pay: Yes/No/Unknown
Right_Party_Contact: Yes/No/Unknown
Willingness: Score the customer's willingness to repay based on their behaviour in the call. Choose exactly one:
  - "0 Unwilling (Not Speaking)" — Customer picked up but said nothing at all, completely silent
  - "1 Unwilling" — Customer spoke but explicitly refused to pay or showed no willingness
  - "2 Medium" — Customer was reluctant but did not outright refuse, OR agreed without a confirmed PTP (a confirmed PTP requires both a ptp_date AND an Amount_Promised)
  - "3 Willing" — Customer willingly agreed to pay, most often with a confirmed PTP (date + amount)
  - "Unknown" — Cannot be determined (e.g. call dropped, debt denied, transcript unclear)
agent: Always set to "Voice Bot"
alternative_phone: Alternate number provided by customer (blank if none)
callback_requested: Yes/No/Unknown
contact_method: Always set to "Voice Call"
denied_debt: Yes/No/Unknown
employment_status: Business Owner/Salary Earner/Both/Unknown
federal_deduction: Yes/No/Unknown
name_of_officer: Name of officer customer claims to have paid (blank if none)
offset_loan_with_savings: Yes/No/Unknown
other_information: Concise summary of the conversation written from the agent's perspective. Do not attribute agent observations to the customer.
ptp_date: Specific payment date in YYYY-MM-DD format. Convert relative dates (e.g. "tomorrow", "next Friday") to exact dates based on call date. Blank if none.
reason_for_default: Brief summary of why customer defaulted (blank if not provided)
language: Language customer spoke to the bot. One of: Yoruba, English, Igbo, Hausa, Sheng, Swahili, Pidgin
discount_accepted: true/false/null
Category: The classification category from the list above
Call_Summary: A concise 1-2 sentence summary of what happened on the call, written for a human reviewer skimming a dashboard (not the customer). Cover what was discussed and the outcome.

EXTRACTION RULES:
- other_information must reflect what the CUSTOMER said or did, not what the bot observed
- "Not reachable", "no response", "switched off" = agent observation, NOT a customer complaint
- If the customer was unreachable, leave complaint-related fields blank and classify as Unresolved or System Error
- Never infer a ptp_date that was not stated or clearly implied

---

STRICT OUTPUT RULES:
1. Return a single valid JSON object — no markdown fences, no extra text
2. discount_accepted must be true, false, or null — never a string
3. All dates must be YYYY-MM-DD strings
4. If a field cannot be determined, use "" for text fields and null for booleans
5. Category is required — never leave it blank
6. Call_Summary is required — never leave it blank\
"""


# ── Google clients ───────────────────────────────────────────────────────
def _creds():
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/datastore",
    ]
    return service_account.Credentials.from_service_account_file(GOOGLE_SA_FILE, scopes=scopes)


def _bq():
    return bigquery.Client(project=BQ_PROJECT, credentials=_creds())


def _fs():
    return firestore.Client(project=FS_PROJECT, credentials=_creds())


# ── OpenAI ────────────────────────────────────────────────────────────────
_oai_client: Optional[OpenAI] = None


def _oai() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)
    return _oai_client


def extract_call_data(transcript: str, amd_category: str, duration_seconds: float) -> dict:
    user_content = (
        f"Transcript:\n{transcript}\n\n"
        f"AMD Classification (answering-machine-detection result): {amd_category or 'n/a'}\n"
        f"Call Duration: {duration_seconds} seconds"
    )
    resp = _oai().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def auto_classify_amd(amd_category: str) -> dict:
    """Builds the same shape extract_call_data() would return, for calls AMD
    already told us reached a machine — skips the GPT call entirely since
    there's no conversation to read (the bot hangs up immediately on these,
    see va_bot.py's AMD handling)."""
    return {
        "Amount_Promised": "",
        "Payment_Method": "",
        "Promise_to_Pay": "Unknown",
        "Right_Party_Contact": "Unknown",
        "Willingness": "Unknown",
        "agent": "Voice Bot",
        "alternative_phone": "",
        "callback_requested": "Unknown",
        "contact_method": "Voice Call",
        "denied_debt": "Unknown",
        "employment_status": "Unknown",
        "federal_deduction": "Unknown",
        "name_of_officer": "",
        "offset_loan_with_savings": "Unknown",
        "other_information": f"No human engagement — call reached a machine (AMD: {amd_category}).",
        "ptp_date": "",
        "reason_for_default": "",
        "language": "",
        "discount_accepted": None,
        "Category": "Non-Engagement",
        "Call_Summary": f"No human engagement — call reached a machine (AMD: {amd_category}).",
    }


# ── BigQuery ──────────────────────────────────────────────────────────────
def query_institution(phone_norm: str, bq_client: bigquery.Client) -> Optional[str]:
    sql = """
        SELECT ANY_VALUE(institution) AS institution
        FROM `fssspark.original_cohorts.all_leads`
        WHERE phone = @phone
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("phone", "STRING", phone_norm)]
    )
    try:
        for row in bq_client.query(sql, job_config=cfg).result():
            return row.institution
    except Exception as exc:
        log.warning("BigQuery error for %s: %s", phone_norm, exc)
    return None


# ── Gmail ─────────────────────────────────────────────────────────────────
def route_recipient(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    last_two = int(digits[-2:]) if len(digits) >= 2 else 0
    return RECIPIENT_MAP[last_two % 6]


def send_alert(
    phone: str,
    category: str,
    institution: str,
    call_dt: str,
    other_info: str,
    recipient: str,
    summary: str = "",
):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Voice Bot Follow-up Required - {category} | {phone}"
    msg["From"] = f"voicebot <{GMAIL_SENDER}>"
    msg["To"] = recipient
    msg["Cc"] = ALERT_RECIPIENT

    html = (
        "<h2>Voice Bot Follow-up Required</h2>"
        f"<p><strong>Phone:</strong> {phone}</p>"
        f"<p><strong>Category:</strong> {category}</p>"
        f"<p><strong>Institution:</strong> {institution}</p>"
        f"<p><strong>Date of Call:</strong> {call_dt}</p>"
        + (f"<p><strong>Call Summary:</strong></p><p>{summary}</p>" if summary else "")
        + "<p><strong>Other Information:</strong></p>"
        f"<p>{other_info}</p>"
    )
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        smtp.sendmail(GMAIL_SENDER, [recipient, ALERT_RECIPIENT], msg.as_string())


# ── Helpers ───────────────────────────────────────────────────────────────
def normalise_phone(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("+234"):
        raw = "0" + raw[4:]
    elif raw.startswith("234") and len(raw) >= 13:
        raw = "0" + raw[3:]
    return ("0" + raw)[-11:]


def to_iso(dt: datetime) -> str:
    """Format datetime as 2026-05-20T17:19:00.000Z, matching the original
    pipeline's timestamp format for call_notes consumers."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def fetch_unprocessed_calls(db: firestore.Client) -> list[dict]:
    """Every call va_bot.py has written that this pipeline hasn't handled yet."""
    docs = db.collection(SOURCE_COLLECTION).where("pipeline_processed", "==", False).stream()
    calls = []
    for d in docs:
        data = d.to_dict() or {}
        data["_doc_id"] = d.id
        data["_doc_ref"] = d.reference
        calls.append(data)
    log.info("Fetched %d unprocessed calls", len(calls))
    return calls


def fetch_all_calls_sorted(db: firestore.Client) -> list[dict]:
    """One-time historical backfill: every call regardless of pipeline_processed,
    oldest first. Ascending order matters here — call_notes_latest is keyed by
    phone and meant to hold each customer's MOST RECENT note, so calls must be
    applied in chronological order or an older call processed later could
    overwrite a newer one's note."""
    docs = (
        db.collection(SOURCE_COLLECTION)
        .order_by("started_at", direction=firestore.Query.ASCENDING)
        .stream()
    )
    calls = []
    for d in docs:
        data = d.to_dict() or {}
        data["_doc_id"] = d.id
        data["_doc_ref"] = d.reference
        calls.append(data)
    log.info("Fetched %d total calls for historical backfill (oldest first)", len(calls))
    return calls


# ── Main pipeline ─────────────────────────────────────────────────────────
def run(calls: list[dict] | None = None):
    log.info("=== Pipeline started ===")
    bq = _bq()
    db = _fs()

    if calls is None:
        calls = fetch_unprocessed_calls(db)

    for call in calls:
        room_name = call.get("room_name") or call["_doc_id"]
        log.info("▶ %s", room_name)

        phone_raw = call.get("phone_number")
        started_at = call.get("started_at")
        if not isinstance(started_at, datetime):
            started_at = datetime.now(timezone.utc)
        duration = call.get("duration_seconds") or 0
        transcript = call.get("transcript") or ""
        amd_category = call.get("amd_category") or ""
        call_dt = to_iso(started_at)

        phone_norm = normalise_phone(phone_raw) if phone_raw else None

        # ── 1. AMD already told us this is a machine — skip GPT entirely ────
        skip_after_raw_log = False  # duration-gate case: log + mark processed, nothing else
        retry_later = False  # GPT failure: log, but leave unprocessed for retry
        if amd_category in AMD_MACHINE_CATEGORIES:
            extracted = auto_classify_amd(amd_category)
            log.info("  AMD machine call — auto-classified as Non-Engagement, skipping GPT")
        else:
            # ── 2. Duration gate — skip deeper analysis if too short ────────
            if duration < MIN_DURATION_SECS:
                log.info("  skip — %ss < %ss minimum", duration, MIN_DURATION_SECS)
                extracted = {
                    "Call_Summary": f"Call too short ({int(duration)}s) for meaningful engagement — no analysis performed.",
                    "Category": "",
                }
                skip_after_raw_log = True
            else:
                # ── 3. AI extraction ─────────────────────────────────────────
                try:
                    extracted = extract_call_data(transcript, amd_category, duration)
                    log.info("  ✓ AI — Category: %s", extracted.get("Category"))
                except Exception as exc:
                    log.error("  ✗ AI extraction: %s — leaving unprocessed for retry", exc)
                    extracted = {"Call_Summary": ""}
                    retry_later = True

        transcript_summary = extracted.get("Call_Summary", "")

        # ── 4. Mirror into calls_va_answered (raw log, all voice-bot calls) ──
        try:
            db.collection(RAW_LOG_COLLECTION).document(room_name).set(
                {
                    "phone_number": phone_norm or "",
                    "duration": str(duration),
                    "category": None,
                    "status": "",
                    "datetime": call_dt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_type": "inbound",
                    "agent_id": "fola",
                    "agent_name": "fola",
                    "voicemail_detection": amd_category,
                    "transcript_summary": transcript_summary,
                    "conversation_id": room_name,
                    "source": "livekit",
                    "recording_url": call.get("recording_url") or "",
                },
                merge=True,
            )
            log.info("  ✓ calls_va_answered")
        except Exception as exc:
            log.error("  ✗ calls_va_answered: %s", exc)

        if retry_later:
            continue  # don't mark processed; retry next run

        if skip_after_raw_log:
            call["_doc_ref"].update({"pipeline_processed": True})
            continue

        category = extracted.get("Category", "")

        if not phone_norm:
            log.info("  skip — no phone number on this call")
            call["_doc_ref"].update({"pipeline_processed": True})
            continue

        # ── 5. BigQuery — institution lookup ─────────────────────────────────
        institution = query_institution(phone_norm, bq)
        if not institution:
            log.info("  skip — no institution found for %s", phone_norm)
            call["_doc_ref"].update({"pipeline_processed": True})
            continue

        # ── 6. Build shared document ──────────────────────────────────────────
        doc_id = f"{phone_norm}_{int(started_at.timestamp() * 1000)}"
        now_dt = to_iso(datetime.now(LAGOS_TZ))

        call_notes = {
            "Amount_Promised": extracted.get("Amount_Promised", ""),
            "Payment_Method": extracted.get("Payment_Method", ""),
            "Promise_to_Pay": extracted.get("Promise_to_Pay", ""),
            "Right_Party_Contact": extracted.get("Right_Party_Contact", ""),
            "Willingness": extracted.get("Willingness", ""),
            "agent": extracted.get("agent", "Voice Bot"),
            "alternative_phone": extracted.get("alternative_phone", ""),
            "callback_requested": extracted.get("callback_requested", ""),
            "contact_method": extracted.get("contact_method", "Voice Call"),
            "denied_debt": extracted.get("denied_debt", ""),
            "employment_status": extracted.get("employment_status", ""),
            "federal_deduction": extracted.get("federal_deduction", ""),
            "institution": institution,
            "name_of_officer": extracted.get("name_of_officer", ""),
            "offset_loan_with_savings": extracted.get("offset_loan_with_savings", ""),
            "other_information": extracted.get("other_information", ""),
            "phone": phone_norm,
            "ptp_date": extracted.get("ptp_date", ""),
            "reason_for_default": extracted.get("reason_for_default", ""),
            "timestamp": call_dt,
            "updated_at": now_dt,
            "discount_accepted": extracted.get("discount_accepted"),
            "language": extracted.get("language", ""),
            "recording_url": call.get("recording_url") or "",
            "source": "livekit",
        }

        # ── 7. Firestore — call_notes_latest (doc ID = phone) ───────────────
        try:
            db.collection("call_notes_latest").document(phone_norm).set(call_notes, merge=True)
            log.info("  ✓ call_notes_latest")
        except Exception as exc:
            log.error("  ✗ call_notes_latest: %s", exc)

        # ── 8. Firestore — call_notes_2 (doc ID = phone_timestamp) ───────────
        try:
            db.collection("call_notes_2").document(doc_id).set(call_notes, merge=True)
            log.info("  ✓ call_notes_2")
        except Exception as exc:
            log.error("  ✗ call_notes_2: %s", exc)

        # ── 9. Email alert + agent_notifications (alert categories only) ─────
        # Idempotency guard: an agent_notifications doc for this room means an
        # email already went out for it on a prior run (e.g. before the
        # historical backfill was interrupted and restarted) — never re-send.
        already_notified = (
            category in ALERT_CATEGORIES
            and db.collection("agent_notifications").document(room_name).get().exists
        )

        if category in ALERT_CATEGORIES and already_notified:
            log.info("  skip — email already sent for %s on a prior run", room_name)
        elif category in ALERT_CATEGORIES:
            try:
                send_alert(
                    phone=phone_norm,
                    category=category,
                    institution=institution,
                    call_dt=call_dt,
                    other_info=extracted.get("other_information", ""),
                    recipient=route_recipient(phone_norm),
                    summary=transcript_summary,
                )
                log.info("  ✓ email sent — %s", category)
            except Exception as exc:
                log.error("  ✗ email: %s", exc)

            try:
                db.collection("agent_notifications").document(room_name).set(
                    {
                        "type": "voice_call",
                        "recipient_email": route_recipient(phone_norm),
                        "phone": phone_norm,
                        "category": category,
                        "institution": institution,
                        "summary": transcript_summary,
                        "other_info": extracted.get("other_information", ""),
                        "timestamp": call_dt,
                        "read": False,
                        "received_at": firestore.SERVER_TIMESTAMP,
                        "source": "livekit",
                    }
                )
                log.info("  ✓ agent_notifications")
            except Exception as exc:
                log.error("  ✗ agent_notifications: %s", exc)

        # ── 10. Mark processed so future runs skip this call ─────────────────
        call["_doc_ref"].update({"pipeline_processed": True})

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run()
