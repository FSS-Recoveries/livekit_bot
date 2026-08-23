#!/usr/bin/env python3
import asyncio
import os
import re
import requests
import inspect
import time
import uuid
import wave
from datetime import datetime, timedelta, timezone
from urllib.parse import quote as _urlquote
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from livekit import rtc
from livekit import api as lk_api
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    get_job_context,
    tts as agents_tts,
    stt as agents_stt,
    llm as agents_llm,
    inference,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
)
from livekit.agents.inference import TurnDetector
from livekit.agents.metrics import ModelUsageCollector
from livekit.plugins import silero, noise_cancellation, elevenlabs
from livekit.agents.voice.amd import AMD, AMDCategory

import firebase_admin
from firebase_admin import credentials as fb_credentials
from firebase_admin import firestore as fb_firestore
from firebase_admin import storage as fb_storage

from livekit.plugins import deepgram, openai, silero, elevenlabs, noise_cancellation, azure
from livekit.agents import inference

# ── Quick compat shim: Mp3StreamDecoder → AudioStreamDecoder ─────────────
# Some livekit-agents versions removed Mp3StreamDecoder in favor of AudioStreamDecoder.
# If the direct ElevenLabs plugin (used for TTS, to preserve our custom
# voice — see build_elevenlabs_tts below) tries to use Mp3StreamDecoder,
# provide a lightweight alias.
try:
    from livekit.agents import utils as _lk_utils  # type: ignore
    _codecs = _lk_utils.codecs
    if not hasattr(_codecs, "Mp3StreamDecoder") and hasattr(_codecs, "AudioStreamDecoder"):
        import inspect as _inspect

        class _Mp3Shim(_codecs.AudioStreamDecoder):  # type: ignore[attr-defined]
            def __init__(self, *args, **kwargs):
                # Newer AudioStreamDecoder may accept a 'format' kwarg; prefer mp3 if available.
                params = set()
                try:
                    params = set(_inspect.signature(_codecs.AudioStreamDecoder).parameters.keys())  # type: ignore[attr-defined]
                except Exception:
                    pass
                if "format" in params:
                    kwargs.setdefault("format", "mp3")
                super().__init__(*args, **kwargs)

        _codecs.Mp3StreamDecoder = _Mp3Shim  # alias so older plugins keep working
except Exception:
    # Non-fatal; if this fails we still might succeed via PCM encoding below.
    pass

# ── Load env ────────────────────────────────────────────────────────────
# .env.local should contain LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET.
# STT and LLM run through LiveKit Inference (no separate keys needed there).
# TTS is the exception: it uses the direct Azure plugin as primary (needs
# AZURE_SPEECH_KEY/AZURE_SPEECH_REGION) and the direct ElevenLabs plugin as
# fallback, to keep our custom cloned ElevenLabs voice (LiveKit Inference
# only supports ElevenLabs' own default voices, not custom/cloned ones — see
# build_elevenlabs_tts below), which needs ELEVENLABS_API_KEY set here.
load_dotenv(".env.local")

# Map ELEVENLABS_API_KEY -> ELEVEN_API_KEY for plugin compatibility
if not os.getenv("ELEVEN_API_KEY") and os.getenv("ELEVENLABS_API_KEY"):
    os.environ["ELEVEN_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")

# ── Firebase (call recordings + call records) ────────────────────────────
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "service_account.json")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "fssspark.firebasestorage.app")
RECORDINGS_FOLDER = "live_kit_calls"
CALLS_COLLECTION = "live_kit_bot_calls"

_firebase_app = None
if os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
    try:
        _cred = fb_credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
        _firebase_app = firebase_admin.initialize_app(
            _cred, {"storageBucket": FIREBASE_STORAGE_BUCKET}
        )
    except ValueError:
        _firebase_app = firebase_admin.get_app()
    except Exception as e:
        print(f"Failed to initialize Firebase: {e}")
else:
    print(
        f"No Firebase service account found at {FIREBASE_SERVICE_ACCOUNT_PATH}; "
        "call recording/storage is disabled."
    )


def _load_gcp_credentials_json() -> str | None:
    if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
        return None
    with open(FIREBASE_SERVICE_ACCOUNT_PATH, "r", encoding="utf-8") as f:
        return f.read()


async def _start_call_recording(ctx: JobContext) -> "lk_api.EgressInfo | None":
    """Records the call (audio-only) straight to Firebase Storage via LiveKit Egress."""
    creds_json = _load_gcp_credentials_json()
    if not creds_json:
        return None
    req = lk_api.RoomCompositeEgressRequest(
        room_name=ctx.room.name,
        audio_only=True,
        file_outputs=[
            lk_api.EncodedFileOutput(
                file_type=lk_api.EncodedFileType.OGG,
                filepath=f"{RECORDINGS_FOLDER}/{ctx.room.name}.ogg",
                gcp=lk_api.GCPUpload(credentials=creds_json, bucket=FIREBASE_STORAGE_BUCKET),
            )
        ],
    )
    try:
        return await ctx.api.egress.start_room_composite_egress(req)
    except Exception as e:
        print(f"Failed to start call recording: {e}")
        return None


async def _finish_call_recording(ctx: JobContext, egress_info) -> str | None:
    """Stops egress, waits for the upload to finish, and returns a Firebase download URL."""
    if egress_info is None or _firebase_app is None:
        return None
    try:
        await ctx.api.egress.stop_egress(lk_api.StopEgressRequest(egress_id=egress_info.egress_id))
    except Exception as e:
        print(f"Failed to stop call recording: {e}")
        return None

    # Egress finalizes and uploads the file asynchronously after stop; poll briefly.
    for _ in range(15):
        await asyncio.sleep(2)
        try:
            resp = await ctx.api.egress.list_egress(
                lk_api.ListEgressRequest(egress_id=egress_info.egress_id)
            )
        except Exception as e:
            print(f"Failed to poll call recording status: {e}")
            return None
        if not resp.items:
            return None
        info = resp.items[0]
        if info.status == lk_api.EgressStatus.EGRESS_COMPLETE:
            break
        if info.status in (lk_api.EgressStatus.EGRESS_FAILED, lk_api.EgressStatus.EGRESS_ABORTED):
            print(f"Call recording failed: {info.error}")
            return None
    else:
        print("Timed out waiting for call recording to finish uploading.")
        return None

    filepath = f"{RECORDINGS_FOLDER}/{ctx.room.name}.ogg"
    try:
        bucket = fb_storage.bucket()
        blob = bucket.blob(filepath)
        token = str(uuid.uuid4())
        blob.metadata = {"firebaseStorageDownloadTokens": token}
        blob.patch()
        # Full percent-encoding, not just "/" -> "%2F": room names contain a
        # literal "+" from the E.164 phone number (e.g. call-_+234...), and
        # an un-encoded "+" in the URL causes Firebase Storage to 404 even
        # though the object exists — every real call's recording_url was
        # broken by this until it was caught.
        encoded_path = _urlquote(filepath, safe="")
        return (
            f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_STORAGE_BUCKET}"
            f"/o/{encoded_path}?alt=media&token={token}"
        )
    except Exception as e:
        print(f"Failed to finalize call recording URL: {e}")
        return None


# Rates below are LiveKit's own Inference rate card (Build/Ship plan),
# sourced directly from livekit.com/pricing/inference as of 2026-08-12 —
# NOT the underlying providers' own direct API prices, since stt/llm/tts
# for the entries below all route through LiveKit Inference
# (livekit.agents.inference) and are billed at LiveKit's rate, not
# Google's/OpenAI's/Deepgram's/Fish Audio's/Inworld's own. The one
# exception is _ELEVENLABS_FLASH_PER_CHARACTER, which IS the direct
# ElevenLabs rate — build_elevenlabs_tts() uses the direct plugin, not
# Inference, so it's billed straight to the ElevenLabs account instead.
# If you switch LiveKit Cloud plans (Build/Ship vs Scale), re-check these:
# Inworld and Deepgram/ElevenLabs STT rates differ by plan; Gemini/GPT-5-mini
# LLM rates and the Fish Audio TTS rate were confirmed identical on both as
# of the same date.
#
# The `provider`/`model` strings these usage entries actually contain once
# routed through the gateway haven't all been individually confirmed against
# a real call (see per-model notes below) — matching may still silently
# return None for an entry if a provider string differs from what's assumed.
# Azure isn't priced here at all: its plugin doesn't report
# model_provider/model_name metadata the way the others do, so it hasn't
# been confirmed against a real usage entry to know the exact keys it would
# use (even though it's currently unused as a fallback — see build_azure_tts).
#
# Gemini 3.1 Flash-Lite is matched on `model` alone (no `provider` check,
# unlike the OpenAI/ElevenLabs/Deepgram entries below) — the exact
# `provider` string the Google plugin reports for its usage metrics hasn't
# been confirmed against a real call. If estimated_cost_usd comes back None
# for Gemini usage entries, check a real usage dict's `provider` value and
# add that condition back for a tighter match.
_GEMINI_FLASH_LITE_INPUT_PER_TOKEN = 0.25 / 1_000_000
_GEMINI_FLASH_LITE_CACHED_INPUT_PER_TOKEN = 0.025 / 1_000_000
_GEMINI_FLASH_LITE_OUTPUT_PER_TOKEN = 1.50 / 1_000_000  # includes thinking tokens
_GPT5_MINI_INPUT_PER_TOKEN = 0.25 / 1_000_000
_GPT5_MINI_CACHED_INPUT_PER_TOKEN = 0.030 / 1_000_000
_GPT5_MINI_OUTPUT_PER_TOKEN = 2.00 / 1_000_000
# Direct ElevenLabs API rate (not Inference — see note above), from
# elevenlabs.io/pricing/api as of 2026-08-12: "$0.05 per 1K characters" for
# the Flash/Turbo tier, which eleven_flash_v2_5 falls under.
_ELEVENLABS_FLASH_PER_CHARACTER = 0.05 / 1_000
# ElevenLabs Scribe v2 Realtime (the primary STT) had no cost entry at all
# before this — it was silently falling through to the `return None` below
# on every call. Matched on `model` alone, same reasoning as Gemini above.
_ELEVENLABS_SCRIBE_V2_REALTIME_PER_SECOND = 0.0105 / 60
# Was 0.0077/min here previously — that was Deepgram's own direct nova-3
# rate, not LiveKit's Inference rate for it, which is lower.
_DEEPGRAM_NOVA3_PER_SECOND = 0.0048 / 60
# Fish Audio and Inworld are both billed per character of input text, not
# per minute of output audio — the per-minute figure on livekit.com/pricing
# is a simplified marketing conversion; the granular rate card at
# livekit.com/pricing/inference gives the actual metered unit as
# $/million characters, which is what these two constants use. Matched on
# `model` alone, same reasoning as Gemini above — the `provider` string
# these entries report hasn't been confirmed against a real usage entry.
_FISHAUDIO_S21PRO_PER_CHARACTER = 15.00 / 1_000_000
_INWORLD_TTS_15_MAX_PER_CHARACTER = 35.00 / 1_000_000
# AssemblyAI Universal-Streaming-Multilingual (third STT fallback): $0.0025/min,
# the cheapest STT on LiveKit's Inference rate card. Matched on `model`
# alone, same reasoning as Gemini above.
_ASSEMBLYAI_UNIVERSAL_STREAMING_MULTILINGUAL_PER_SECOND = 0.0025 / 60
# xAI Grok 4.1 Fast non-reasoning (third LLM fallback): $0.20/$0.50 per
# million input/output tokens. LiveKit's rate card lists no separate cached-
# input rate for this model (shown as N/A) — all input tokens are billed at
# the same rate, unlike the Gemini/GPT-5-mini entries above.
_GROK_4_1_FAST_INPUT_PER_TOKEN = 0.20 / 1_000_000
_GROK_4_1_FAST_OUTPUT_PER_TOKEN = 0.50 / 1_000_000


def _estimate_entry_cost(entry: dict) -> float | None:
    provider, model = entry.get("provider"), entry.get("model")
    if model == "gemini-3.1-flash-lite":
        # input_tokens already includes input_cached_tokens as a subset —
        # only the non-cached remainder is billed at the full input rate.
        cached = entry.get("input_cached_tokens", 0)
        uncached = max(entry.get("input_tokens", 0) - cached, 0)
        return (
            uncached * _GEMINI_FLASH_LITE_INPUT_PER_TOKEN
            + cached * _GEMINI_FLASH_LITE_CACHED_INPUT_PER_TOKEN
            + entry.get("output_tokens", 0) * _GEMINI_FLASH_LITE_OUTPUT_PER_TOKEN
        )
    if provider == "api.openai.com" and model == "gpt-5-mini":
        cached = entry.get("input_cached_tokens", 0)
        uncached = max(entry.get("input_tokens", 0) - cached, 0)
        return (
            uncached * _GPT5_MINI_INPUT_PER_TOKEN
            + cached * _GPT5_MINI_CACHED_INPUT_PER_TOKEN
            + entry.get("output_tokens", 0) * _GPT5_MINI_OUTPUT_PER_TOKEN
        )
    if provider == "ElevenLabs" and model == "eleven_flash_v2_5":
        return entry.get("characters_count", 0) * _ELEVENLABS_FLASH_PER_CHARACTER
    if model == "elevenlabs/scribe_v2_realtime":
        return entry.get("audio_duration", 0) * _ELEVENLABS_SCRIBE_V2_REALTIME_PER_SECOND
    if provider == "Deepgram" and model == "nova-3":
        return entry.get("audio_duration", 0) * _DEEPGRAM_NOVA3_PER_SECOND
    if model == "fishaudio/s2.1-pro":
        return entry.get("characters_count", 0) * _FISHAUDIO_S21PRO_PER_CHARACTER
    if model == "inworld/inworld-tts-1.5-max":
        return entry.get("characters_count", 0) * _INWORLD_TTS_15_MAX_PER_CHARACTER
    if model == "assemblyai/universal-streaming-multilingual":
        return entry.get("audio_duration", 0) * _ASSEMBLYAI_UNIVERSAL_STREAMING_MULTILINGUAL_PER_SECOND
    if model == "xai/grok-4-1-fast-non-reasoning":
        return (
            entry.get("input_tokens", 0) * _GROK_4_1_FAST_INPUT_PER_TOKEN
            + entry.get("output_tokens", 0) * _GROK_4_1_FAST_OUTPUT_PER_TOKEN
        )
    return None


def _add_estimated_costs(usage: list) -> tuple[list, float]:
    """Annotates each usage entry with an estimated_cost_usd and returns the
    enriched list alongside the total. Entries with no known pricing (e.g.
    Azure TTS, LiveKit's own interruption/turn-detector usage) are left
    with estimated_cost_usd=None rather than guessed."""
    total = 0.0
    enriched = []
    for entry in usage:
        entry = dict(entry)
        cost = _estimate_entry_cost(entry)
        if cost is not None:
            total += cost
        entry["estimated_cost_usd"] = round(cost, 6) if cost is not None else None
        enriched.append(entry)
    return enriched, round(total, 6)


def _save_call_record(
    room_name: str,
    transcript_lines: list,
    amd_category,
    recording_url,
    started_at: datetime,
    duration_seconds: float,
    usage: list,
    phone_number: str | None,
) -> None:
    if _firebase_app is None:
        return
    usage, total_estimated_cost_usd = _add_estimated_costs(usage)
    try:
        db = fb_firestore.client()
        db.collection(CALLS_COLLECTION).document(room_name).set(
            {
                "room_name": room_name,
                "phone_number": phone_number,
                "transcript": "\n".join(transcript_lines),
                "amd_category": amd_category,
                "recording_url": recording_url,
                "started_at": started_at,
                "duration_seconds": round(duration_seconds, 1),
                "usage": usage,
                "total_estimated_cost_usd": total_estimated_cost_usd,
                "ended_at": fb_firestore.SERVER_TIMESTAMP,
                # Picked up by the separate livekit_pipeline.py batch script
                # (AI extraction, institution lookup, call_notes upserts,
                # email alerts). False here, flipped to True once processed —
                # explicit so the pipeline can query for it directly (Firestore
                # can't cleanly query "field is missing").
                "pipeline_processed": False,
            }
        )
    except Exception as e:
        print(f"Failed to save call record to Firestore: {e}")


# ── Hearing-difficulty backstop (see _on_conversation_item_added) ───────
# Real call transcripts showed the model NOT reliably applying its own
# "3+ times → hearing-difficulty dead end" instruction — one real call ran
# well past a dozen consecutive "Hello?"-only exchanges before ever
# recognizing it. This is a deterministic, code-level backstop that doesn't
# depend on the model noticing the pattern: every fragment of the
# utterance (split on sentence punctuation, so "Hello? Hello. Good
# afternoon." counts as three) must be nothing but a bare greeting for it
# to count — any real content anywhere (a "yes", a number, a name) fails
# the match, so genuine short answers are never caught by this.
_GREETING_ONLY_PHRASES = {
    "hello", "hi", "hey", "yo",
    "good morning", "good afternoon", "good evening", "good day",
    "who is this", "who is speaking", "who is calling",
    "can you hear me", "are you there",
}


def _is_greeting_only_utterance(text: str) -> bool:
    fragments = [f.strip() for f in re.split(r"[.?!]+", text.lower()) if f.strip()]
    return bool(fragments) and all(f in _GREETING_ONLY_PHRASES for f in fragments)


# ── Function tools ──────────────────────────────────────────────────────
def _normalize_kenyan_phone(raw: str) -> str:
    """Strip non-digits, then convert 254XXXXXXXXX → 0XXXXXXXXX."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("254"):
        digits = "0" + digits[3:]
    return digits


async def _fetch_customer_snapshot(phone_number: str) -> dict:
    """Fetches the customer snapshot for a (Kenyan-normalized) phone number.

    Raises requests.HTTPError/RequestException on failure — callers decide
    how to handle that. The blocking `requests` call is offloaded to a
    thread so this can run concurrently with other async work (e.g. the
    greeting WAV playing, or provider setup).

    Confirmed this same fss-api endpoint serves Kenya customer data too
    (looked up by 254-format-normalized phone number) — no separate
    Kenya-specific endpoint needed.
    """
    normalized = _normalize_kenyan_phone(phone_number)
    url = f"https://fss-api.onrender.com/customer_snapshot/{normalized}"
    resp = await asyncio.to_thread(requests.get, url, timeout=15)
    resp.raise_for_status()
    return resp.json()


@function_tool(
    description=(
        "Get customer info from the API endpoint once the caller's phone number is available. "
        "Pass the phone number in Kenyan local format (0XXXXXXXXX): remove spaces/special "
        "characters and replace a leading 254 country code with 0. "
        "Example: 254712345678 → 0712345678."
    )
)
async def get_customer_info(ctx: RunContext, phone_number: str) -> str:
    normalized = _normalize_kenyan_phone(phone_number)
    try:
        data = await _fetch_customer_snapshot(phone_number)
        active_lead = data.get("active_lead") or {}
        first_name = active_lead.get("first_name", "")
        last_name = active_lead.get("surname", "")
        return (
            f"Customer found. first_name={first_name}, last_name={last_name}. "
            f"Full response: {data}"
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"No customer record found for phone number {normalized}."
        return f"HTTP error fetching customer info: {e}"
    except requests.RequestException as e:
        return f"Error fetching customer info: {e}"


@function_tool(
    description=(
        "Switch the language you speak in for the rest of the call. Pass "
        "'english' or 'swahili'. Call this once, immediately after you "
        "determine which language the customer is using from their first "
        "substantive response, per the Language Selection Rule — before "
        "that, you are already speaking English by default, so there is no "
        "need to call this if the customer's first response is in English. "
        "Call it again only if the customer explicitly asks you to switch "
        "language mid-call. This does not change what you understand — you "
        "always understand English, Swahili, and Sheng regardless of which "
        "language is currently active — it only changes which language your "
        "own voice speaks in."
    )
)
async def set_active_language(ctx: RunContext, language: str) -> str:
    normalized = language.strip().lower()
    tts_by_language = ctx.userdata.get("tts_by_language") if ctx.userdata else None
    if not tts_by_language or normalized not in tts_by_language:
        return f"Unknown language '{language}'. Use 'english' or 'swahili'."
    ctx.session.current_agent.update_options(tts=tts_by_language[normalized])
    return f"Active language switched to {normalized}."


@function_tool(
    description=(
        "End the call and hang up. Call this whenever the prompt instructs you to "
        "'stop completely' — after your final spoken message, a dead-end exit, or an "
        "escalation close. Always speak your closing message first; this tool waits "
        "for it to finish playing before disconnecting, so it is safe to call "
        "immediately after your final message."
    )
)
async def end_call(ctx: RunContext) -> None:
    await ctx.wait_for_playout()
    job_ctx = get_job_context()
    job_ctx.shutdown(reason="call ended by agent")
    asyncio.create_task(_ensure_call_ends(job_ctx))


# ctx.shutdown() only signals the job runner to *start* an async teardown
# (wait for the entrypoint task, session.aclose(), room.disconnect(), then
# our own shutdown callback's ctx.delete_room() — the actual SIP hangup).
# If any step in that chain stalls, the call stays connected and the agent
# keeps responding to the customer indefinitely, even though end_call
# already ran. Rather than trust that chain to always complete, check back
# after a grace period and, if the room is still connected, force the
# actual hangup directly and re-signal shutdown — retrying a few times in
# case the stall (or the retry itself) was transient.
_END_CALL_RETRY_DELAY_SECONDS = 5
_END_CALL_MAX_RETRIES = 3


async def _ensure_call_ends(job_ctx: JobContext) -> None:
    for attempt in range(1, _END_CALL_MAX_RETRIES + 1):
        await asyncio.sleep(_END_CALL_RETRY_DELAY_SECONDS)
        if job_ctx.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            return  # call actually ended; nothing more to do
        print(
            f"end_call: room still connected {attempt * _END_CALL_RETRY_DELAY_SECONDS}s "
            f"after shutdown was requested; forcing hangup directly (attempt {attempt})"
        )
        try:
            await job_ctx.delete_room()
        except Exception as e:
            print(f"end_call retry: failed to delete room: {e}")
        job_ctx.shutdown(reason="call ended by agent (retry)")

    if job_ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        print("end_call: room still connected after all retries; giving up")


_NAIROBI_TZ = ZoneInfo("Africa/Nairobi")


@function_tool(
    description=(
        "Re-check the current date and time in Kenya (Africa/Nairobi, East "
        "Africa Time), plus the two standard payment deadline dates ('the 7-day "
        "deadline' and 'the 30-day deadline'). Your instructions already give "
        "you today's date and both deadline dates at the start of this call — "
        "treat those as authoritative and do not call this tool to re-derive "
        "them. Only call this if the customer disputes today's date, the call "
        "has been running for an unusually long time, or you need to convert "
        "something the customer says ('this Friday', 'kesho') into an exact "
        "calendar date. Never guess, assume, do your own date arithmetic, or "
        "rely on your training data for today's date — call this instead, and "
        "always speak the actual returned dates, never the '7 days'/'30 days' "
        "reasoning behind them."
    )
)
async def get_current_datetime(ctx: RunContext) -> str:
    now = datetime.now(_NAIROBI_TZ)
    seven_day_deadline = now + timedelta(days=7)
    thirty_day_deadline = now + timedelta(days=30)
    return (
        f"Current date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p')} "
        f"East Africa Time. ISO date: {now.date().isoformat()}. "
        f"7-day payment deadline (speak this exact date, never 'in 7 days' or "
        f"'this week'): {seven_day_deadline.strftime('%A, %B %d, %Y')}. "
        f"30-day payment deadline (speak this exact date, never 'in 30 days' or "
        f"'end of month'): {thirty_day_deadline.strftime('%A, %B %d, %Y')}."
    )


# ── Version-safe ElevenLabs TTS builder ─────────────────────────────────
def build_elevenlabs_tts():
    """
    Builds an ElevenLabs TTS instance that works across multiple versions of
    livekit-plugins-elevenlabs by:
      - Creating VoiceSettings with only supported kwargs
      - Creating Voice with only supported kwargs
      - Passing either `voice` or `voice_id` to TTS depending on signature
      - Prefer PCM encoding when available to bypass MP3 decoding
      - Filtering optional params (language, streaming_latency, enable_ssml_parsing, encoding)
    Uses ELEVEN_API_KEY from the environment (mapped from ELEVENLABS_API_KEY above).

    Uses the direct ElevenLabs plugin rather than LiveKit Inference
    specifically because our voice_id below is a custom cloned voice on our
    own ElevenLabs account — LiveKit Inference only supports ElevenLabs'
    own default voices, not custom/cloned ones.
    """
    model     = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
    # TODO: "me1JPr2K6H7KZB9nz2Wk" is va_bot.py's Nigeria voice clone — carried
    # over as a placeholder only. Needs a real decision: either clone a Kenya
    # agent voice the same way, or pick a stock ElevenLabs voice confirmed to
    # sound natural in both English and Swahili (flash_v2_5 handles the
    # language switch fine per testing, but voice/accent choice is separate).
    voice_id  = os.getenv("ELEVENLABS_VOICE_ID", "me1JPr2K6H7KZB9nz2Wk")
    v_name    = os.getenv("ELEVENLABS_VOICE_NAME", "custom")
    v_cat     = os.getenv("ELEVENLABS_VOICE_CATEGORY", "premade")

    desired_vs = {
        "stability": 0.9,
        "similarity_boost": 0.9,
        "style": 0.9,
        "use_speaker_boost": False,#True,
        "speed": .4,  # dropped automatically if unsupported
    }

    def _filter_kwargs(cls, kwargs):
        try:
            params = set(inspect.signature(cls).parameters.keys())
            return {k: v for k, v in kwargs.items() if k in params}
        except (ValueError, TypeError):
            return {}

    # VoiceSettings
    voice_settings = None
    try:
        vs_kwargs = _filter_kwargs(elevenlabs.VoiceSettings, desired_vs)
        if vs_kwargs:
            try:
                voice_settings = elevenlabs.VoiceSettings(**vs_kwargs)
            except TypeError:
                for k in list(vs_kwargs.keys()):
                    test = dict(vs_kwargs)
                    test.pop(k, None)
                    try:
                        voice_settings = elevenlabs.VoiceSettings(**test)
                        break
                    except TypeError:
                        continue
    except AttributeError:
        voice_settings = None

    # Voice
    voice = None
    try:
        voice_kwargs = _filter_kwargs(
            elevenlabs.Voice,
            {"id": voice_id, "name": v_name, "category": v_cat, "settings": voice_settings},
        )
        if "settings" in voice_kwargs and voice_kwargs["settings"] is None:
            voice_kwargs.pop("settings", None)
        voice = elevenlabs.Voice(**voice_kwargs)
    except AttributeError:
        voice = None  # fall back to voice_id on TTS

    # TTS kwargs
    try:
        tts_params = set(inspect.signature(elevenlabs.TTS).parameters.keys())
    except (ValueError, TypeError):
        tts_params = set()

    tts_kwargs = {"model": model}

    if "voice" in tts_params and voice is not None:
        tts_kwargs["voice"] = voice
    elif "voice_id" in tts_params:
        tts_kwargs["voice_id"] = voice_id

    # Prefer PCM encoding if exposed (avoids MP3 decoder path).
    # The plugin exposes an 'encoding' parameter with a TTSEncoding enum in newer builds.
    try:
        Enc = getattr(elevenlabs, "TTSEncoding", None)
        chosen_enc = None
        if Enc is not None:
            # Try common PCM enum names across releases
            for attr in ("pcm", "pcm_16000", "s16le_16000", "linear16", "pcm16_16000"):
                if hasattr(Enc, attr):
                    chosen_enc = getattr(Enc, attr)
                    break
        if "encoding" in tts_params and chosen_enc is not None:
            tts_kwargs["encoding"] = chosen_enc
    except Exception:
        pass

    # Optional args
    # No "language" override here (unlike va_bot.py's Nigeria version, which
    # pins "en") — customers here switch between English and Swahili within
    # a call, and flash_v2_5 auto-detects the language per request from the
    # text itself; pinning "en" would fight that on Swahili turns.
    for opt_key, opt_val in {
        "streaming_latency": 3,
        "enable_ssml_parsing": True,
        # some versions also support inactivity_timeout, chunk_length_schedule, etc.
    }.items():
        if opt_key in tts_params:
            tts_kwargs[opt_key] = opt_val

    return elevenlabs.TTS(**tts_kwargs)


# ── Azure TTS builder ────────────────────────────────────────────────────
def build_azure_tts(voice: str | None = None):
    """
    Builds an Azure neural TTS voice. AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
    are picked up automatically from the environment by azure.TTS itself when
    not passed explicitly.

    Azure neural voices are locale-specific — en-KE-AsiliaNeural only speaks
    English, sw-KE-ZuriNeural/sw-KE-RafikiNeural only speak Swahili, there's
    no auto-detecting multilingual Azure voice the way ElevenLabs' models
    work. That's why this now takes an explicit `voice` argument rather than
    a single hardcoded default: the entrypoint below builds one
    FallbackAdapter per active language (english/swahili), each pairing the
    matching Azure voice as primary with ElevenLabs as fallback, and swaps
    between them at runtime via the set_active_language tool. Falls back to
    AZURE_TTS_VOICE / en-KE-AsiliaNeural if no voice is passed, so this still
    works standalone.
    """
    resolved = voice or os.getenv("AZURE_TTS_VOICE", "en-KE-AsiliaNeural")
    return azure.TTS(voice=resolved)


# ── SIP participant + caller phone number ────────────────────────────────
async def _wait_for_sip_participant(
    room: rtc.Room, timeout: float = 10.0
) -> rtc.RemoteParticipant | None:
    """Waits for the SIP participant to join. On outbound calls the
    participant may not be present immediately after connect()."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for participant in room.remote_participants.values():
            if participant.attributes.get("sip.callID"):
                return participant
        await asyncio.sleep(0.2)
    return None


def _get_caller_phone_number(participant: "rtc.RemoteParticipant | None") -> str | None:
    """Extracts the customer's phone number from SIP participant attributes.

    Our inbound trunk sets `"includeHeaders": "SIP_ALL_HEADERS"`, which maps
    every INVITE header LiveKit sees onto a `sip.h.<lowercase-header-name>`
    participant attribute. Our Asterisk dialplan sets an `X-Caller-ID` header
    carrying the real customer number, so that lands on the attribute key
    `sip.h.x-caller-id` — NOT `sip.headers.X-Caller-ID` / `X-Caller-ID`
    (previous, incorrect keys).

    `sip.phoneNumber` is only used as a last-resort fallback: on this setup
    it reflects the dialplan's fixed `fromuser` (our own DID), not the actual
    caller, because Asterisk relays every call into LiveKit under that same
    From value.
    """
    if participant is None:
        return None
    phone_number = participant.attributes.get("sip.h.x-caller-id")
    if phone_number:
        return phone_number.strip()
    return participant.attributes.get("sip.phoneNumber")


# ── Pre-recorded greeting (bypasses TTS) ─────────────────────────────────
GREETING_WAV_PATH = os.path.join(os.getcwd(), "Greeting.wav")
# Fallback bridge clip: played once if the agent still hasn't spoken again
# OPENING_STALL_DELAY_SECONDS after GREETING_WAV_PATH finishes (customer
# info / AMD still resolving).
GREETING_V2_WAV_PATH = os.path.join(os.getcwd(), "Greeting v2.wav")
OPENING_STALL_DELAY_SECONDS = 3
# From the agent's first real turn onward, how long it can sit "thinking"
# before speaking a short "Hmm" filler while the real reply keeps loading.
HMM_FILLER_DELAY_SECONDS = 3
# How long to wait after the "Are you still with me?" check-in before
# giving up and hanging up, if the caller still hasn't responded.
SILENCE_SHUTDOWN_DELAY_SECONDS = 10


async def _play_wav_greeting(room: rtc.Room, wav_path: str) -> None:
    """Publishes a pre-recorded WAV directly to the room as a raw audio
    track, bypassing TTS entirely. Avoids TTS cold-start latency (and any
    single provider's outage) for this one fixed opening line. The agent
    stays silent until the caller actually replies, since nothing here
    touches the LLM/STT turn-taking pipeline."""
    if not os.path.exists(wav_path):
        print(f"Greeting WAV not found at {wav_path}; skipping.")
        return

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            print(f"Greeting WAV must be 16-bit PCM, got {sample_width * 8}-bit; skipping.")
            return

        source = rtc.AudioSource(sample_rate, num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("greeting", source)
        publication = await room.local_participant.publish_track(track)

        frame_ms = 20
        frame_samples = int(sample_rate * frame_ms / 1000)
        bytes_per_frame = frame_samples * num_channels * sample_width

        try:
            while True:
                chunk = wf.readframes(frame_samples)
                if not chunk:
                    break
                if len(chunk) < bytes_per_frame:
                    chunk += b"\x00" * (bytes_per_frame - len(chunk))
                audio_frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=len(chunk) // (2 * num_channels),
                )
                await source.capture_frame(audio_frame)
        finally:
            await asyncio.sleep(0.5)
            await room.local_participant.unpublish_track(publication.sid)


# ── Entrypoint ──────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room {ctx.room.name}")

    is_console = ctx.room.name == "console" or ctx.room.name.startswith("console-")

    # Call recording + transcript storage (Firebase). Kicked off as a
    # background task (not awaited here) so the network round-trip to the
    # Egress API doesn't block everything downstream — SIP participant wait,
    # provider setup, AMD, and the greeting — from starting immediately.
    # Recording doesn't gate any of that; it's only actually needed later,
    # in _on_shutdown.
    call_started_at = time.time()
    call_state = {"amd_category": None}
    transcript_lines: list[str] = []
    # Constructed here (immediately, no dependency on session/provider setup)
    # rather than right before it's first used further down — _on_shutdown is
    # registered just below and reads this as a closure variable. If it were
    # only assigned later, any crash in between (e.g. provider setup blowing
    # up, as the STT FallbackAdapter did before vad= was added) would leave
    # this unbound, and the shutdown handler would itself raise a NameError
    # on top of — and masking — the real error in the logs.
    usage_collector = ModelUsageCollector()
    egress_task = asyncio.create_task(_start_call_recording(ctx))
    max_duration_task: asyncio.Task | None = None
    opening_stall_task: asyncio.Task | None = None
    hmm_filler_task: asyncio.Task | None = None
    silence_shutdown_task: asyncio.Task | None = None

    async def _on_shutdown():
        # ctx.shutdown() (used everywhere we decide to end the call) only
        # disconnects our own agent from the room — it does NOT hang up the
        # actual phone call. The SIP participant stays connected with dead
        # air unless the room itself is deleted. Do this first, immediately,
        # rather than after the (up to ~30s) recording finalization below,
        # so the customer's line actually drops the moment the agent decides
        # to end the call. No-ops safely in console mode.
        try:
            await ctx.delete_room()
        except Exception as e:
            print(f"Failed to delete room / hang up call: {e}")

        # Stop the max-duration timer if the call ended for some other
        # reason first (end_call, AMD, silence timeout, caller hangup) —
        # otherwise it's still sleeping and would fire pointlessly later.
        if max_duration_task is not None:
            max_duration_task.cancel()
        if opening_stall_task is not None:
            opening_stall_task.cancel()
        if hmm_filler_task is not None:
            hmm_filler_task.cancel()
        if silence_shutdown_task is not None:
            silence_shutdown_task.cancel()

        egress_info = await egress_task
        recording_url = await _finish_call_recording(ctx, egress_info)
        usage = [u.model_dump(mode="json") for u in usage_collector.flatten()]
        _save_call_record(
            ctx.room.name,
            transcript_lines,
            call_state["amd_category"],
            recording_url,
            datetime.fromtimestamp(call_started_at, tz=timezone.utc),
            time.time() - call_started_at,
            usage,
            phone_number,
        )

    ctx.add_shutdown_callback(_on_shutdown)

    # SIP participant + caller phone number. Skipped in console mode: no SIP
    # participant will ever join a local mic-test session.
    sip_participant = None
    phone_number = None
    lookup_task = None
    if not is_console:
        sip_participant = await _wait_for_sip_participant(ctx.room)
        if sip_participant:
            print(f"SIP participant joined: {sip_participant.identity}")
            print(f"Participant attributes: {dict(sip_participant.attributes)}")
            phone_number = _get_caller_phone_number(sip_participant)
        else:
            print("Warning: no SIP participant found within timeout")
        print(f"Customer phone number: {phone_number}")

        if phone_number:
            lookup_task = asyncio.create_task(_fetch_customer_snapshot(phone_number))

    # Load system prompt (fallback if file missing)
    prompt_path = os.path.join(os.getcwd(), "prompt_kenya.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "You are Shanice, a debt collection assistant."

    # Ground the model in the real current date via code, not tool-call
    # discipline. Ported from va_bot.py: the model was observed relying on
    # get_current_datetime being called (and remembered correctly) throughout
    # a long negotiation, and instead doing its own date arithmetic mid-call
    # and confidently stating a wrong year for the 7-day/30-day deadlines.
    # Baking the actual dates into the instructions means they're present in
    # every single turn's context for the rest of the call — nothing to
    # recall, nothing to compute, nothing to get wrong. get_current_datetime
    # still exists as a fallback for edge cases (the customer disputes
    # today's date, or the call runs unusually long).
    _call_start_dt = datetime.now(_NAIROBI_TZ)
    _seven_day_deadline_dt = _call_start_dt + timedelta(days=7)
    _thirty_day_deadline_dt = _call_start_dt + timedelta(days=30)
    system_prompt += (
        f"\n\nCURRENT DATE CONTEXT (authoritative for this entire call — never "
        f"override with your own sense of today's date, and never do your own "
        f"date arithmetic): today is "
        f"{_call_start_dt.strftime('%A, %B %d, %Y')}. The 7-day payment "
        f"deadline is {_seven_day_deadline_dt.strftime('%A, %B %d, %Y')}. The "
        f"30-day payment deadline is "
        f"{_thirty_day_deadline_dt.strftime('%A, %B %d, %Y')}. These are fixed "
        f"for the rest of this call — speak them exactly as given here, never "
        f"as 'in 7 days', 'this week', 'in 30 days', or 'end of month'. Only "
        f"call get_current_datetime if the customer disputes today's date or "
        f"the call has been running for a long time."
    )

    # Providers — STT and LLM via LiveKit Inference (livekit.agents.inference),
    # billed through your LiveKit Cloud account: no DEEPGRAM_API_KEY,
    # GOOGLE_API_KEY, OPENAI_API_KEY needed for those two.
    #
    # TTS is the exception (see below): it uses direct Azure and ElevenLabs
    # plugins instead of Inference, so AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    # and ELEVENLABS_API_KEY are still required for that component.
    #
    # STT: ElevenLabs Scribe v2 Realtime primary, OpenAI transcription as
    # fallback. NOT Deepgram Nova-3 / AssemblyAI Universal-Streaming (the
    # Nigeria bot's primary/fallback) — verified against both providers'
    # published language lists that neither supports Swahili at all, so for
    # Kenya they'd silently mangle any Swahili speech rather than erroring,
    # which is worse than not having them in the chain. Scribe v2 has
    # confirmed Swahili support; no language pinned here (unlike the Nigeria
    # version's language="en") so it can auto-detect English vs Swahili
    # per utterance rather than being forced into one.
    #
    # OpenAI fallback via gpt-4o-mini-transcribe (detect_language=True):
    # confirmed to support Swahili among its 99+ languages, though OpenAI's
    # own docs note lower accuracy on lower-resource languages like Swahili
    # compared to major languages — still a real fallback, unlike Deepgram/
    # AssemblyAI which simply don't have the language at all.
    #
    # Sheng (Nairobi's English/Swahili code-mixed slang) has no dedicated
    # STT support anywhere — no provider treats it as a distinct language.
    # Both engines below get a workable-but-imperfect transcription of it as
    # a mix of their English/Swahili training data; the prompt's job is to
    # interpret that via context, the same way va_bot.py's Nigeria prompt
    # already handles Pidgin.
    #
    # Silero VAD runs locally, not through any external provider — it never
    # needed an API key, so it's left as-is (still prewarmed per worker
    # process in prewarm(), not per call). Needed here (ahead of the STT
    # FallbackAdapter below, unlike va_bot.py's Nigeria version where every
    # STT streams natively) so openai.STT can be auto-wrapped for streaming.
    vad = ctx.proc.userdata["vad"]
    # vad= is required here: openai.STT doesn't support streaming, and
    # FallbackAdapter raises at construction time unless every non-streaming
    # STT in the list can be auto-wrapped with stt.StreamAdapter via a VAD —
    # caught by a console-mode smoke test before this ever reached a real
    # call (every call would have crashed on connect otherwise).
    stt = agents_stt.FallbackAdapter(
        [
            # model="scribe_v2_realtime" already selects realtime mode; the
            # plugin logs a warning if use_realtime=True is also passed
            # alongside it (it's simply ignored), so it's left out here.
            elevenlabs.STT(model="scribe_v2_realtime"),
            openai.STT(detect_language=True),
        ],
        vad=vad,
    )
    # LLM: Gemini 3.1 Flash-Lite as primary, gpt-5-mini as fallback if
    # Gemini errors out. Worth knowing: Gemini 2.5 (Flash and Flash-Lite) has
    # a documented upstream quirk where it can return finish_reason=STOP
    # with no actual text/candidates — most commonly reported with
    # tool-calling agents like this one. Not independently confirmed on
    # 3.1 specifically, but keeping the fallback below either way. When the
    # gateway surfaces that as an error, this fallback catches it; if it
    # doesn't raise (silently empty instead), this won't help and you'd
    # want to drop Gemini entirely at that point.
    #
    # gpt-5-mini: not passing temperature — GPT-5-series models reject a
    # custom value (400 error unless left at the API default of 1).
    #
    # xai/grok-4-1-fast-non-reasoning as third fallback: cheapest credible
    # option on the whole Inference LLM rate card ($0.20/$0.50 in/out per
    # million tokens — below both Gemini Flash-Lite and gpt-5-mini, and a
    # fraction of gpt-5-mini's $2.00 output rate), and a third distinct
    # provider family (neither Google nor OpenAI) so a Gemini- or
    # OpenAI-wide outage doesn't take out two of the three at once. The
    # "non-reasoning" variant specifically, not "-reasoning": same price on
    # LiveKit's rate card, but reasoning mode spends extra tokens on
    # internal chain-of-thought before answering, which only adds latency
    # here — this prompt needs reliable instruction-following and tool
    # calls, not multi-step reasoning.
    llm = agents_llm.FallbackAdapter(
        [
            inference.LLM("google/gemini-3.1-flash-lite"),
            inference.LLM("openai/gpt-5-mini"),
            inference.LLM("xai/grok-4-1-fast-non-reasoning"),
        ]
    )

    # TTS: Azure Kenyan-accented neural voice as primary, ElevenLabs
    # (flash_v2_5) as fallback if Azure errors mid-call. NOT Fish Audio /
    # Inworld (the Nigeria bot's primary/2nd fallback) — neither has
    # confirmed Swahili support (Inworld TTS-1.5's 15 languages definitively
    # exclude it; Fish Audio S2.1 Pro's exact language list isn't publicly
    # enumerated).
    #
    # Azure neural voices are locale-specific — en-KE-AsiliaNeural only
    # speaks English, sw-KE-ZuriNeural only speaks Swahili, there's no
    # auto-detecting multilingual Azure voice the way ElevenLabs works. So
    # unlike a simple FallbackAdapter, Azure-as-primary needs ONE
    # FallbackAdapter per active language — each pairing the matching Azure
    # voice with ElevenLabs as fallback — and the active one gets swapped at
    # runtime via the set_active_language tool (see its docstring) as soon
    # as the model determines which language the customer is using, per the
    # Language Selection Rule. ElevenLabs auto-detects English vs Swahili
    # from the text itself, so it works correctly as the fallback regardless
    # of which language Azure was speaking when it failed — confirmed via a
    # direct test: generating Swahili speech with flash_v2_5, then feeding
    # that audio back through Swahili-capable STT, round-tripped with a
    # perfect transcript, despite flash_v2_5 not being officially
    # Swahili-listed. (Only eleven_v3 is officially Swahili-listed, but
    # ElevenLabs explicitly advises against v3 for real-time conversational
    # use due to higher latency — flash_v2_5 is the right tradeoff here.)
    #
    # max_retry_per_tts=0 on both: on any mid-stream failure, switch to the
    # fallback immediately instead of retrying the same provider first. Each
    # retry re-opens a fresh stream for the same text; if the failed attempt
    # had already pushed partial audio to the room before erroring, a retry
    # starts that same sentence over from the beginning — overlapping with
    # the tail of what's already playing. Skipping straight to the fallback
    # on the first failure avoids that overlap.
    tts_by_language = {
        "english": agents_tts.FallbackAdapter(
            [build_azure_tts(voice="en-KE-AsiliaNeural"), build_elevenlabs_tts()],
            max_retry_per_tts=0,
        ),
        "swahili": agents_tts.FallbackAdapter(
            [build_azure_tts(voice="sw-KE-ZuriNeural"), build_elevenlabs_tts()],
            max_retry_per_tts=0,
        ),
    }
    # Every call opens in English per the Language Selection Rule — the
    # model hasn't heard the customer speak yet, so there's nothing to match.
    tts = tts_by_language["english"]



    # Session
    session = AgentSession(
        stt=stt,
        llm=llm,
        vad=vad,
        tts=tts,
        # Exposed to the set_active_language tool via ctx.userdata, so it
        # can swap the active per-language FallbackAdapter at runtime — see
        # the TTS setup above and set_active_language's docstring.
        userdata={"tts_by_language": tts_by_language},
        turn_detection=TurnDetector(),
        preemptive_generation=True,
        # Real call transcripts showed a repeating "Hello? Hello? Hello?"
        # cascade: the customer says a bare "Hello?" mid-sentence, it cuts
        # off Shanice's current speech as a real interruption, she restarts,
        # gets cut off again, and it loops. A 1-word utterance doesn't hit
        # min_interruption_duration (0.5s default) reliably enough to be
        # filtered by that alone. min_interruption_words=2 stops a single
        # word like "Hello"/"Hi"/"Yes"/"No" from counting as an
        # interruption at all — it only suppresses cutting Shanice off
        # mid-sentence; the transcript is still tracked, so if the customer
        # keeps talking past 2 words it interrupts normally, and a bare
        # "Hello?" that stops there still gets handled as the next turn
        # once her current line finishes. Trade-off: genuine one-word
        # interruptions ("Stop!", "Wait!") also won't cut her off
        # instantly — they wait for her current (already short) sentence to
        # finish rather than being lost. "Hold on" / "Wait, stop" (2+
        # words) still interrupt immediately.
        min_interruption_words=2,
        # How long the customer can go quiet before the silence safety net
        # below (_on_user_state_changed) fires "Are you still with me?".
        # Framework default is 15s; shortened so a genuinely dead line (or a
        # stuck turn, e.g. from an STT language-detection glitch) gets
        # caught and closed out faster.
        user_away_timeout=6.0,
    )

    # session.emit("metrics_collected", MetricsCollectedEvent(metrics=...)) wraps
    # the metrics — ModelUsageCollector.collect() expects the raw metrics object,
    # so it must be unwrapped here or every call silently no-ops.
    session.on("metrics_collected", lambda ev: usage_collector.collect(ev.metrics))

    # Deterministic backstop for the "Hello? Hello? Hello?" pattern found in
    # real call transcripts — see _is_greeting_only_utterance above for why
    # this doesn't depend on the model's own judgment. Consecutive, not a
    # whole-call total: a call that recovers into a real conversation after
    # a rocky start (which happens — seen in real data) must not be killed
    # just because it had a few greeting-only turns much earlier.
    HEARING_DIFFICULTY_BACKSTOP_THRESHOLD = 3
    consecutive_greeting_only_turns = 0

    async def _end_call_for_hearing_difficulty() -> None:
        try:
            await session.say(
                "It seems we're having trouble hearing each other — I'll try you again another time.",
                allow_interruptions=False,
            )
        except Exception as e:
            print(f"Failed to speak hearing-difficulty backstop closing line: {e}")
        ctx.shutdown(reason="hearing-difficulty backstop: repeated greeting-only exchanges")

    def _on_conversation_item_added(ev):
        nonlocal hmm_filler_task, consecutive_greeting_only_turns
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role and text:
            transcript_lines.append(f"{role}: {text}")
            # Diagnostic logging (STT/turn-detection reliability investigation,
            # Aug 2026): a real test call showed clear English speech never
            # becoming a registered turn at all — this line at least confirms
            # WHEN a turn does land, for comparison against what was actually
            # said on the recording.
            print(f"[turn] conversation_item_added role={role} text={text!r}")
        # Belt-and-suspenders: a real assistant utterance just landed, so any
        # pending "Hmm" filler is moot. Cancelling only on the "speaking"
        # state transition isn't always enough — a slow/chunked TTS
        # generation can bounce agent_state between "thinking" and
        # "speaking" mid-turn, occasionally leaving a stale filler timer
        # armed that then fires right after real content was just spoken.
        if role == "assistant" and hmm_filler_task is not None and not hmm_filler_task.done():
            hmm_filler_task.cancel()
        if role == "user" and text:
            if _is_greeting_only_utterance(text):
                consecutive_greeting_only_turns += 1
                if consecutive_greeting_only_turns >= HEARING_DIFFICULTY_BACKSTOP_THRESHOLD:
                    asyncio.create_task(_end_call_for_hearing_difficulty())
            else:
                consecutive_greeting_only_turns = 0

    session.on("conversation_item_added", _on_conversation_item_added)

    # Two-tier "the bot has gone quiet too long" filler system:
    #   - Before the agent's first real turn (the raw-track WAV greeting
    #     below doesn't count — it bypasses this session's TTS entirely):
    #     OPENING_STALL_DELAY_SECONDS of silence after that WAV greeting
    #     finishes triggers a second pre-recorded WAV bridge clip, since
    #     customer-info lookup / AMD may still be resolving. This covers the
    #     window before the caller has said anything yet, so agent_state
    #     isn't "thinking" and the Hmm filler below can't apply to it.
    #   - Once the caller's first turn lands and the agent is "thinking" —
    #     including the very first real reply, e.g. while get_customer_info
    #     / get_current_datetime tool calls are running — HMM_FILLER_DELAY_SECONDS
    #     of continuous "thinking" triggers a short spoken "Hmm" filler via
    #     normal TTS, re-arming for as long as that turn keeps thinking.
    # Both are cancelled the instant the agent actually starts speaking; the
    # Hmm filler is also cancelled the moment a real assistant utterance is
    # added to the transcript (see _on_conversation_item_added above), since
    # a slow/chunked TTS generation can otherwise bounce agent_state between
    # "thinking" and "speaking" mid-turn and leave a stale timer armed.
    first_response_delivered = False

    async def _play_opening_stall_after_delay() -> None:
        await asyncio.sleep(OPENING_STALL_DELAY_SECONDS)
        await _play_wav_greeting(ctx.room, GREETING_V2_WAV_PATH)

    async def _speak_hmm_after_delay() -> None:
        await asyncio.sleep(HMM_FILLER_DELAY_SECONDS)
        # add_to_chat_ctx=False: this is a stalling interjection, not real
        # dialogue — it must never enter the LLM's own conversation history,
        # and if TTS happens to be down when this fires, it must not leave a
        # phantom "assistant said this" line in the transcript either.
        session.say("Hmm?", allow_interruptions=True, add_to_chat_ctx=False)

    def _on_agent_state_changed(ev) -> None:
        nonlocal first_response_delivered, opening_stall_task, hmm_filler_task
        # Diagnostic logging (STT/turn-detection reliability investigation,
        # Aug 2026) — correlates against [user_state]/[stt] log lines to see
        # exactly what the agent was doing when a turn should have landed.
        print(f"[agent_state] {ev.old_state} -> {ev.new_state}")
        if ev.new_state == "speaking":
            if not first_response_delivered:
                first_response_delivered = True
                if opening_stall_task is not None and not opening_stall_task.done():
                    opening_stall_task.cancel()
            if hmm_filler_task is not None and not hmm_filler_task.done():
                hmm_filler_task.cancel()
            return
        if ev.new_state == "thinking":
            if hmm_filler_task is None or hmm_filler_task.done():
                hmm_filler_task = asyncio.create_task(_speak_hmm_after_delay())
        elif hmm_filler_task is not None and not hmm_filler_task.done():
            hmm_filler_task.cancel()

    session.on("agent_state_changed", _on_agent_state_changed)

    # Silence safety net: the LLM only reacts to turns, so if the customer
    # goes quiet it can't act on its own. After user_away_timeout (6s, set
    # above) of no user activity, check in once, then hang up if nothing
    # follows.
    #
    # Registered here — before session.start()/the greeting/AMD — rather
    # than after them: AgentSession arms its away-timer as soon as the
    # session starts, so a listener only attached after session.start()
    # (where this used to live) could miss the very first "away" transition
    # entirely. That transition doesn't reliably repeat on its own
    # (AgentSession only re-arms the timer when the agent and user both
    # transition INTO "listening" at the same moment), so a missed one means
    # no safety net at all until some unrelated event happens to re-arm it.
    # AMD only pauses the agent's own auto-reply authorization
    # (AgentActivity._pause_authorization), not user-state tracking, so this
    # stays live and correct straight through the greeting/AMD window too —
    # session.say() below simply queues until AMD releases that lock if this
    # fires mid-detection.
    checked_in = False

    def _cancel_pending_silence_shutdown() -> None:
        nonlocal checked_in, silence_shutdown_task
        checked_in = False
        if silence_shutdown_task is not None and not silence_shutdown_task.done():
            silence_shutdown_task.cancel()

    def _on_user_state_changed(ev):
        nonlocal checked_in, silence_shutdown_task
        # Diagnostic logging (STT/turn-detection reliability investigation,
        # Aug 2026) — every VAD-level state transition, to correlate against
        # actual speech timing from the call recording.
        print(f"[user_state] {ev.old_state} -> {ev.new_state}")
        if ev.new_state == "speaking":
            # Unambiguous real speech (VAD-confirmed, not just a stray final
            # transcript) — safe to cancel here directly rather than waiting
            # on _on_user_input_transcribed, which can lag several seconds
            # behind the STT provider (e.g. mid-fallover on the STT
            # FallbackAdapter).
            _cancel_pending_silence_shutdown()
            return
        if ev.new_state != "away":
            # A bare away→listening transition isn't itself trustworthy: per
            # AgentSession._user_input_transcribed, ANY final transcript —
            # including an empty one from a stray noise/VAD-miss blip, or —
            # seen directly in a real Kenya test call — a language-detection
            # glitch on an ambiguous short utterance producing hallucinated
            # foreign-language text — resets away straight back to
            # listening, with no content check. Don't cancel our own
            # countdown on that alone; wait for _on_user_input_transcribed
            # below to confirm real content actually came through.
            return
        if not checked_in:
            checked_in = True
            # session.say() returns a SpeechHandle (already queued/running,
            # not a coroutine) — wrapping it in asyncio.create_task() raises
            # TypeError, so it's called directly, fire-and-forget.
            session.say("Are you still with me?", allow_interruptions=True)
            silence_shutdown_task = asyncio.create_task(_shutdown_after_silence())

    async def _shutdown_after_silence() -> None:
        await asyncio.sleep(SILENCE_SHUTDOWN_DELAY_SECONDS)
        ctx.shutdown(reason="no response after silence check")

    def _on_user_input_transcribed(ev) -> None:
        # Diagnostic logging (STT/turn-detection reliability investigation,
        # Aug 2026): a real test call showed the caller speaking two clear,
        # ordinary English sentences (confirmed independently by transcribing
        # the call recording itself afterward) that never once produced a
        # registered turn — no STT-level exception was logged either, so
        # whatever failed did so silently. This logs EVERY event from this
        # handler, interim and final alike, with the detected language, so
        # the next occurrence is caught live instead of reconstructed after
        # the fact from a recording.
        print(
            f"[stt] is_final={ev.is_final} language={ev.language!r} "
            f"transcript={ev.transcript!r}"
        )
        # Without this: a stray or hallucinated *final* transcript (empty
        # noise blip, or a language-detection glitch producing foreign-
        # language text from an ambiguous short utterance — both observed in
        # practice) resets the framework's away state to "listening"
        # unconditionally (see note above) and, previously, cancelled our
        # hang-up countdown right along with it — so the countdown never ran
        # to completion and the agent was left stuck with no recovery path
        # for the rest of the call. Only a final transcript with actual
        # non-blank content counts as evidence the caller is really there.
        if ev.is_final and ev.transcript.strip():
            _cancel_pending_silence_shutdown()

    session.on("user_state_changed", _on_user_state_changed)
    session.on("user_input_transcribed", _on_user_input_transcribed)

    # Build the initial instructions synchronously — no blocking wait on the
    # customer lookup here. This previously did `await asyncio.wait_for(
    # lookup_task, timeout=3)` before creating the Agent, which added a fixed
    # delay of up to 3s before session.start()/AMD/the greeting on every
    # single call, regardless of where the greeting itself sits in the flow.
    # The lookup hits an external Render-hosted API, which can be slow to
    # wake from a cold start — so that 3s timeout was often fully consumed.
    # The "not yet retrieved" fallback below is always a safe starting
    # point, since the agent can fetch the data itself via get_customer_info.
    if phone_number:
        system_prompt += (
            f"\n\nThe customer's phone number is {phone_number} but their account "
            f"could not be retrieved automatically. Call get_customer_info with "
            f"{phone_number} before proceeding."
        )
    elif not is_console:
        system_prompt += (
            "\n\nThe customer's phone number is not available. Ask the customer for their "
            "phone number, then call get_customer_info to look up their account."
        )

    # Agent with tools
    agent = Agent(
        instructions=system_prompt,
        tools=[get_customer_info, end_call, get_current_datetime, set_active_language],
    )

    async def _apply_customer_lookup_result() -> None:
        """Runs in the background — doesn't block session start, AMD, or the
        greeting. If the lookup that was kicked off right after connecting
        resolves in time, swaps in the enriched instructions so the model
        doesn't need to call get_customer_info itself; if it doesn't, the
        fallback instructions set above already have the model do that
        itself once the caller starts talking, so nothing breaks either way.
        """
        try:
            customer_data = await asyncio.wait_for(lookup_task, timeout=3)
        except (asyncio.TimeoutError, requests.RequestException) as e:
            print(f"Customer lookup failed or timed out: {e}")
            return
        if not customer_data:
            return
        active_lead = customer_data.get("active_lead") or {}
        enriched_prompt = system_prompt + (
            f"\n\nThe customer has already been identified as "
            f"{active_lead.get('first_name', '')} {active_lead.get('surname', '')} "
            f"(phone: {phone_number}). Full customer data: {customer_data}. "
            f"You do not need to call get_customer_info again unless the details seem wrong."
        )
        await agent.update_instructions(enriched_prompt)

    if lookup_task:
        asyncio.create_task(_apply_customer_lookup_result())

    # Start
    # BVCTelephony is tuned for the narrow, compressed audio band of real
    # phone calls (SIP); plain BVC is tuned for regular WebRTC audio and
    # doesn't filter phone-call background noise well. Only a real SIP call
    # gets BVCTelephony — console and Playground/WebRTC testing keep BVC.
    noise_cancellation_model = (
        noise_cancellation.BVCTelephony() if sip_participant else noise_cancellation.BVC()
    )
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation_model),
    )

    # Background ambient office noise so Shanice sounds like a real call-center
    # agent rather than speaking from a silent void. Was volume=0.3 — likely
    # too quiet to notice under phone-codec compression; LiveKit's own docs
    # examples all use 0.8. Bumped to 0.6 as a middle ground; tune from here
    # (too high will compete with the voice track and hurt STT/turn
    # detection on the caller's side, since it's picked up by the mic too
    # if the ambience were ever played on an open speaker — not a concern
    # here since this only plays into the outbound track, but worth knowing
    # if it starts sounding intrusive).
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    # Max call duration: hard cap at 6 minutes from call start (not from
    # here — it accounts for time already spent on SIP wait/AMD/greeting
    # above, using the same call_started_at as the recording/transcript).
    # Speaks a short closing line first rather than cutting off silently
    # mid-sentence; allow_interruptions=False so the caller talking over it
    # doesn't extend the call past the cap.
    MAX_CALL_DURATION_SECONDS = 6 * 60

    async def _enforce_max_call_duration() -> None:
        remaining = MAX_CALL_DURATION_SECONDS - (time.time() - call_started_at)
        if remaining > 0:
            await asyncio.sleep(remaining)
        if ctx.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            return  # call already ended for some other reason
        try:
            await session.say(
                "We're at the time limit for this call, so I have to go now. Thank you.",
                allow_interruptions=False,
            )
        except Exception as e:
            print(f"Failed to speak max-call-duration closing line: {e}")
        ctx.shutdown(reason="max call duration (6 min) reached")

    max_duration_task = asyncio.create_task(_enforce_max_call_duration())

    # Greeting first, then Answering Machine Detection. Previously AMD ran
    # before anything was said (to avoid talking over a voicemail greeting);
    # this now plays the pre-recorded greeting immediately so the caller
    # hears something right away instead of waiting through AMD's
    # classification window. Tradeoff: on a real voicemail/IVR line, the
    # greeting audio plays before we know that and hang up/navigate — AMD
    # still runs right after and still cuts the call short in those cases,
    # just after the greeting rather than before it.
    #
    # `lookup_task` (the customer snapshot fetch) was already kicked off as
    # soon as the phone number was known, well before this point, so it runs
    # concurrently with the greeting rather than after it.
    if is_console:
        # Console mode has no SIP participant/greeting WAV workflow, and no
        # answering-machine scenario when testing with your own mic — keep
        # the simple TTS line, no AMD.
        result = None
        call_state["amd_category"] = "skipped_console"
        await session.say(
            "Hello, how can I assist you with your account today?",
            allow_interruptions=True,
        )
    else:
        # AMD.__aenter__() pauses the session's normal auto-reply
        # authorization the instant it's entered — that's what keeps a
        # reply to the caller's speech from playing out mid-classification.
        # Entering AMD *before* the WAV greeting (not after) makes that lock
        # cover the greeting too: if the caller says something while
        # Greeting.wav is still playing — very common, most people speak the
        # instant they pick up — a reply can't leak out and overlap it. The
        # WAV greeting itself still bypasses this session's TTS entirely, so
        # from the session's own point of view the agent stays silent
        # through it, then through AMD. `opening_stall_task` covers that
        # whole window: if the agent still hasn't spoken for real 6s after
        # the WAV finishes, it plays a second bridge clip.
        #
        # `wait_until_finished=False` caps detection at the 20s timeout even
        # if the caller talks continuously without a clean pause —
        # otherwise AMD extends indefinitely waiting for silence.
        async with AMD(
            session,
            llm=llm,
            stt=stt,
            suppress_compatibility_warning=True,
            wait_until_finished=False,
        ) as detector:
            await _play_wav_greeting(ctx.room, GREETING_WAV_PATH)
            opening_stall_task = asyncio.create_task(_play_opening_stall_after_delay())
            result = await detector.execute()
        call_state["amd_category"] = result.category.value

        if result.category == AMDCategory.MACHINE_VM:
            opening_stall_task.cancel()
            ctx.shutdown(reason="voicemail detected")
            return
        elif result.category == AMDCategory.MACHINE_UNAVAILABLE:
            opening_stall_task.cancel()
            ctx.shutdown(reason="mailbox unavailable")
            return
        elif result.category == AMDCategory.MACHINE_IVR:
            # AMD's built-in IVR navigation already kicked off inside
            # execute(); nothing further to do here — and no proactive
            # human-facing greeting should play into an automated menu.
            opening_stall_task.cancel()
        else:
            # human/uncertain: greeting was already played above; nothing
            # more to trigger here. AMD._on_end_of_turn only skips the
            # session's normal auto-reply pipeline once it has already
            # decided "machine" — for human/uncertain that pipeline runs as
            # usual, just gated behind AMD's authorization lock, which
            # execute() eagerly releases as soon as a verdict is in ("so
            # agent can speak immediately to a human"). So: if the caller
            # already said something during the WAV greeting or AMD, a
            # reply is already queued and plays the instant that lock
            # lifts — respond ASAP, no extra call needed. If the caller
            # hasn't spoken yet, nothing is queued, and correctly so —
            # opening_stall_task/GREETING_V2_WAV_PATH is the only thing
            # that covers the silence until they do.
            pass

    # No keep-alive loop here on purpose. The call's actual lifetime is
    # governed by the job's internal shutdown future (set by ctx.shutdown()
    # or the room disconnecting on its own) — entirely independent of
    # whether entrypoint() itself has returned. session event listeners and
    # background tasks (max_duration_task, silence checks, etc.) keep
    # running either way. A `while connection_state == CONNECTED: sleep(1)`
    # loop here would actively hurt: the job runner waits (up to 15s) for
    # this task to finish before it disconnects the room on shutdown, and
    # that loop only exits once the room disconnects — a circular wait that
    # stalls every hangup (end_call, max call duration, silence timeout) by
    # up to 15 real seconds for no benefit.


def prewarm(proc):
    """Runs once per worker process (not per call). Loading the Silero VAD
    model here instead of inside entrypoint() means every call reuses the
    already-loaded model instead of paying its load cost before the greeting
    can play."""
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    # STT and LLM run through LiveKit Inference, billed via your LiveKit
    # Cloud account — no DEEPGRAM_API_KEY/GOOGLE_API_KEY/OPENAI_API_KEY
    # needed for those. TTS still needs AZURE_SPEECH_KEY/AZURE_SPEECH_REGION
    # (direct plugin, primary voice) and ELEVENLABS_API_KEY (direct plugin,
    # fallback, to preserve our custom cloned voice — Inference only supports
    # ElevenLabs' default voices). .env.local needs LIVEKIT_URL/
    # LIVEKIT_API_KEY/LIVEKIT_API_SECRET + AZURE_SPEECH_KEY/
    # AZURE_SPEECH_REGION + ELEVENLABS_API_KEY (+ Firebase creds for call
    # recording, unrelated to voice AI billing).
    #
    # agent_name="Shanice" (not "Mary" — that name is already taken by the
    # live production Nigeria agent, va_bot.py, deployed under agent id
    # CA_Qyuo7orfGVYY) is required because the SIP dispatch rule
    # (roomConfig.agents: [{"agentName": "Shanice"}]) uses explicit/named
    # dispatch. Without a matching agent_name here, this worker only
    # registers for automatic dispatch and never receives jobs from that
    # rule — the dispatch name below must match the dispatch rule exactly.
    # NOTE: the SIP dispatch rule itself still needs to be created on
    # LiveKit Cloud for Shanice (mirroring however Mary's was set up) —
    # this WorkerOptions change alone doesn't create it.
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="Shanice")
    )