#!/usr/bin/env python3
import asyncio
import os
import re
import requests
import inspect
import time
import uuid
import wave
from datetime import datetime, timezone
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
        encoded_path = filepath.replace("/", "%2F")
        return (
            f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_STORAGE_BUCKET}"
            f"/o/{encoded_path}?alt=media&token={token}"
        )
    except Exception as e:
        print(f"Failed to finalize call recording URL: {e}")
        return None


# ⚠️ STALE AS OF THE INFERENCE SWITCH: every rate below was the direct
# provider's own API price (OpenAI/Google/ElevenLabs/Deepgram billing you
# directly). Now that stt/llm/tts all go through LiveKit Inference
# (livekit.agents.inference), you're billed at LiveKit's own Inference rate
# card instead — which is not confirmed to be identical to these numbers —
# and the `provider`/`model` strings these usage entries actually contain
# once routed through the gateway haven't been re-verified either (matching
# below may now silently return None for everything, or match against the
# wrong rate). Check LiveKit Cloud's own usage/billing dashboard for real
# figures; treat this function as unverified until checked against a real
# post-switch usage entry and LiveKit's published Inference pricing.
#
# Approximate USD rates per provider/model, from each provider's public
# pricing page as of 2026-08-09. These are ESTIMATES for internal cost
# tracking only — always reconcile against actual provider invoices, since
# rates change and this doesn't account for volume/commitment discounts.
# Azure isn't priced here: its plugin doesn't report model_provider/model_name
# metadata the way the others do, so it hasn't been confirmed against a real
# usage entry to know the exact keys it would use (even though it's now the
# primary TTS voice — see build_azure_tts).
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
_GPT5_MINI_CACHED_INPUT_PER_TOKEN = 0.025 / 1_000_000
_GPT5_MINI_OUTPUT_PER_TOKEN = 2.00 / 1_000_000
_ELEVENLABS_FLASH_PER_CHARACTER = 0.05 / 1_000
_DEEPGRAM_NOVA3_PER_SECOND = 0.0077 / 60


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
    if provider == "Deepgram" and model == "nova-3":
        return entry.get("audio_duration", 0) * _DEEPGRAM_NOVA3_PER_SECOND
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
            }
        )
    except Exception as e:
        print(f"Failed to save call record to Firestore: {e}")


# ── Function tools ──────────────────────────────────────────────────────
def _normalize_nigerian_phone(raw: str) -> str:
    """Strip non-digits, then convert 234XXXXXXXXXX → 0XXXXXXXXX."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("234"):
        digits = "0" + digits[3:]
    return digits


async def _fetch_customer_snapshot(phone_number: str) -> dict:
    """Fetches the customer snapshot for a (Nigerian-normalized) phone number.

    Raises requests.HTTPError/RequestException on failure — callers decide
    how to handle that. The blocking `requests` call is offloaded to a
    thread so this can run concurrently with other async work (e.g. the
    greeting WAV playing, or provider setup).
    """
    normalized = _normalize_nigerian_phone(phone_number)
    url = f"https://fss-api.onrender.com/customer_snapshot/{normalized}"
    resp = await asyncio.to_thread(requests.get, url, timeout=15)
    resp.raise_for_status()
    return resp.json()


@function_tool(
    description=(
        "Get customer info from the API endpoint once the caller's phone number is available. "
        "Pass the phone number in Nigerian local format (0XXXXXXXXX): remove spaces/special "
        "characters and replace a leading 234 country code with 0. "
        "Example: 2348134073764 → 08134073764."
    )
)
async def get_customer_info(ctx: RunContext, phone_number: str) -> str:
    normalized = _normalize_nigerian_phone(phone_number)
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
        "End the call and hang up. Call this whenever the prompt instructs you to "
        "'stop completely' — after your final spoken message, a dead-end exit, or an "
        "escalation close. Always speak your closing message first; this tool waits "
        "for it to finish playing before disconnecting, so it is safe to call "
        "immediately after your final message."
    )
)
async def end_call(ctx: RunContext) -> None:
    await ctx.wait_for_playout()
    get_job_context().shutdown(reason="call ended by agent")


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
    voice_id  = "me1JPr2K6H7KZB9nz2Wk"#os.getenv("ELEVENLABS_VOICE_ID", "RAVWJW17BPoSIf05iXxf")  # example default
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
    for opt_key, opt_val in {
        "language": "en",
        "streaming_latency": 3,
        "enable_ssml_parsing": True,
        # some versions also support inactivity_timeout, chunk_length_schedule, etc.
    }.items():
        if opt_key in tts_params:
            tts_kwargs[opt_key] = opt_val

    return elevenlabs.TTS(**tts_kwargs)


# ── Azure TTS builder ────────────────────────────────────────────────────
def build_azure_tts():
    """
    Builds the primary Azure neural TTS voice (en-NG-EzinneNeural by default).
    AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are picked up automatically from
    the environment by azure.TTS itself when not passed explicitly.
    """
    voice = os.getenv("AZURE_TTS_VOICE", "en-NG-EzinneNeural")
    return azure.TTS(voice=voice)


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
    egress_task = asyncio.create_task(_start_call_recording(ctx))
    max_duration_task: asyncio.Task | None = None

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
    prompt_path = os.path.join(os.getcwd(), "prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "You are Kolawole, a debt collection assistant."

    # Providers — STT and LLM via LiveKit Inference (livekit.agents.inference),
    # billed through your LiveKit Cloud account: no DEEPGRAM_API_KEY,
    # GOOGLE_API_KEY, OPENAI_API_KEY needed for those two.
    #
    # TTS is the exception (see below): it uses direct Azure and ElevenLabs
    # plugins instead of Inference, so AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
    # and ELEVENLABS_API_KEY are still required for that component.
    #
    # STT: ElevenLabs Scribe v2 Realtime as primary, Deepgram as fallback if
    # ElevenLabs errors out mid-call.
    stt = agents_stt.FallbackAdapter(
        [
            inference.STT("elevenlabs/scribe_v2_realtime"),
            inference.STT("deepgram/nova-3", language="multi"),
        ]
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
    llm = agents_llm.FallbackAdapter(
        [
            inference.LLM("google/gemini-3.1-flash-lite"),
            inference.LLM("openai/gpt-5-mini"),
        ]
    )
    # Silero VAD runs locally, not through any external provider — it never
    # needed an API key, so it's left as-is (still prewarmed per worker
    # process in prewarm(), not per call).
    vad = ctx.proc.userdata["vad"]

    # TTS: Fish Audio S2.1 Pro (our own custom voice) as primary, via LiveKit
    # Inference — no separate API key needed, billed through LiveKit Cloud.
    #
    # build_azure_tts() and build_elevenlabs_tts() are kept defined below but
    # commented out here — swap either back in as the 2nd list entry to
    # restore it as a fallback. Note there's currently no active fallback:
    # if Fish Audio errors out mid-call, TTS has nothing to fall back to.
    tts = agents_tts.FallbackAdapter(
        [
            inference.TTS(
                model="fishaudio/s2-pro",
                voice="v_ebJJAf8QhLMs",#"v_tkbNkcSD62zN",#"",
                extra_kwargs={"speed": 1.2, "temperature": 0, "latency": "low"},
            ),
            #build_azure_tts(),
            #build_elevenlabs_tts(),
        ]
    )



    # Usage metrics (LLM tokens, TTS characters, STT audio seconds), broken
    # down per provider/model — persisted to the call's Firestore record.
    usage_collector = ModelUsageCollector()

    # Session
    session = AgentSession(
        stt=stt,
        llm=llm,
        vad=vad,
        tts=tts,
        turn_detection=TurnDetector(),
        preemptive_generation=True,
    )

    # session.emit("metrics_collected", MetricsCollectedEvent(metrics=...)) wraps
    # the metrics — ModelUsageCollector.collect() expects the raw metrics object,
    # so it must be unwrapped here or every call silently no-ops.
    session.on("metrics_collected", lambda ev: usage_collector.collect(ev.metrics))

    def _on_conversation_item_added(ev):
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role and text:
            transcript_lines.append(f"{role}: {text}")

    session.on("conversation_item_added", _on_conversation_item_added)

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
        tools=[get_customer_info, end_call],
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

    # Background ambient office noise so Fola sounds like a real call-center
    # agent rather than speaking from a silent void. Was volume=0.3 — likely
    # too quiet to notice under phone-codec compression; LiveKit's own docs
    # examples all use 0.8. Bumped to 0.6 as a middle ground; tune from here
    # (too high will compete with the voice track and hurt STT/turn
    # detection on the caller's side, since it's picked up by the mic too
    # if the ambience were ever played on an open speaker — not a concern
    # here since this only plays into the outbound track, but worth knowing
    # if it starts sounding intrusive).
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.CROWDED_ROOM, volume=0.4),
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
        # Nothing is said after this — the LLM only fires once STT/VAD
        # detects the caller's reply, so the agent naturally stays silent
        # until they actually respond, then follows the prompt's own
        # Opening Flow script via normal TTS.
        await _play_wav_greeting(ctx.room, GREETING_WAV_PATH)

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
            result = await detector.execute()
        call_state["amd_category"] = result.category.value

        if result.category == AMDCategory.MACHINE_VM:
            ctx.shutdown(reason="voicemail detected")
            return
        elif result.category == AMDCategory.MACHINE_UNAVAILABLE:
            ctx.shutdown(reason="mailbox unavailable")
            return
        elif result.category == AMDCategory.MACHINE_IVR:
            # AMD's built-in IVR navigation already kicked off inside
            # execute(); nothing further to do here.
            pass
        # human/uncertain: greeting was already played above.

    # Silence safety net: the LLM only reacts to turns, so if the customer
    # goes quiet it can't act on its own. After user_away_timeout (15s
    # default) of no user activity, check in once per the prompt's silence
    # rule; if they're still away after a second consecutive check, hang up
    # instead of holding the line open indefinitely.
    checked_in = False

    def _on_user_state_changed(ev):
        nonlocal checked_in
        if ev.new_state != "away":
            checked_in = False
            return
        if not checked_in:
            checked_in = True
            asyncio.create_task(
                session.say("Are you still with me?", allow_interruptions=True)
            )
        else:
            ctx.shutdown(reason="no response after silence check")

    session.on("user_state_changed", _on_user_state_changed)

    # Keep alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)


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
    # agent_name="fola" is required because the SIP dispatch rule
    # (roomConfig.agents: [{"agentName": "fola"}]) uses explicit/named
    # dispatch. Without a matching agent_name here, this worker only
    # registers for automatic dispatch and never receives jobs from that
    # rule — the dispatch name below must match the dispatch rule exactly.
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="fola")
    )