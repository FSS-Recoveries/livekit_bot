#!/usr/bin/env python3
import asyncio
import os
import re
import requests
import inspect
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import deepgram, openai, silero, elevenlabs

# ── Quick compat shim: Mp3StreamDecoder → AudioStreamDecoder ─────────────
# Some livekit-agents versions removed Mp3StreamDecoder in favor of AudioStreamDecoder.
# If the ElevenLabs plugin tries to use Mp3StreamDecoder, provide a lightweight alias.
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
# .env.local should contain: ELEVENLABS_API_KEY=your_real_key
load_dotenv(".env.local")

# Map ELEVENLABS_API_KEY -> ELEVEN_API_KEY for plugin compatibility
if not os.getenv("ELEVEN_API_KEY") and os.getenv("ELEVENLABS_API_KEY"):
    os.environ["ELEVEN_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")

# ── Function tools ──────────────────────────────────────────────────────
def _normalize_nigerian_phone(raw: str) -> str:
    """Strip non-digits, then convert 234XXXXXXXXXX → 0XXXXXXXXX."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("234"):
        digits = "0" + digits[3:]
    return digits


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
    url = f"https://fss-api.onrender.com/customer_snapshot/{normalized}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        first_name = data.get("first_name", "")
        last_name = data.get("surname", "")
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


@function_tool(description="Schedule a payment-plan link or instructions to the provided email.")
async def schedule_payment_plan(ctx: RunContext, email: str, name: str, amount: float) -> str:
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return "The email address appears invalid."
    webhook = os.getenv("PAYMENT_PLAN_WEBHOOK_URL")
    if not webhook:
        return "Payment plan service unavailable."
    try:
        resp = requests.post(
            webhook,
            json={"email": email, "name": name, "amount": amount},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return f"A payment plan has been scheduled. Check your email at {email}."
    except requests.RequestException:
        return "Error scheduling the payment plan; please try again later."


@function_tool(description="Check account status by email and prompt for payment plan if delinquent.")
async def check_account_status(ctx: RunContext, email: str) -> str:
    token = os.getenv("API_TOKEN")
    url = os.getenv("ACCOUNT_STATUS_ENDPOINT")
    try:
        resp = requests.get(
            f"{url}?email={email}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "delinquent":
            return "Your account is delinquent. Would you like to schedule a payment plan?"
        return "Your account is in good standing."
    except requests.RequestException:
        return "Error checking account status; please try again later."


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


# ── Entrypoint ──────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room {ctx.room.name}")

    # Load system prompt (fallback if file missing)
    prompt_path = os.path.join(os.getcwd(), "prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "You are Kolawole, a debt collection assistant."

    # Providers
    stt = deepgram.STT(model="nova-3")
    llm = openai.LLM(model="gpt-4o", temperature=0.1)
    vad = silero.VAD.load()
    tts = build_elevenlabs_tts()

    # Session
    session = AgentSession(
        stt=stt,
        llm=llm,
        vad=vad,
        tts=tts,
    )

    # Agent with tools
    agent = Agent(
        instructions=system_prompt,
        tools=[get_customer_info, schedule_payment_plan, check_account_status],
    )

    # Start
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=None),
    )

    await session.say(
        "Hello, how can I assist you with your account today?",
        allow_interruptions=True,
    )

    # Keep alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Requires ELEVENLABS_API_KEY in .env.local (auto-mapped to ELEVEN_API_KEY)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
