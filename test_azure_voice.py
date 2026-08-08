#!/usr/bin/env python3
"""Standalone preview of Azure's en-NG neural voices. Not part of the bot pipeline.

Usage:
    uv run python test_azure_voice.py
Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env.local.
"""
import os

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv(".env.local")

SAMPLE_TEXT = (
    "Hello, this is Kolawole calling regarding your account. "
    "Do you have a moment to discuss your outstanding balance?"
)

VOICES = ["en-NG-EzinneNeural", "en-NG-AbeoNeural"]


def main():
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise SystemExit("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env.local first.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)

    for voice in VOICES:
        speech_config.speech_synthesis_voice_name = voice
        out_path = f"preview_{voice}.wav"
        audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        result = synthesizer.speak_text_async(SAMPLE_TEXT).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Saved {out_path}")
        else:
            details = result.cancellation_details
            print(f"Failed for {voice}: {details.reason} — {details.error_details}")


if __name__ == "__main__":
    main()
