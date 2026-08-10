#!/usr/bin/env python3
"""Fetch exact session/room billing details from LiveKit's Analytics API.

Uses the same LIVEKIT_API_KEY / LIVEKIT_API_SECRET already in .env.local.
One new value is needed that you likely don't have yet: LIVEKIT_PROJECT_ID
— the "p_..." string in your dashboard URL, e.g.
https://cloud.livekit.io/projects/p_abc123/... -> LIVEKIT_PROJECT_ID=p_abc123
Add it to .env.local alongside the others.

Note: the Analytics API is only available on LiveKit Cloud Scale plan or
higher. If you're on Build/Ship, use the dashboard's Sessions tab instead
(search for the room ID directly).
"""
import os
import sys
from datetime import timedelta

import requests
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv(".env.local")

LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
PROJECT_ID = os.environ["LIVEKIT_PROJECT_ID"]


def get_session_details(room_id: str) -> dict:
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity("cost-lookup-script")
        .with_grants(VideoGrants(room_list=True))
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )

    resp = requests.get(
        f"https://cloud-api.livekit.io/api/project/{PROJECT_ID}/sessions/{room_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python get_session_cost.py <ROOM_ID>")
        print("Example: python get_session_cost.py RM_o4qeUZCBDopa")
        sys.exit(1)

    room_id = sys.argv[1]

    try:
        details = get_session_details(room_id)
    except requests.HTTPError as e:
        if e.response is not None and "scale plan" in e.response.text.lower():
            print(
                f"{e.response.status_code} — the Analytics API requires a "
                "Scale plan or higher. Use the dashboard's Sessions tab "
                "instead."
            )
        elif e.response is not None and e.response.status_code == 404:
            print(f"No session found for room ID {room_id}.")
        else:
            print(f"HTTP error: {e}")
        sys.exit(1)

    connection_minutes = details.get("connectionMinutes")

    print(f"Room ID:              {details.get('roomId', room_id)}")
    print(f"Start:                {details.get('startTime')}")
    print(f"End:                  {details.get('endTime')}")
    print(f"Participants:         {details.get('numParticipants')}")
    print(f"Billable connection minutes: {connection_minutes}")

    if connection_minutes is not None:
        agent_session_cost = connection_minutes * 0.01
        telephony_cost = connection_minutes * 0.004
        print()
        print(f"Agent session cost (${'0.01'}/min):  ${agent_session_cost:.4f}")
        print(f"Telephony cost     (${'0.004'}/min): ${telephony_cost:.4f}")
        print(
            "(Telephony rate assumes Build/Ship third-party SIP; adjust if "
            "you're on a different plan/number type.)"
        )
        print(
            "Note: this does NOT include LLM/STT/TTS inference cost — pull "
            "that from the usage payload logged by ModelUsageCollector for "
            "the same call instead."
        )


if __name__ == "__main__":
    main()