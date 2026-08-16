#!/usr/bin/env python3
"""One-off driver for the full historical backfill — runs every call in
live_kit_bot_calls (oldest first) through livekit_pipeline.run().

NOTE: pipeline_processed is NOT a reliable "already done" signal for this
one-time historical backfill — a separate, earlier normalization step
pre-flagged pipeline_processed=True on ~1600 historical docs that were never
actually run through this pipeline (no matching calls_va_answered doc, no
institution lookup, no call_notes, no alert email ever sent for them). Using
fetch_unprocessed_calls() to resume would silently skip all of those.

Instead, "genuinely done" == has a matching calls_va_answered doc (that
collection is only ever written by this pipeline's run(), unconditionally,
for every call it actually processes). This script fetches every call
sorted oldest-first (preserving the ordering call_notes_latest depends on)
and filters out only calls that already have that marker."""
from datetime import datetime, timezone

import livekit_pipeline as p

if __name__ == "__main__":
    db = p._fs()

    calls = p.fetch_all_calls_sorted(db)
    done_ids = {
        d.id
        for d in db.collection(p.RAW_LOG_COLLECTION).select([]).stream()
    }

    remaining = [
        c for c in calls
        if (c.get("room_name") or c["_doc_id"]) not in done_ids
    ]
    p.log.info(
        "%d of %d calls already genuinely processed (have a %s doc) — %d remaining",
        len(calls) - len(remaining), len(calls), p.RAW_LOG_COLLECTION, len(remaining),
    )

    p.run(remaining)
