"""
ingestion.py — Ingest events with idempotency, partial-success semantics,
               and WebSocket broadcast after every successful commit.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from db import EventRow
from models import IngestRequest, IngestResponse


def _dict_to_row(raw: dict[str, Any]) -> EventRow:
    ts_raw = raw.get("timestamp")
    if isinstance(ts_raw, str):
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        raise ValueError(f"Missing or invalid timestamp in event {raw.get('event_id')}")

    dwell = raw.get("dwell_seconds") or raw.get("dwell_secs")

    return EventRow(
        event_id    = raw["event_id"],
        event_type  = raw.get("event_type", "UNKNOWN"),
        visitor_id  = raw["visitor_id"],
        store_id    = raw["store_id"],
        camera_id   = raw.get("camera_id"),
        timestamp   = ts,
        is_staff    = bool(raw.get("is_staff", False)),
        confidence  = float(raw.get("confidence", 1.0)),
        zone_id     = raw.get("zone_id"),
        zone_name   = raw.get("zone_name"),
        queue_depth = raw.get("queue_depth"),
        dwell_secs  = dwell,
        raw_json    = json.dumps(raw),
    )


def ingest_events(payload: IngestRequest, db: Session) -> IngestResponse:
    ingested  = 0
    duplicate = 0
    errors: list[str] = []

    # Group successfully ingested raw events by store_id for broadcast
    ingested_by_store: dict[str, list[dict]] = {}

    for raw in payload.events:
        try:
            event_id = raw.get("event_id")
            if not event_id:
                errors.append(f"Missing event_id in event: {raw}")
                continue

            exists = db.query(EventRow.id).filter_by(event_id=event_id).first()
            if exists:
                duplicate += 1
                continue

            row = _dict_to_row(raw)
            db.add(row)
            db.flush()
            ingested += 1

            store_id = raw.get("store_id", "unknown")
            ingested_by_store.setdefault(store_id, []).append(raw)

        except IntegrityError:
            db.rollback()
            duplicate += 1
        except (KeyError, ValueError) as exc:
            db.rollback()
            errors.append(str(exc))

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        errors.append(f"Commit failed: {exc}")
        return IngestResponse(ingested=ingested, duplicate=duplicate, errors=errors)

    # ── Broadcast to WebSocket clients after successful commit ─────────────────
    # Import here to avoid circular import at module load time
    if ingested > 0:
        try:
            from ws_manager import manager
            loop = asyncio.get_event_loop()
            if loop.is_running():
                for store_id, events in ingested_by_store.items():
                    asyncio.ensure_future(manager.broadcast(store_id, events))
        except RuntimeError:
            pass   # no event loop in test context — skip broadcast silently

    return IngestResponse(ingested=ingested, duplicate=duplicate, errors=errors)
