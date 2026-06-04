"""
health.py — /health endpoint.

Returns per-store feed status and overall system health.
  OK          — last event < 10 minutes ago
  STALE_FEED  — last event >= 10 minutes ago
  NO_DATA     — no events at all for this store

HTTP 503 is returned (by main.py) if the database is unreachable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from db import EventRow
from models import HealthResponse, StoreHealth


STALE_THRESHOLD_MINUTES = 10


def get_health(db: DBSession, store_ids: Optional[list[str]] = None) -> HealthResponse:
    now = datetime.now(timezone.utc)

    if store_ids is None:
        # Discover all stores that have ever sent events
        rows = db.query(func.distinct(EventRow.store_id)).all()
        store_ids = [r[0] for r in rows]

    store_healths: list[StoreHealth] = []

    for store_id in store_ids:
        last_ts = (
            db.query(func.max(EventRow.timestamp))
              .filter(EventRow.store_id == store_id)
              .scalar()
        )

        if last_ts is None:
            store_healths.append(StoreHealth(
                store_id=store_id,
                status="NO_DATA",
                last_event_time=None,
                lag_seconds=None,
            ))
            continue

        # Normalise timezone
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        lag = (now - last_ts).total_seconds()
        status = "STALE_FEED" if lag > STALE_THRESHOLD_MINUTES * 60 else "OK"

        store_healths.append(StoreHealth(
            store_id=store_id,
            status=status,
            last_event_time=last_ts,
            lag_seconds=round(lag, 1),
        ))

    overall = "OK"
    if any(s.status == "STALE_FEED" for s in store_healths):
        overall = "DEGRADED"
    if not store_healths:
        overall = "NO_DATA"

    return HealthResponse(
        status=overall,
        stores=store_healths,
        checked_at=now,
    )
