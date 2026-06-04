"""
metrics.py — Real-time metric computation from the events table.

Handles edge cases:
  - Empty store  → conversion_rate=None, data_confidence="low"
  - All-staff    → unique_visitors=0, conversion_rate=None
  - Zero purchases → conversion_rate=0.0 (not None)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, distinct
from sqlalchemy.orm import Session

from db import EventRow, SessionRow
from models import MetricsResponse


STALE_CONFIDENCE_HOURS = 2   # if no events in last N hours → data_confidence="low"


def get_metrics(
    store_id:     str,
    db:           Session,
    period_start: Optional[datetime] = None,
    period_end:   Optional[datetime] = None,
) -> MetricsResponse:
    now = datetime.now(timezone.utc)
    period_end   = period_end   or now
    period_start = period_start or (now - timedelta(hours=24))

    # ── Base filter ───────────────────────────────────────────────────────────
    base_q = (
        db.query(EventRow)
          .filter(
              EventRow.store_id  == store_id,
              EventRow.timestamp >= period_start,
              EventRow.timestamp <= period_end,
              EventRow.is_staff  == False,
          )
    )

    # ── Unique visitors (non-staff ENTRY events only) ─────────────────────────
    unique_visitors = (
        base_q
        .filter(EventRow.event_type == "ENTRY")
        .with_entities(func.count(distinct(EventRow.visitor_id)))
        .scalar()
    ) or 0

    # ── Empty / all-staff guard ───────────────────────────────────────────────
    if unique_visitors == 0:
        # Check whether there are ANY events (including staff) in this period.
        any_events = (
            db.query(func.count(EventRow.id))
              .filter(
                  EventRow.store_id  == store_id,
                  EventRow.timestamp >= period_start,
                  EventRow.timestamp <= period_end,
              )
              .scalar()
        ) or 0

        if any_events > 0:
            # Events exist but ALL are staff — no customer data → always low
            confidence = "low"
        else:
            # Truly empty: check how stale the last event is
            last_event_ts = (
                db.query(func.max(EventRow.timestamp))
                  .filter(EventRow.store_id == store_id)
                  .scalar()
            )
            confidence = _data_confidence(last_event_ts, now)

        return MetricsResponse(
            store_id=store_id,
            period_start=period_start,
            period_end=period_end,
            unique_visitors=0,
            conversion_rate=None,
            avg_dwell_minutes=None,
            peak_hour=None,
            total_zone_visits=0,
            data_confidence=confidence,
        )

    # ── Conversions (sessions with converted=True) ───────────────────────────
    converted_count = (
        db.query(func.count(distinct(SessionRow.visitor_id)))
          .filter(
              SessionRow.store_id   == store_id,
              SessionRow.entry_time >= period_start,
              SessionRow.entry_time <= period_end,
              SessionRow.converted  == True,
              SessionRow.is_staff   == False,
          )
          .scalar()
    ) or 0

    conversion_rate = round(converted_count / unique_visitors, 4)

    # ── Avg dwell ─────────────────────────────────────────────────────────────
    avg_dwell_s = (
        db.query(func.avg(SessionRow.dwell_seconds))
          .filter(
              SessionRow.store_id   == store_id,
              SessionRow.entry_time >= period_start,
              SessionRow.entry_time <= period_end,
              SessionRow.is_staff   == False,
              SessionRow.dwell_seconds.isnot(None),
          )
          .scalar()
    )
    avg_dwell_minutes = round(avg_dwell_s / 60, 1) if avg_dwell_s else None

    # ── Peak hour ─────────────────────────────────────────────────────────────
    peak_row = (
        base_q
        .filter(EventRow.event_type == "ENTRY")
        .with_entities(
            func.strftime("%H", EventRow.timestamp).label("hr"),
            func.count().label("cnt"),
        )
        .group_by("hr")
        .order_by(func.count().desc())
        .first()
    )
    peak_hour = int(peak_row.hr) if peak_row else None

    # ── Zone visits ───────────────────────────────────────────────────────────
    total_zone_visits = (
        base_q
        .filter(EventRow.event_type == "ZONE_ENTER")
        .count()
    )

    # ── Data confidence ───────────────────────────────────────────────────────
    last_ts = (
        base_q.with_entities(func.max(EventRow.timestamp)).scalar()
    )
    confidence = _data_confidence(last_ts, now)

    return MetricsResponse(
        store_id=store_id,
        period_start=period_start,
        period_end=period_end,
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_minutes=avg_dwell_minutes,
        peak_hour=peak_hour,
        total_zone_visits=total_zone_visits,
        data_confidence=confidence,
    )


def _data_confidence(last_ts: Optional[datetime], now: datetime) -> str:
    if last_ts is None:
        return "low"
    last_ts = last_ts.replace(tzinfo=timezone.utc) if last_ts.tzinfo is None else last_ts
    age_hours = (now - last_ts).total_seconds() / 3600
    if age_hours > STALE_CONFIDENCE_HOURS:
        return "low"
    if age_hours > 0.5:
        return "medium"
    return "high"
