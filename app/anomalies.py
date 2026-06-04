"""
anomalies.py — Three anomaly detectors.

  BILLING_QUEUE_SPIKE  — queue depth > threshold for > 5 consecutive minutes
  CONVERSION_DROP      — today's hourly conversion rate is > 1 stddev below 7-day avg
  DEAD_ZONE            — no zone visits in 30 min during open hours

Each anomaly includes severity and suggested_action.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from db import EventRow
from models import AnomalyItem, AnomaliesResponse


# ─── Config ───────────────────────────────────────────────────────────────────

QUEUE_DEPTH_THRESHOLD   = 8     # people
QUEUE_SPIKE_DURATION_M  = 5     # minutes sustained to trigger
DEAD_ZONE_WINDOW_M      = 30    # minutes of silence before flagging
# Brigade Road Bangalore: open 11:00-22:00 IST
STORE_OPEN_HOUR         = int(os.getenv("STORE_OPEN_HOUR",  11))
STORE_CLOSE_HOUR        = int(os.getenv("STORE_CLOSE_HOUR", 22))
CONVERSION_LOOKBACK_D   = 7     # days for baseline
CONVERSION_STDDEV_MULT  = 1.0   # how many stddevs below to trigger WARN


# ─── Detector 1: Billing queue spike ─────────────────────────────────────────

def detect_queue_spike(store_id: str, db: DBSession, now: datetime) -> list[AnomalyItem]:
    window_start = now - timedelta(minutes=QUEUE_SPIKE_DURATION_M + 5)

    rows = (
        db.query(EventRow)
          .filter(
              EventRow.store_id   == store_id,
              EventRow.event_type == "BILLING_QUEUE",
              EventRow.timestamp  >= window_start,
              EventRow.timestamp  <= now,
              EventRow.queue_depth.isnot(None),
          )
          .order_by(EventRow.timestamp)
          .all()
    )

    if not rows:
        return []

    # Check if queue_depth > threshold for the last QUEUE_SPIKE_DURATION_M minutes
    cutoff = now - timedelta(minutes=QUEUE_SPIKE_DURATION_M)
    recent = [r for r in rows if r.timestamp >= cutoff]

    if not recent:
        return []

    max_depth = max(r.queue_depth for r in recent)
    all_above = all(r.queue_depth >= QUEUE_DEPTH_THRESHOLD for r in recent)

    if not all_above or len(recent) < 2:
        return []

    severity = "CRITICAL" if max_depth >= QUEUE_DEPTH_THRESHOLD * 2 else "WARN"

    return [AnomalyItem(
        anomaly_type="BILLING_QUEUE_SPIKE",
        severity=severity,
        store_id=store_id,
        detected_at=now,
        description=(
            f"Billing queue depth has been ≥ {QUEUE_DEPTH_THRESHOLD} "
            f"for the past {QUEUE_SPIKE_DURATION_M} minutes "
            f"(current max: {max_depth})."
        ),
        suggested_action="Open an additional checkout lane immediately.",
        metadata={"max_depth": max_depth, "sustained_minutes": QUEUE_SPIKE_DURATION_M},
    )]


# ─── Detector 2: Conversion drop ─────────────────────────────────────────────

def detect_conversion_drop(store_id: str, db: DBSession, now: datetime) -> list[AnomalyItem]:
    """
    Compare today's rolling hourly conversion rate against 7-day baseline.
    Requires SessionRow data — uses EventRow ENTRY counts as a proxy here.
    """
    # Today's conversion rate for the last complete hour
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    hour_end   = now

    today_entries = (
        db.query(func.count(func.distinct(EventRow.visitor_id)))
          .filter(
              EventRow.store_id   == store_id,
              EventRow.event_type == "ENTRY",
              EventRow.is_staff   == False,
              EventRow.timestamp  >= hour_start,
              EventRow.timestamp  <= hour_end,
          )
          .scalar()
    ) or 0

    if today_entries == 0:
        return []   # no data to compare

    # 7-day baseline: same hour-of-day for last 7 days
    target_hour = now.hour
    baseline_rates: list[float] = []

    for days_back in range(1, CONVERSION_LOOKBACK_D + 1):
        base_start = (now - timedelta(days=days_back)).replace(
            hour=target_hour, minute=0, second=0, microsecond=0
        )
        base_end = base_start + timedelta(hours=1)

        entries = (
            db.query(func.count(func.distinct(EventRow.visitor_id)))
              .filter(
                  EventRow.store_id   == store_id,
                  EventRow.event_type == "ENTRY",
                  EventRow.is_staff   == False,
                  EventRow.timestamp  >= base_start,
                  EventRow.timestamp  <= base_end,
              )
              .scalar()
        ) or 0

        if entries > 0:
            # Use entry count as proxy; replace with real conversion rate if sessions table is ready
            baseline_rates.append(float(entries))

    if len(baseline_rates) < 3:
        return []   # not enough history

    avg_baseline = mean(baseline_rates)
    std_baseline = stdev(baseline_rates) if len(baseline_rates) > 1 else 0.0
    threshold    = avg_baseline - CONVERSION_STDDEV_MULT * std_baseline

    if float(today_entries) < threshold:
        drop_pct = round((1 - today_entries / avg_baseline) * 100, 1) if avg_baseline > 0 else 0
        severity = "CRITICAL" if drop_pct >= 30 else "WARN"
        return [AnomalyItem(
            anomaly_type="CONVERSION_DROP",
            severity=severity,
            store_id=store_id,
            detected_at=now,
            description=(
                f"Visitor traffic at {now.hour:02d}:00 is {drop_pct}% below the "
                f"7-day average for this hour ({avg_baseline:.0f} avg vs {today_entries} today)."
            ),
            suggested_action=(
                "Check for floor-level issues (missing promos, stock gaps) "
                "or review camera feed for detection errors."
            ),
            metadata={
                "today_count": today_entries,
                "baseline_avg": round(avg_baseline, 1),
                "baseline_std": round(std_baseline, 1),
                "drop_pct": drop_pct,
            },
        )]

    return []


# ─── Detector 3: Dead zone ────────────────────────────────────────────────────

def detect_dead_zones(store_id: str, db: DBSession, now: datetime) -> list[AnomalyItem]:
    """Flag any named zone with no ZONE_ENTER events in the last 30 minutes during open hours."""
    if not (STORE_OPEN_HOUR <= now.hour < STORE_CLOSE_HOUR):
        return []

    window_start = now - timedelta(minutes=DEAD_ZONE_WINDOW_M)

    # All zones that have had ANY activity today (so we know which zones exist)
    active_today = (
        db.query(func.distinct(EventRow.zone_id))
          .filter(
              EventRow.store_id   == store_id,
              EventRow.event_type == "ZONE_ENTER",
              EventRow.zone_id.isnot(None),
              EventRow.timestamp  >= now - timedelta(hours=12),
          )
          .all()
    )
    known_zones = {row[0] for row in active_today}

    if not known_zones:
        return []

    # Zones active in the last 30 min
    recently_active = (
        db.query(func.distinct(EventRow.zone_id))
          .filter(
              EventRow.store_id   == store_id,
              EventRow.event_type == "ZONE_ENTER",
              EventRow.zone_id.isnot(None),
              EventRow.timestamp  >= window_start,
          )
          .all()
    )
    recently_active_ids = {row[0] for row in recently_active}

    dead = known_zones - recently_active_ids
    anomalies: list[AnomalyItem] = []

    for zone_id in dead:
        # Get zone_name from last known event
        last_evt = (
            db.query(EventRow.zone_name)
              .filter(EventRow.zone_id == zone_id)
              .order_by(EventRow.timestamp.desc())
              .first()
        )
        zone_name = last_evt.zone_name if last_evt else zone_id

        anomalies.append(AnomalyItem(
            anomaly_type="DEAD_ZONE",
            severity="INFO",
            store_id=store_id,
            detected_at=now,
            description=(
                f"Zone '{zone_name}' has had no visitor activity in the last "
                f"{DEAD_ZONE_WINDOW_M} minutes during store open hours."
            ),
            suggested_action=(
                f"Check if {zone_name} area is blocked, understocked, or "
                "has a camera/tracking issue."
            ),
            metadata={"zone_id": zone_id, "zone_name": zone_name, "window_minutes": DEAD_ZONE_WINDOW_M},
        ))

    return anomalies


# ─── Combined endpoint ────────────────────────────────────────────────────────

def get_anomalies(store_id: str, db: DBSession) -> AnomaliesResponse:
    # Use naive UTC datetime for all DB comparisons.
    # SQLite stores timestamps without timezone info; comparing an aware datetime
    # against a naive DB column silently returns no rows, breaking every detector.
    now = datetime.utcnow()
    anomalies: list[AnomalyItem] = []

    anomalies += detect_queue_spike(store_id, db, now)
    anomalies += detect_conversion_drop(store_id, db, now)
    anomalies += detect_dead_zones(store_id, db, now)

    # Sort: CRITICAL first, then WARN, then INFO
    order = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
    anomalies.sort(key=lambda a: order.get(a.severity, 9))

    return AnomaliesResponse(store_id=store_id, anomalies=anomalies)
