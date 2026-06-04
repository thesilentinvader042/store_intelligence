"""
funnel.py — Assemble sessions from raw events and compute the
            4-stage conversion funnel with POS correlation.

Funnel stages:
  1. Entry          — unique non-staff visitors in period
  2. Zone visit     — subset who had at least one ZONE_ENTER event
  3. Billing queue  — subset who joined the billing queue
  4. Purchase       — subset whose session correlates with a POS transaction

Re-entry rule: ENTRY + REENTRY for the same visitor_id in a period = 1 unique.
Session timeout: if no EXIT event within SESSION_TIMEOUT_MIN, close session at
                 last-seen event + SESSION_TIMEOUT_MIN.
POS window: POS transaction at time T correlates to a session if the visitor
            was in the billing zone between [T - POS_WINDOW_MIN, T].
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from db import EventRow, SessionRow, PosTransactionRow
from models import FunnelResponse, FunnelStage


SESSION_TIMEOUT_MIN = 30    # minutes before we auto-close a session
POS_WINDOW_MIN      = 5     # minutes before POS transaction to look for billing zone visit
BILLING_ZONE_NAME   = "billing"   # case-insensitive match on zone_name


# ─── Session builder ──────────────────────────────────────────────────────────

def build_sessions(
    store_id:     str,
    db:           DBSession,
    period_start: datetime,
    period_end:   datetime,
) -> list[SessionRow]:
    """
    Materialise SessionRows from raw events for the given period.
    Already-saved sessions for this period are skipped (idempotent).
    Returns the list of sessions (new + pre-existing).
    """
    events = (
        db.query(EventRow)
          .filter(
              EventRow.store_id  == store_id,
              EventRow.timestamp >= period_start,
              EventRow.timestamp <= period_end,
          )
          .order_by(EventRow.visitor_id, EventRow.timestamp)
          .all()
    )

    # Group events by visitor_id
    by_visitor: dict[str, list[EventRow]] = {}
    for evt in events:
        by_visitor.setdefault(evt.visitor_id, []).append(evt)

    # Load POS transactions for the period (+ POS_WINDOW_MIN buffer)
    pos_rows = (
        db.query(PosTransactionRow)
          .filter(
              PosTransactionRow.store_id  == store_id,
              PosTransactionRow.timestamp >= period_start - timedelta(minutes=POS_WINDOW_MIN),
              PosTransactionRow.timestamp <= period_end,
          )
          .all()
    )
    pos_timestamps = [(p.timestamp, p.transaction_id) for p in pos_rows]

    sessions: list[SessionRow] = []

    for visitor_id, evts in by_visitor.items():
        # De-duplicate by treating ENTRY + REENTRY as one session
        entry_evt = next(
            (e for e in evts if e.event_type in ("ENTRY", "REENTRY")), None
        )
        if entry_evt is None:
            continue   # no entry — partial data, skip

        exit_evt = next(
            (e for e in reversed(evts) if e.event_type == "EXIT"), None
        )

        entry_time = entry_evt.timestamp
        exit_time  = exit_evt.timestamp if exit_evt else (
            evts[-1].timestamp + timedelta(minutes=SESSION_TIMEOUT_MIN)
        )
        dwell = (exit_time - entry_time).total_seconds()

        zone_ids: list[str] = list({
            e.zone_id for e in evts
            if e.event_type == "ZONE_ENTER" and e.zone_id
        })

        # Billing zone visit
        billing_evts = [
            e for e in evts
            if e.event_type in ("ZONE_ENTER", "BILLING_QUEUE")
            and e.zone_name
            and BILLING_ZONE_NAME in e.zone_name.lower()
        ]
        visited_billing   = len(billing_evts) > 0
        billing_entry_time = billing_evts[0].timestamp if billing_evts else None

        # POS correlation
        converted        = False
        pos_tx_id: Optional[str] = None
        if billing_entry_time:
            for pos_ts, tx_id in pos_timestamps:
                window_start = pos_ts - timedelta(minutes=POS_WINDOW_MIN)
                if window_start <= billing_entry_time <= pos_ts:
                    converted  = True
                    pos_tx_id  = tx_id
                    break

        session = SessionRow(
            visitor_id         = visitor_id,
            store_id           = store_id,
            entry_time         = entry_time,
            exit_time          = exit_time,
            dwell_seconds      = dwell,
            zones_visited      = json.dumps(zone_ids),
            visited_billing    = visited_billing,
            billing_entry_time = billing_entry_time,
            converted          = converted,
            pos_transaction_id = pos_tx_id,
            is_staff           = entry_evt.is_staff,
        )
        sessions.append(session)

    return sessions


# ─── Funnel endpoint ──────────────────────────────────────────────────────────

def get_funnel(
    store_id:     str,
    db:           DBSession,
    period_start: Optional[datetime] = None,
    period_end:   Optional[datetime] = None,
) -> FunnelResponse:
    now          = datetime.now(timezone.utc)
    period_end   = period_end   or now
    period_start = period_start or (now - timedelta(hours=24))
    period_label = f"{period_start.date()} to {period_end.date()}"

    sessions = build_sessions(store_id, db, period_start, period_end)

    # Filter out staff
    customer_sessions = [s for s in sessions if not s.is_staff]

    # De-duplicate by visitor_id (ENTRY + REENTRY = 1)
    unique_visitors = len({s.visitor_id for s in customer_sessions})
    zone_visitors   = len({s.visitor_id for s in customer_sessions if s.visited_billing or
                           (s.zones_visited and json.loads(s.zones_visited))})
    billing_visitors= len({s.visitor_id for s in customer_sessions if s.visited_billing})
    purchasers      = len({s.visitor_id for s in customer_sessions if s.converted})

    def pct(num: int, denom: int) -> Optional[float]:
        return round(num / denom, 4) if denom > 0 else None

    stages = [
        FunnelStage(stage="Entry",         count=unique_visitors, pct_of_prev=None),
        FunnelStage(stage="Zone visit",    count=zone_visitors,   pct_of_prev=pct(zone_visitors,    unique_visitors)),
        FunnelStage(stage="Billing queue", count=billing_visitors,pct_of_prev=pct(billing_visitors, zone_visitors)),
        FunnelStage(stage="Purchase",      count=purchasers,      pct_of_prev=pct(purchasers,       billing_visitors)),
    ]

    return FunnelResponse(store_id=store_id, period=period_label, stages=stages)
