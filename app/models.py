"""
models.py — Pydantic v2 request/response schemas for every API endpoint.
Keep these in sync with db.py column shapes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Ingest ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    events: list[dict[str, Any]] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested:   int
    duplicate:  int
    errors:     list[str] = []


# ─── Metrics ─────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    store_id:              str
    period_start:          datetime
    period_end:            datetime
    unique_visitors:       int
    conversion_rate:       Optional[float] = None   # null when no visitors
    avg_dwell_minutes:     Optional[float] = None
    peak_hour:             Optional[int]   = None   # hour-of-day 0-23
    total_zone_visits:     int             = 0
    data_confidence:       str             = "high" # high | medium | low


# ─── Funnel ───────────────────────────────────────────────────────────────────

class FunnelStage(BaseModel):
    stage:       str
    count:       int
    pct_of_prev: Optional[float] = None   # drop-off vs previous stage


class FunnelResponse(BaseModel):
    store_id: str
    period:   str
    stages:   list[FunnelStage]


# ─── Anomalies ───────────────────────────────────────────────────────────────

class AnomalyItem(BaseModel):
    anomaly_type:     str           # BILLING_QUEUE_SPIKE | CONVERSION_DROP | DEAD_ZONE
    severity:         str           # INFO | WARN | CRITICAL
    store_id:         str
    detected_at:      datetime
    description:      str
    suggested_action: str
    metadata:         dict[str, Any] = {}


class AnomaliesResponse(BaseModel):
    store_id:  str
    anomalies: list[AnomalyItem]


# ─── Health ───────────────────────────────────────────────────────────────────

class StoreHealth(BaseModel):
    store_id:        str
    status:          str            # OK | STALE_FEED | NO_DATA
    last_event_time: Optional[datetime] = None
    lag_seconds:     Optional[float]    = None


class HealthResponse(BaseModel):
    status:  str                    # OK | DEGRADED | DATABASE_UNAVAILABLE
    stores:  list[StoreHealth]
    checked_at: datetime


# ─── Error ────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None
