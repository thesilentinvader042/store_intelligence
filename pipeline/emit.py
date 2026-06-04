"""
emit.py — Pydantic event schemas and JSONL writer.
This is the single source of truth for all event shapes.
Everything else (API, tests, pipeline) imports from here.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ──────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    ENTRY            = "ENTRY"
    EXIT             = "EXIT"
    REENTRY          = "REENTRY"
    ZONE_ENTER       = "ZONE_ENTER"
    ZONE_EXIT        = "ZONE_EXIT"
    BILLING_QUEUE    = "BILLING_QUEUE"


class Severity(str, Enum):
    INFO     = "INFO"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"


# ─── Base event ──────────────────────────────────────────────────────────────

class BaseEvent(BaseModel):
    event_id:   str       = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    visitor_id: str       = Field(..., pattern=r"^VIS_[a-f0-9]{6,}$")
    store_id:   str
    camera_id:  str
    timestamp:  datetime
    is_staff:   bool      = False
    confidence: float     = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("visitor_id")
    @classmethod
    def validate_visitor_id(cls, v: str) -> str:
        if not v.startswith("VIS_"):
            raise ValueError("visitor_id must start with VIS_")
        return v

    def to_jsonl(self) -> str:
        return self.model_dump_json()


# ─── Concrete event types ────────────────────────────────────────────────────

class EntryEvent(BaseEvent):
    event_type: EventType = EventType.ENTRY
    bbox:       list[float] = Field(..., description="[x1, y1, x2, y2] normalised 0-1")
    group_size: int         = Field(default=1, ge=1)


class ExitEvent(BaseEvent):
    event_type:    EventType = EventType.EXIT
    dwell_seconds: Optional[float] = None


class ReentryEvent(BaseEvent):
    event_type:          EventType = EventType.REENTRY
    original_entry_time: Optional[datetime] = None
    gap_seconds:         Optional[float]    = None


class ZoneEnterEvent(BaseEvent):
    event_type: EventType = EventType.ZONE_ENTER
    zone_id:    str
    zone_name:  str


class ZoneExitEvent(BaseEvent):
    event_type:    EventType = EventType.ZONE_EXIT
    zone_id:       str
    zone_name:     str
    dwell_seconds: Optional[float] = None


class BillingQueueEvent(BaseEvent):
    event_type:  EventType = EventType.BILLING_QUEUE
    queue_depth: int        = Field(..., ge=0)


# Union type for ingest endpoint
AnyEvent = (
    EntryEvent
    | ExitEvent
    | ReentryEvent
    | ZoneEnterEvent
    | ZoneExitEvent
    | BillingQueueEvent
)


# ─── JSONL writer ────────────────────────────────────────────────────────────

class EventWriter:
    """Append events to a JSONL file, one event per line."""

    def __init__(self, output_path: str | Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: BaseEvent) -> None:
        self._fh.write(event.to_jsonl() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def new_visitor_id() -> str:
    return f"VIS_{uuid.uuid4().hex[:8]}"


def load_events_from_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file back into raw dicts (for ingestion)."""
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
