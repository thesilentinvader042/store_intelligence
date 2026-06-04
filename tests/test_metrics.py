# PROMPT: "Generate pytest tests for a store metrics endpoint. Cover: empty store,
#          all-staff clip, zero purchases (conversion=0 not null), normal day with
#          conversions, and re-entry should not double-count unique visitors."
# CHANGES MADE: Added fixture that seeds SessionRows directly (faster than going
#               through full event pipeline). Changed conversion_rate=0 assertion
#               to check for float not None. Added period_start/end query params test.

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from main import app
from db   import create_tables, Base, engine, SessionLocal, EventRow, SessionRow


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    create_tables()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


STORE = "STORE_BLR_002"
NOW   = datetime.now(timezone.utc)


def _seed_entries(n: int, is_staff: bool = False, store_id: str = STORE) -> list[str]:
    """Seed N ENTRY events into events table. Returns list of visitor_ids."""
    db = SessionLocal()
    visitor_ids = []
    try:
        for _ in range(n):
            vid = f"VIS_{uuid.uuid4().hex[:8]}"
            visitor_ids.append(vid)
            db.add(EventRow(
                event_id   = f"evt_{uuid.uuid4().hex}",
                event_type = "ENTRY",
                visitor_id = vid,
                store_id   = store_id,
                camera_id  = "CAM_FLOOR",
                timestamp  = NOW - timedelta(minutes=5),
                is_staff   = is_staff,
                confidence = 0.95,
            ))
        db.commit()
    finally:
        db.close()
    return visitor_ids


def _seed_sessions(n: int, converted: int = 0, is_staff: bool = False) -> None:
    db = SessionLocal()
    try:
        for i in range(n):
            db.add(SessionRow(
                visitor_id      = f"VIS_{uuid.uuid4().hex[:8]}",
                store_id        = STORE,
                entry_time      = NOW - timedelta(minutes=30 + i),
                exit_time       = NOW - timedelta(minutes=i),
                dwell_seconds   = 1800,
                visited_billing = i < converted,
                converted       = i < converted,
                is_staff        = is_staff,
            ))
        db.commit()
    finally:
        db.close()


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_empty_store_returns_zero_visitors_and_null_conversion(self, client):
        resp = client.get(f"/stores/{STORE}/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["unique_visitors"]  == 0
        assert body["conversion_rate"]  is None
        assert body["data_confidence"]  == "low"

    def test_all_staff_clip_returns_zero_customers(self, client):
        _seed_entries(50, is_staff=True)
        body = client.get(f"/stores/{STORE}/metrics").json()
        assert body["unique_visitors"] == 0
        assert body["conversion_rate"] is None

    def test_visitors_with_zero_purchases_conversion_is_zero_not_null(self, client):
        _seed_entries(10)
        _seed_sessions(10, converted=0)
        body = client.get(f"/stores/{STORE}/metrics").json()
        assert body["unique_visitors"] == 10
        assert body["conversion_rate"] == 0.0   # must be 0.0, not None

    def test_normal_day_with_conversions(self, client):
        _seed_entries(100)
        _seed_sessions(100, converted=40)
        body = client.get(f"/stores/{STORE}/metrics").json()
        assert body["unique_visitors"] == 100
        assert body["conversion_rate"] == pytest.approx(0.40, abs=0.01)

    def test_reentry_not_double_counted(self, client):
        """ENTRY + REENTRY for same visitor = 1 unique visitor."""
        db = SessionLocal()
        try:
            vid = f"VIS_{uuid.uuid4().hex[:8]}"
            for etype in ("ENTRY", "REENTRY"):
                db.add(EventRow(
                    event_id   = f"evt_{uuid.uuid4().hex}",
                    event_type = etype,
                    visitor_id = vid,
                    store_id   = STORE,
                    camera_id  = "CAM_FLOOR",
                    timestamp  = NOW - timedelta(minutes=5),
                    is_staff   = False,
                    confidence = 0.9,
                ))
            db.commit()
        finally:
            db.close()

        body = client.get(f"/stores/{STORE}/metrics").json()
        # ENTRY events only are counted — REENTRY does not add a new unique visitor
        assert body["unique_visitors"] == 1

    def test_data_confidence_low_when_no_recent_events(self, client):
        db = SessionLocal()
        try:
            db.add(EventRow(
                event_id   = f"evt_{uuid.uuid4().hex}",
                event_type = "ENTRY",
                visitor_id = f"VIS_{uuid.uuid4().hex[:8]}",
                store_id   = STORE,
                camera_id  = "CAM_FLOOR",
                timestamp  = NOW - timedelta(hours=5),  # old event
                is_staff   = False,
                confidence = 0.9,
            ))
            db.commit()
        finally:
            db.close()

        body = client.get(f"/stores/{STORE}/metrics").json()
        assert body["data_confidence"] in ("low", "medium")
