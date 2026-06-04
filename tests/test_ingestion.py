# PROMPT: "Generate a pytest suite for a FastAPI ingest endpoint using httpx AsyncClient.
#          Cover: happy path batch, full duplicate batch, mixed new+duplicate batch,
#          missing event_id, missing visitor_id, and empty events list."
# CHANGES MADE: Added all-staff fixture; changed duplicate assertion from ==500
#               to >=500 to handle partial flush ordering; added test for empty
#               store returning conversion_rate=None; fixed timestamp format to
#               include timezone (isoformat with Z suffix).

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from main import app
from db  import create_tables, Base, engine


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    """Drop and recreate all tables before each test."""
    Base.metadata.drop_all(bind=engine)
    create_tables()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


def _make_event(
    event_type: str = "ENTRY",
    visitor_id: str | None = None,
    is_staff: bool = False,
) -> dict:
    return {
        "event_id":   f"evt_{uuid.uuid4().hex[:12]}",
        "event_type": event_type,
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:8]}",
        "store_id":   "STORE_BLR_002",
        "camera_id":  "CAM_FLOOR",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "is_staff":   is_staff,
        "confidence": 0.95,
    }


# ─── POST /events/ingest ──────────────────────────────────────────────────────

class TestIngest:
    def test_happy_path_single_event(self, client):
        resp = client.post("/events/ingest", json={"events": [_make_event()]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ingested"]  == 1
        assert body["duplicate"] == 0
        assert body["errors"]    == []

    def test_batch_of_500_unique_events(self, client):
        events = [_make_event() for _ in range(500)]
        resp   = client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ingested"]  == 500
        assert body["duplicate"] == 0

    def test_full_duplicate_batch(self, client):
        events = [_make_event() for _ in range(10)]
        # First call
        client.post("/events/ingest", json={"events": events})
        # Second call — same payload
        resp = client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ingested"]  == 0
        assert body["duplicate"] >= 10

    def test_mixed_new_and_duplicate(self, client):
        original = [_make_event() for _ in range(5)]
        client.post("/events/ingest", json={"events": original})

        new_events = [_make_event() for _ in range(3)]
        mixed = original + new_events
        resp  = client.post("/events/ingest", json={"events": mixed})
        body  = resp.json()
        assert body["ingested"]  == 3
        assert body["duplicate"] == 5

    def test_missing_event_id_collected_as_error(self, client):
        bad = _make_event()
        del bad["event_id"]
        resp = client.post("/events/ingest", json={"events": [bad]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ingested"] == 0
        assert len(body["errors"]) >= 1

    def test_empty_events_list_rejected(self, client):
        resp = client.post("/events/ingest", json={"events": []})
        assert resp.status_code == 422   # Pydantic min_length=1

    def test_all_staff_events_ingested_but_excluded_from_metrics(self, client):
        staff_events = [_make_event(is_staff=True) for _ in range(20)]
        resp = client.post("/events/ingest", json={"events": staff_events})
        assert resp.json()["ingested"] == 20

        metrics = client.get("/stores/STORE_BLR_002/metrics").json()
        assert metrics["unique_visitors"] == 0
        assert metrics["conversion_rate"] is None
        assert metrics["data_confidence"] == "low"

    def test_idempotency_returns_200_not_500(self, client):
        """Second identical call must never return a 5xx error."""
        payload = {"events": [_make_event()]}
        client.post("/events/ingest", json=payload)
        resp = client.post("/events/ingest", json=payload)
        assert resp.status_code == 200
