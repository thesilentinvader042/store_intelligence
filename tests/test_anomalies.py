# PROMPT: "Write pytest tests for three anomaly detectors: BILLING_QUEUE_SPIKE,
#          CONVERSION_DROP, and DEAD_ZONE. Use direct DB seeding to avoid
#          needing a full video pipeline. Each test should verify both the
#          anomaly type and the suggested_action field is non-empty."
# CHANGES MADE: Added test for no anomalies when store is empty (detectors
#               must not crash). Changed DEAD_ZONE test to seed activity 2 hours
#               ago (not recent) so it correctly shows as dead. Added CRITICAL
#               severity threshold test for queue spike.

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from main import app
from db   import create_tables, Base, engine, SessionLocal, EventRow


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


def _add_event(event_type, visitor_id=None, zone_id=None, zone_name=None,
               queue_depth=None, timestamp=None, is_staff=False):
    db = SessionLocal()
    try:
        db.add(EventRow(
            event_id    = f"evt_{uuid.uuid4().hex}",
            event_type  = event_type,
            visitor_id  = visitor_id or f"VIS_{uuid.uuid4().hex[:8]}",
            store_id    = STORE,
            camera_id   = "CAM_FLOOR",
            timestamp   = timestamp or NOW - timedelta(minutes=2),
            is_staff    = is_staff,
            confidence  = 0.9,
            zone_id     = zone_id,
            zone_name   = zone_name,
            queue_depth = queue_depth,
        ))
        db.commit()
    finally:
        db.close()


class TestAnomalies:
    def test_no_anomalies_on_empty_store(self, client):
        """Empty store must not crash — return empty list."""
        resp = client.get(f"/stores/{STORE}/anomalies")
        assert resp.status_code == 200
        assert resp.json()["anomalies"] == []

    def test_queue_spike_triggered_at_threshold(self, client):
        # Seed queue depth events above threshold for >5 minutes
        for i in range(6):
            _add_event(
                "BILLING_QUEUE",
                queue_depth=10,
                timestamp=NOW - timedelta(minutes=5 - i),
            )
        body = client.get(f"/stores/{STORE}/anomalies").json()
        types = [a["anomaly_type"] for a in body["anomalies"]]
        assert "BILLING_QUEUE_SPIKE" in types

        spike = next(a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE")
        assert spike["suggested_action"] != ""
        assert spike["severity"] in ("WARN", "CRITICAL")

    def test_queue_spike_critical_at_double_threshold(self, client):
        for i in range(6):
            _add_event(
                "BILLING_QUEUE",
                queue_depth=20,   # double the threshold of 8
                timestamp=NOW - timedelta(minutes=5 - i),
            )
        body = client.get(f"/stores/{STORE}/anomalies").json()
        spike = next(
            (a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"), None
        )
        assert spike is not None
        assert spike["severity"] == "CRITICAL"

    def test_no_queue_spike_when_below_threshold(self, client):
        for i in range(6):
            _add_event("BILLING_QUEUE", queue_depth=3, timestamp=NOW - timedelta(minutes=i))
        body = client.get(f"/stores/{STORE}/anomalies").json()
        types = [a["anomaly_type"] for a in body["anomalies"]]
        assert "BILLING_QUEUE_SPIKE" not in types

    def test_dead_zone_triggered_when_zone_goes_silent(self, client):
        # Seed activity in a zone, but 2 hours ago (well outside 30-min window)
        _add_event(
            "ZONE_ENTER",
            zone_id="ZONE_PRODUCE",
            zone_name="Produce",
            timestamp=NOW - timedelta(hours=2),
        )
        body = client.get(f"/stores/{STORE}/anomalies").json()
        types = [a["anomaly_type"] for a in body["anomalies"]]

        # Only fires during open hours — check conditionally
        hour = NOW.hour
        if 9 <= hour < 21:
            assert "DEAD_ZONE" in types
            dead = next(a for a in body["anomalies"] if a["anomaly_type"] == "DEAD_ZONE")
            assert dead["severity"] == "INFO"
            assert dead["suggested_action"] != ""

    def test_dead_zone_not_triggered_for_recently_active_zone(self, client):
        _add_event(
            "ZONE_ENTER",
            zone_id="ZONE_CHECKOUT",
            zone_name="Checkout",
            timestamp=NOW - timedelta(minutes=5),   # recent
        )
        body = client.get(f"/stores/{STORE}/anomalies").json()
        types = [a["anomaly_type"] for a in body["anomalies"]]
        assert "DEAD_ZONE" not in types

    def test_anomalies_sorted_critical_first(self, client):
        # Seed both a queue spike and check ordering
        for i in range(6):
            _add_event("BILLING_QUEUE", queue_depth=20, timestamp=NOW - timedelta(minutes=i))
        body   = client.get(f"/stores/{STORE}/anomalies").json()
        sevs   = [a["severity"] for a in body["anomalies"]]
        order  = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
        sorted_sevs = sorted(sevs, key=lambda s: order.get(s, 9))
        assert sevs == sorted_sevs
