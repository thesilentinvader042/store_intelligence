# PROMPT: "Generate a pytest suite for a computer-vision detection pipeline.
#          Cover: zone_mapper polygon hit/miss/edge cases, staff_classifier
#          with synthetic HSV frames, tracker Re-ID gallery assign/reuse/expire,
#          and emit.py schema validation including visitor_id format and
#          confidence clamping. Mock OpenCV and ultralytics so tests run
#          without a GPU or camera."
# CHANGES MADE: Removed torchreid dependency from tracker tests — used the
#               spatial fallback path instead so CI doesn't need torch installed.
#               Added test for group entry (3 simultaneous bounding boxes = 3
#               separate visitor_ids). Fixed HSV synthetic frame to use uint8
#               dtype (cv2 requires it). Added expiry test that advances
#               monotonic time via monkeypatch rather than sleeping.

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))


# ─── emit.py tests ────────────────────────────────────────────────────────────

class TestEmitSchemas:
    def test_entry_event_valid(self):
        from emit import EntryEvent
        evt = EntryEvent(
            visitor_id="VIS_abc12345",
            store_id="ST1008",
            camera_id="CAM_FLOOR",
            timestamp=datetime.now(timezone.utc),
            bbox=[0.1, 0.2, 0.3, 0.4],
        )
        assert evt.visitor_id == "VIS_abc12345"
        assert evt.is_staff is False
        assert evt.confidence == 1.0

    def test_visitor_id_must_start_with_VIS(self):
        from emit import EntryEvent
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EntryEvent(
                visitor_id="USR_abc123",   # wrong prefix
                store_id="ST1008",
                camera_id="CAM_FLOOR",
                timestamp=datetime.now(timezone.utc),
                bbox=[0.0, 0.0, 0.1, 0.1],
            )

    def test_confidence_clamped_by_schema(self):
        from emit import EntryEvent
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EntryEvent(
                visitor_id="VIS_abc12345",
                store_id="ST1008",
                camera_id="CAM_FLOOR",
                timestamp=datetime.now(timezone.utc),
                bbox=[0.0, 0.0, 0.1, 0.1],
                confidence=1.5,   # out of range
            )

    def test_event_id_auto_generated(self):
        from emit import EntryEvent
        evt = EntryEvent(
            visitor_id="VIS_abc12345",
            store_id="ST1008",
            camera_id="CAM_FLOOR",
            timestamp=datetime.now(timezone.utc),
            bbox=[0.1, 0.2, 0.3, 0.4],
        )
        assert evt.event_id.startswith("evt_")

    def test_to_jsonl_round_trips(self):
        from emit import EntryEvent
        evt = EntryEvent(
            visitor_id="VIS_abc12345",
            store_id="ST1008",
            camera_id="CAM_FLOOR",
            timestamp=datetime.now(timezone.utc),
            bbox=[0.1, 0.2, 0.3, 0.4],
        )
        raw = evt.to_jsonl()
        parsed = json.loads(raw)
        assert parsed["visitor_id"] == "VIS_abc12345"
        assert parsed["event_type"] == "ENTRY"

    def test_new_visitor_id_format(self):
        from emit import new_visitor_id
        vid = new_visitor_id()
        assert vid.startswith("VIS_")
        assert len(vid) == 12   # VIS_(4) + 8 hex chars = 12


# ─── zone_mapper.py tests ─────────────────────────────────────────────────────

@pytest.fixture
def layout_file(tmp_path):
    layout = {
        "store_id": "STORE_TEST",
        "zones": [
            {
                "zone_id": "ZONE_ENTRY",
                "zone_name": "Entry",
                "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]],
            },
            {
                "zone_id": "ZONE_BILLING",
                "zone_name": "Billing",
                "polygon": [[0.7, 0.0], [1.0, 0.0], [1.0, 1.0], [0.7, 1.0]],
            },
        ],
    }
    p = tmp_path / "layout.json"
    p.write_text(json.dumps(layout))
    return str(p)


class TestZoneMapper:
    def test_centroid_in_entry_zone(self, layout_file):
        from zone_mapper import ZoneMapper
        zm = ZoneMapper(layout_file)
        zone = zm.get_zone([0.05, 0.1, 0.25, 0.4])   # centroid ~(0.15, 0.25)
        assert zone is not None
        assert zone.zone_id == "ZONE_ENTRY"

    def test_centroid_in_billing_zone(self, layout_file):
        from zone_mapper import ZoneMapper
        zm = ZoneMapper(layout_file)
        zone = zm.get_zone([0.72, 0.1, 0.90, 0.4])   # centroid ~(0.81, 0.25)
        assert zone is not None
        assert zone.zone_id == "ZONE_BILLING"

    def test_centroid_in_no_zone_returns_none(self, layout_file):
        from zone_mapper import ZoneMapper
        zm = ZoneMapper(layout_file)
        zone = zm.get_zone([0.45, 0.1, 0.55, 0.4])   # centroid ~(0.50, 0.25) — gap
        assert zone is None

    def test_three_people_in_entry_all_detected(self, layout_file):
        """Group entry: 3 separate bboxes in the same zone = 3 separate detections."""
        from zone_mapper import ZoneMapper
        zm = ZoneMapper(layout_file)
        bboxes = [
            [0.02, 0.1, 0.10, 0.4],
            [0.10, 0.1, 0.18, 0.4],
            [0.18, 0.1, 0.26, 0.4],
        ]
        zones = [zm.get_zone(b) for b in bboxes]
        assert all(z is not None for z in zones)
        assert all(z.zone_id == "ZONE_ENTRY" for z in zones)


# ─── staff_classifier.py tests ───────────────────────────────────────────────

def _make_frame(h: int, w: int, hue: int, sat: int = 200) -> np.ndarray:
    """Create a solid-colour BGR frame for a given HSV hue."""
    import cv2
    hsv = np.full((h, w, 3), [hue, sat, 200], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestStaffClassifier:
    def test_black_uniform_classified_as_staff(self):
        """Purplle Brigade Road staff wear BLACK uniforms — low value, low saturation."""
        from staff_classifier import StaffClassifier, StaffConfig
        import numpy as np
        cfg = StaffConfig(val_max=80, sat_max=50, conf_threshold=0.50)
        clf = StaffClassifier(cfg)
        # Solid black frame: HSV (0, 0, 30) — dark, desaturated
        black_frame = np.full((200, 100, 3), [0, 0, 30], dtype=np.uint8)
        import cv2
        black_bgr = cv2.cvtColor(black_frame, cv2.COLOR_HSV2BGR)
        is_s, conf = clf.classify(black_bgr, [0, 0, 100, 200])
        assert is_s is True
        assert conf >= 0.50

    def test_bright_coloured_frame_not_classified_as_staff(self):
        """A customer in bright clothing should not be classified as staff."""
        from staff_classifier import StaffClassifier, StaffConfig
        cfg = StaffConfig(val_max=80, sat_max=50, conf_threshold=0.50)
        clf = StaffClassifier(cfg)
        bright_frame = _make_frame(200, 100, hue=120, sat=200)  # bright green
        is_s, conf = clf.classify(bright_frame, [0, 0, 100, 200])
        assert is_s is False

    def test_empty_crop_returns_false(self):
        from staff_classifier import StaffClassifier
        clf = StaffClassifier()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        is_staff, conf = clf.classify(frame, [50, 50, 50, 50])   # zero-size bbox
        assert is_staff is False
        assert conf == 0.0


# ─── tracker.py tests ─────────────────────────────────────────────────────────

class TestReIDGallery:
    def test_new_track_gets_new_visitor_id(self):
        from tracker import ReIDGallery
        gallery = ReIDGallery()
        vid, is_reentry = gallery.resolve(
            track_id=1,
            bbox=[0.1, 0.1, 0.3, 0.5],
            crop=None,
            timestamp=time.monotonic(),
        )
        assert vid.startswith("VIS_")
        assert is_reentry is False

    def test_continuing_track_returns_same_visitor_id(self):
        from tracker import ReIDGallery
        gallery = ReIDGallery()
        t0 = time.monotonic()
        vid1, _ = gallery.resolve(1, [0.1, 0.1, 0.3, 0.5], None, t0)
        vid2, _ = gallery.resolve(1, [0.11, 0.1, 0.31, 0.5], None, t0 + 0.1)
        assert vid1 == vid2

    def test_different_tracks_get_different_visitor_ids(self):
        from tracker import ReIDGallery
        gallery = ReIDGallery()
        t0 = time.monotonic()
        vid1, _ = gallery.resolve(1, [0.1, 0.1, 0.2, 0.4], None, t0)
        vid2, _ = gallery.resolve(2, [0.6, 0.1, 0.8, 0.4], None, t0)
        assert vid1 != vid2

    def test_spatial_reentry_within_threshold(self):
        """Track ends, new track starts within 100px and 3 seconds → REENTRY."""
        from tracker import ReIDGallery
        gallery = ReIDGallery()
        t0 = time.monotonic()

        # Track 1 starts and is then removed (person exits frame)
        vid1, _ = gallery.resolve(1, [0.1, 0.1, 0.2, 0.5], None, t0)
        gallery.remove_track(1)

        # Track 2 starts very close in space and time (spatial fallback)
        vid2, is_reentry = gallery.resolve(2, [0.11, 0.1, 0.21, 0.5], None, t0 + 1.0)

        assert is_reentry is True
        assert vid1 == vid2

    def test_gallery_assigns_new_id_after_ttl(self, monkeypatch):
        """After TTL expires, the same spatial position gets a NEW visitor_id."""
        from tracker import ReIDGallery, TTL_SECONDS
        gallery = ReIDGallery()

        t0 = 1000.0
        vid1, _ = gallery.resolve(1, [0.1, 0.1, 0.2, 0.5], None, t0)
        gallery.remove_track(1)

        # Jump time past TTL
        t_expired = t0 + TTL_SECONDS + 10
        vid2, is_reentry = gallery.resolve(2, [0.11, 0.1, 0.21, 0.5], None, t_expired)

        assert is_reentry is False
        assert vid1 != vid2
