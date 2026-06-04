"""
detect.py — Main detection pipeline.

Usage:
    python pipeline/detect.py \
        --clip_dir  ./clips/ST1008 \
        --layout    ./store_layout.json \
        --output    ./events.jsonl \
        --store_id  ST1008 \
        --camera_id CAM_FLOOR

Processes every .mp4 / .avi in clip_dir in alphabetical order.
Emits events to a single JSONL file.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from emit import (
    EntryEvent,
    ExitEvent,
    ReentryEvent,
    ZoneEnterEvent,
    ZoneExitEvent,
    BillingQueueEvent,
    EventWriter,
    new_visitor_id,
)
from tracker import ReIDGallery
from zone_mapper import ZoneMapper
from staff_classifier import StaffClassifier
from cross_camera_dedup import CrossCameraDeduplicator


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_WEIGHTS    = "yolov8n.pt"   # swap to yolov8n.pt for speed
PERSON_CLASS     = 0
DETECTION_CONF   = 0.40
FRAME_SKIP       = 2              # process every frame; set >0 to downsample


# ─── Per-track state ──────────────────────────────────────────────────────────

class TrackState:
    """Tracks zone transitions for a single active track."""

    def __init__(self, visitor_id: str, entry_time: datetime):
        self.visitor_id    = visitor_id
        self.entry_time    = entry_time
        self.current_zone  = None          # zone_id | None
        self.zone_entry_ts = None          # datetime when entered current zone


# ─── Main pipeline ────────────────────────────────────────────────────────────

class DetectionPipeline:
    def __init__(
        self,
        store_id:    str,
        camera_id:   str,
        layout_path: str,
        output_path: str,
    ):
        self.store_id   = store_id
        self.camera_id  = camera_id
        self.model      = YOLO(MODEL_WEIGHTS)
        self.model.to("mps")
        self.gallery    = ReIDGallery()
        self.zone_mapper = ZoneMapper(layout_path)
        self.staff_clf  = StaffClassifier()
        self.writer     = EventWriter(output_path)

        # Cross-camera deduplicator (Redis-backed; falls back to in-memory)
        self.dedup = CrossCameraDeduplicator(store_id)

        # active_tracks: track_id (int) → TrackState
        self.active_tracks: dict[int, TrackState] = {}

    def process_clip(self, clip_path: str) -> int:
        """Process a single video clip. Returns number of events emitted."""
        cap   = cv2.VideoCapture(clip_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total = 0
        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if FRAME_SKIP > 0 and frame_idx % (FRAME_SKIP + 1) != 0:
                frame_idx += 1
                continue

            # Timestamp derived from clip filename + frame offset
            ts = datetime.now(timezone.utc)

            results = self.model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS],
                conf=DETECTION_CONF,
                verbose=False,
            )

            seen_track_ids: set[int] = set()

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    track_id = int(box.id.item()) if box.id is not None else None
                    if track_id is None:
                        continue

                    seen_track_ids.add(track_id)
                    conf  = float(box.conf.item())
                    xyxy  = box.xyxyn[0].tolist()   # normalised [x1,y1,x2,y2]

                    # Pixel bbox for staff classification
                    h_px, w_px = frame.shape[:2]
                    bbox_px = [
                        int(box.xyxy[0][0].item()),
                        int(box.xyxy[0][1].item()),
                        int(box.xyxy[0][2].item()),
                        int(box.xyxy[0][3].item()),
                    ]

                    # Crop for Re-ID
                    x1p, y1p, x2p, y2p = bbox_px
                    crop = frame[max(0,y1p):y2p, max(0,x1p):x2p]

                    # Resolve visitor identity
                    visitor_id, is_reentry = self.gallery.resolve(
                        track_id=track_id,
                        bbox=xyxy,
                        crop=crop,
                        timestamp=time.monotonic(),
                    )

                    # Staff classification
                    staff, staff_conf = self.staff_clf.classify(frame, bbox_px)

                    # ── First time we see this track → ENTRY or REENTRY ─────
                    if track_id not in self.active_tracks:
                        self.active_tracks[track_id] = TrackState(visitor_id, ts)

                        # Cross-camera dedup: if the same person already entered
                        # from a different camera in the last 4 seconds, suppress
                        # this ENTRY and reuse their existing visitor_id.
                        embedding = self.gallery._extractor.extract(crop)                             if self.gallery._extractor else None
                        visitor_id, is_cross_cam_dup = self.dedup.check_and_register(
                            visitor_id  = visitor_id,
                            camera_id   = self.camera_id,
                            embedding   = embedding,
                            timestamp   = time.monotonic(),
                        )
                        if is_cross_cam_dup:
                            # Duplicate from another camera — suppress ENTRY, no event emitted
                            pass
                        elif is_reentry:
                            evt = ReentryEvent(
                                visitor_id=visitor_id,
                                store_id=self.store_id,
                                camera_id=self.camera_id,
                                timestamp=ts,
                                is_staff=staff,
                                confidence=conf,
                            )
                            self.writer.write(evt)
                            total += 1
                        else:
                            evt = EntryEvent(
                                visitor_id=visitor_id,
                                store_id=self.store_id,
                                camera_id=self.camera_id,
                                timestamp=ts,
                                is_staff=staff,
                                confidence=conf,
                                bbox=xyxy,
                            )
                            self.writer.write(evt)
                            total += 1

                    # ── Zone transition detection ───────────────────────────
                    zone     = self.zone_mapper.get_zone(xyxy)
                    state    = self.active_tracks[track_id]
                    zone_id  = zone.zone_id if zone else None

                    if zone_id != state.current_zone:
                        # Exiting old zone
                        if state.current_zone is not None and state.zone_entry_ts:
                            dwell = (ts - state.zone_entry_ts).total_seconds()
                            # find zone name from mapper
                            old_zone = next(
                                (z for z in self.zone_mapper.zones
                                 if z.zone_id == state.current_zone), None
                            )
                            z_exit = ZoneExitEvent(
                                visitor_id=visitor_id,
                                store_id=self.store_id,
                                camera_id=self.camera_id,
                                timestamp=ts,
                                is_staff=staff,
                                confidence=conf,
                                zone_id=state.current_zone,
                                zone_name=old_zone.zone_name if old_zone else state.current_zone,
                                dwell_seconds=dwell,
                            )
                            self.writer.write(z_exit)
                            total += 1

                        # Entering new zone
                        if zone is not None:
                            z_enter = ZoneEnterEvent(
                                visitor_id=visitor_id,
                                store_id=self.store_id,
                                camera_id=self.camera_id,
                                timestamp=ts,
                                is_staff=staff,
                                confidence=conf,
                                zone_id=zone.zone_id,
                                zone_name=zone.zone_name,
                            )
                            self.writer.write(z_enter)
                            total += 1

                        state.current_zone = zone_id
                        state.zone_entry_ts = ts if zone else None

            # ── Handle disappeared tracks → EXIT ───────────────────────────
            vanished = set(self.active_tracks.keys()) - seen_track_ids
            for tid in vanished:
                state   = self.active_tracks.pop(tid)
                entry   = self.gallery.remove_track(tid)
                dwell   = (ts - state.entry_time).total_seconds() if state.entry_time else None

                exit_evt = ExitEvent(
                    visitor_id=state.visitor_id,
                    store_id=self.store_id,
                    camera_id=self.camera_id,
                    timestamp=ts,
                    confidence=1.0,
                    dwell_seconds=dwell,
                )
                self.writer.write(exit_evt)
                total += 1

            frame_idx += 1

        cap.release()
        return total

    def run(self, clip_dir: str) -> int:
        """Process all video files in a directory, sorted by name."""
        clips = sorted(Path(clip_dir).glob("*.mp4")) + sorted(Path(clip_dir).glob("*.avi"))
        total = 0
        for clip in clips:
            print(f"  Processing {clip.name} ...")
            total += self.process_clip(str(clip))
        self.writer.close()
        return total


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir",  required=True)
    parser.add_argument("--layout",    default="store_layout.json")
    parser.add_argument("--output",    default="events.jsonl")
    parser.add_argument("--store_id",  default="ST1008")
    parser.add_argument("--camera_id", default="CAM_FLOOR")
    args = parser.parse_args()

    pipeline = DetectionPipeline(
        store_id=args.store_id,
        camera_id=args.camera_id,
        layout_path=args.layout,
        output_path=args.output,
    )

    print(f"Starting detection on {args.clip_dir}")
    n = pipeline.run(args.clip_dir)
    print(f"Done. {n} events written to {args.output}")


if __name__ == "__main__":
    main()
