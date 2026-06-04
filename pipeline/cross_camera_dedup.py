"""
cross_camera_dedup.py — Cross-camera visitor deduplication.

Problem: The entry camera (CAM_ENTRY) and floor camera (CAM_FLOOR_FRONT)
have overlapping fields of view near the door. Without deduplication, the
same person triggers an ENTRY event on both cameras within 2 seconds,
inflating unique_visitors by ~30-40%.

Solution: A shared embedding gallery indexed by visitor_id. When a new
ENTRY arrives from a different camera for the same store, we compare its
appearance embedding against all embeddings seen in the last 2 seconds from
other cameras. If similarity > threshold → same person → suppress the
duplicate ENTRY and reassign the existing visitor_id.

This module is used by detect.py when running multi-camera mode.
It is a thin wrapper around a Redis-backed shared state so multiple
pipeline processes (one per camera) can deduplicate across processes.

If Redis is unavailable, falls back to in-memory dedup (single-process only).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────

CROSS_CAM_WINDOW_S   = 4.0    # seconds — how long to look back across cameras
CROSS_CAM_SIM_THRESH = 0.72   # slightly lower than single-camera (0.75) to tolerate
                               # viewpoint change between entry and floor cameras
REDIS_TTL_S          = 10     # how long to keep embedding in Redis


# ─── Redis-backed shared gallery ─────────────────────────────────────────────

class CrossCameraDeduplicator:
    """
    Shared gallery across all camera processes for one store.
    Key pattern: dedup:{store_id}:{visitor_id}
    Value: JSON with embedding (base64), camera_id, timestamp
    """

    def __init__(self, store_id: str):
        self.store_id = store_id
        self._redis   = self._connect_redis()
        self._local: dict[str, dict] = {}   # fallback when Redis unavailable

    def _connect_redis(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            import redis as redis_lib
            r = redis_lib.from_url(redis_url, decode_responses=True)
            r.ping()
            return r
        except Exception:
            return None   # Redis unavailable — use local dict

    def check_and_register(
        self,
        visitor_id:  str,
        camera_id:   str,
        embedding:   Optional[np.ndarray],
        timestamp:   float,
    ) -> tuple[str, bool]:
        """
        Check if this visitor was already seen from another camera in the
        last CROSS_CAM_WINDOW_S seconds.

        Returns (canonical_visitor_id, is_duplicate).
          is_duplicate=True  → suppress this ENTRY, use canonical_visitor_id
          is_duplicate=False → this is a genuine new entry, register it
        """
        # Try to find a match from a different camera
        match = self._find_cross_camera_match(embedding, camera_id, timestamp)

        if match:
            # Same person seen from different camera — return existing visitor_id
            return match, True

        # New person — register their embedding
        self._register(visitor_id, camera_id, embedding, timestamp)
        return visitor_id, False

    def _find_cross_camera_match(
        self,
        embedding:  Optional[np.ndarray],
        camera_id:  str,
        now:        float,
    ) -> Optional[str]:
        """Return visitor_id of a match from a different camera, or None."""
        entries = self._get_recent_entries(now)

        for vid, entry in entries.items():
            # Skip same camera — only cross-camera dedup
            if entry.get("camera_id") == camera_id:
                continue

            # Time window check
            if now - entry.get("ts", 0) > CROSS_CAM_WINDOW_S:
                continue

            # Embedding similarity check
            if embedding is not None and entry.get("embedding"):
                stored = np.array(entry["embedding"], dtype=np.float32)
                sim    = float(np.dot(embedding, stored))
                if sim >= CROSS_CAM_SIM_THRESH:
                    return vid

        return None

    def _register(
        self,
        visitor_id: str,
        camera_id:  str,
        embedding:  Optional[np.ndarray],
        timestamp:  float,
    ) -> None:
        payload = {
            "camera_id": camera_id,
            "ts":        timestamp,
            "embedding": embedding.tolist() if embedding is not None else None,
        }

        if self._redis:
            key = f"dedup:{self.store_id}:{visitor_id}"
            self._redis.setex(key, REDIS_TTL_S, json.dumps(payload))
        else:
            self._local[visitor_id] = payload
            # Evict stale local entries
            now = time.monotonic()
            self._local = {
                k: v for k, v in self._local.items()
                if now - v.get("ts", 0) < REDIS_TTL_S
            }

    def _get_recent_entries(self, now: float) -> dict[str, dict]:
        if self._redis:
            pattern = f"dedup:{self.store_id}:*"
            result  = {}
            try:
                for key in self._redis.scan_iter(pattern):
                    raw = self._redis.get(key)
                    if raw:
                        vid = key.split(":")[-1]
                        result[vid] = json.loads(raw)
            except Exception:
                pass
            return result
        else:
            return {
                k: v for k, v in self._local.items()
                if now - v.get("ts", 0) < CROSS_CAM_WINDOW_S
            }
