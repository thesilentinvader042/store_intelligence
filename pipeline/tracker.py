"""
tracker.py — Re-ID gallery for persistent visitor_id assignment.

Strategy:
  Primary:  Cosine similarity on OSNet appearance embeddings (torchreid).
            If similarity > SIMILARITY_THRESHOLD → same person (REENTRY).
  Fallback: Spatial proximity heuristic — if a new track starts within
            SPATIAL_PX pixels of where the last track ended, within
            TEMPORAL_GAP_S seconds, treat as the same person.

Key design fix: _gallery holds ACTIVE tracks (by track_id key).
                _archive holds REMOVED tracks (by track_id key, with TTL).
                _match searches _archive — not _gallery — so remove_track
                must move entries to _archive rather than deleting them.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Try to import torchreid; fall back to heuristic-only mode ────────────────
try:
    import torch
    import torchreid
    _TORCHREID_AVAILABLE = True
except ImportError:
    _TORCHREID_AVAILABLE = False


# ─── Config ───────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.75
TTL_SECONDS          = 300    # 5 minutes
SPATIAL_PX           = 100
TEMPORAL_GAP_S       = 3.0
FRAME_H              = 1080
FRAME_W              = 1920


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    visitor_id:  str
    embedding:   Optional[np.ndarray]
    last_bbox:   list[float]
    last_seen:   float
    entry_time:  float


# ─── OSNet feature extractor ─────────────────────────────────────────────────

class AppearanceExtractor:
    def __init__(self):
        if not _TORCHREID_AVAILABLE:
            return
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_x0_25",
            model_path="",
            device="cpu",
        )

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if not _TORCHREID_AVAILABLE or crop is None or crop.size == 0:
            return None
        feat = self.extractor([crop])
        vec  = feat.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ─── Re-ID gallery ────────────────────────────────────────────────────────────

class ReIDGallery:
    def __init__(self):
        self._gallery: dict[str, GalleryEntry] = {}   # active tracks
        self._archive: dict[str, GalleryEntry] = {}   # removed tracks, searchable for Re-ID
        self._extractor = AppearanceExtractor() if _TORCHREID_AVAILABLE else None

    def resolve(
        self,
        track_id: int,
        bbox: list[float],
        crop: Optional[np.ndarray],
        timestamp: float,
    ) -> tuple[str, bool]:
        """Return (visitor_id, is_reentry). Call once per tracked person per frame."""
        self._evict_stale(timestamp)
        key = str(track_id)

        if key in self._gallery:
            entry = self._gallery[key]
            entry.last_bbox = bbox
            entry.last_seen = timestamp
            return entry.visitor_id, False

        embedding = None
        if self._extractor and crop is not None:
            embedding = self._extractor.extract(crop)

        matched = self._match(embedding, bbox, timestamp)

        if matched:
            matched.last_bbox = bbox
            matched.last_seen = timestamp
            matched.embedding = embedding or matched.embedding
            self._gallery[key] = matched
            # Remove from archive — now active again
            self._archive = {k: v for k, v in self._archive.items() if v is not matched}
            return matched.visitor_id, True
        else:
            visitor_id = f"VIS_{uuid.uuid4().hex[:8]}"
            self._gallery[key] = GalleryEntry(
                visitor_id=visitor_id,
                embedding=embedding,
                last_bbox=bbox,
                last_seen=timestamp,
                entry_time=timestamp,
            )
            return visitor_id, False

    def remove_track(self, track_id: int) -> Optional[GalleryEntry]:
        """
        Move an ended track from _gallery to _archive.
        _archive is what _match searches for Re-ID — must NOT delete here.
        """
        key = str(track_id)
        entry = self._gallery.pop(key, None)
        if entry is not None:
            self._archive[key] = entry
        return entry

    # ── Matching ──────────────────────────────────────────────────────────────

    def _match(
        self,
        embedding: Optional[np.ndarray],
        bbox: list[float],
        now: float,
    ) -> Optional[GalleryEntry]:
        """Search _archive (removed tracks) for a match. Active tracks are never candidates."""
        candidates = [
            e for e in self._archive.values()
            if (now - e.last_seen) < TTL_SECONDS
        ]

        # Embedding similarity (primary)
        if embedding is not None:
            best_entry, best_sim = None, -1.0
            for entry in candidates:
                if entry.embedding is None:
                    continue
                sim = float(np.dot(embedding, entry.embedding))
                if sim > SIMILARITY_THRESHOLD and sim > best_sim:
                    best_sim, best_entry = sim, entry
            if best_entry is not None:
                return best_entry

        # Spatial fallback
        cx = (bbox[0] + bbox[2]) / 2.0 * FRAME_W
        cy = (bbox[1] + bbox[3]) / 2.0 * FRAME_H

        for entry in candidates:
            if (now - entry.last_seen) > TEMPORAL_GAP_S:
                continue
            ex = (entry.last_bbox[0] + entry.last_bbox[2]) / 2.0 * FRAME_W
            ey = (entry.last_bbox[1] + entry.last_bbox[3]) / 2.0 * FRAME_H
            if ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 < SPATIAL_PX:
                return entry

        return None

    def _evict_stale(self, now: float) -> None:
        """Remove entries from both gallery and archive older than TTL."""
        for store in (self._gallery, self._archive):
            stale = [k for k, e in store.items() if (now - e.last_seen) > TTL_SECONDS]
            for k in stale:
                del store[k]
