"""
staff_classifier.py — Classify whether a detected person is Purplle staff.

Purplle Brigade Road Bangalore staff wear BLACK uniforms.
Black cannot be identified by hue alone — it has low saturation AND low value.
Strategy: check that a high fraction of upper-body pixels have:
  - HSV Value   < VAL_MAX   (dark / near-black)
  - HSV Sat     < SAT_MAX   (desaturated, not colourful)

Config (from environment or StaffConfig):
  STAFF_VAL_MAX        default 80   (0-255 scale)
  STAFF_SAT_MAX        default 50
  STAFF_CONF_THRESHOLD default 0.55
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StaffConfig:
    val_max:        int   = int(os.getenv("STAFF_VAL_MAX",         80))
    sat_max:        int   = int(os.getenv("STAFF_SAT_MAX",         50))
    conf_threshold: float = float(os.getenv("STAFF_CONF_THRESHOLD", 0.55))
    upper_body_fraction: float = 0.50   # top half of bbox = upper body


DEFAULT_CONFIG = StaffConfig()


class StaffClassifier:
    def __init__(self, config: StaffConfig = DEFAULT_CONFIG):
        self.cfg = config

    def classify(
        self,
        frame: np.ndarray,
        bbox_pixels: list[int],   # [x1, y1, x2, y2]
    ) -> tuple[bool, float]:
        """
        Returns (is_staff, confidence).
        confidence = fraction of upper-body pixels that are dark + desaturated.
        """
        crop = self._crop_upper_body(frame, bbox_pixels)
        if crop is None or crop.size == 0:
            return False, 0.0

        hsv       = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        _, s, v   = cv2.split(hsv)

        # Black: low value AND low saturation
        black_mask  = (v < self.cfg.val_max) & (s < self.cfg.sat_max)
        total_px    = crop.shape[0] * crop.shape[1]

        if total_px == 0:
            return False, 0.0

        confidence = float(np.sum(black_mask)) / total_px
        is_staff   = confidence >= self.cfg.conf_threshold

        return is_staff, round(confidence, 3)

    def _crop_upper_body(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h <= 0 or (x2 - x1) <= 0:
            return None
        crop_y2 = y1 + int(h * self.cfg.upper_body_fraction)
        crop_y2 = min(crop_y2, frame.shape[0])
        x1 = max(0, x1); x2 = min(x2, frame.shape[1]); y1 = max(0, y1)
        if crop_y2 <= y1 or x2 <= x1:
            return None
        return frame[y1:crop_y2, x1:x2]


_default_classifier = StaffClassifier()

def is_staff(frame: np.ndarray, bbox_pixels: list[int],
             config: StaffConfig = DEFAULT_CONFIG) -> tuple[bool, float]:
    return StaffClassifier(config).classify(frame, bbox_pixels)
