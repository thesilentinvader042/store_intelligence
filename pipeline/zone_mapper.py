"""
zone_mapper.py — Map bounding-box centroids to named zones using
polygon ROIs defined in store_layout.json.

Expected store_layout.json shape:
{
  "store_id": "STORE_BLR_002",
  "cameras": {
    "CAM_ENTRY": {"fov_polygon": [[0,0],[1,0],[1,1],[0,1]]},
    "CAM_FLOOR": {"fov_polygon": [[0,0],[1,0],[1,1],[0,1]]}
  },
  "zones": [
    {
      "zone_id": "ZONE_ENTRY",
      "zone_name": "Entry",
      "polygon": [[0.0,0.0],[0.2,0.0],[0.2,1.0],[0.0,1.0]]
    },
    {
      "zone_id": "ZONE_BILLING",
      "zone_name": "Billing",
      "polygon": [[0.8,0.0],[1.0,0.0],[1.0,1.0],[0.8,1.0]]
    }
  ]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Zone:
    zone_id:   str
    zone_name: str
    polygon:   np.ndarray   # shape (N, 2), float32, normalised 0-1


class ZoneMapper:
    def __init__(self, layout_path: str | Path):
        raw = json.loads(Path(layout_path).read_text())
        self.store_id: str = raw["store_id"]
        self.zones: list[Zone] = []

        for z in raw.get("zones", []):
            pts = np.array(z["polygon"], dtype=np.float32)
            self.zones.append(Zone(
                zone_id=z["zone_id"],
                zone_name=z["zone_name"],
                polygon=pts,
            ))

    def get_zone(self, bbox: list[float]) -> Optional[Zone]:
        """
        Given a normalised bounding box [x1, y1, x2, y2],
        return the first zone whose polygon contains the centroid,
        or None if no zone matches.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        point = (float(cx), float(cy))

        for zone in self.zones:
            # pointPolygonTest: >0 inside, 0 on edge, <0 outside
            result = cv2.pointPolygonTest(zone.polygon, point, measureDist=False)
            if result >= 0:
                return zone
        return None

    def get_all_zones(self) -> list[Zone]:
        return self.zones


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, tempfile, json

    dummy_layout = {
        "store_id": "STORE_TEST",
        "zones": [
            {"zone_id": "ZONE_ENTRY",   "zone_name": "Entry",   "polygon": [[0.0,0.0],[0.3,0.0],[0.3,1.0],[0.0,1.0]]},
            {"zone_id": "ZONE_BILLING", "zone_name": "Billing", "polygon": [[0.7,0.0],[1.0,0.0],[1.0,1.0],[0.7,1.0]]},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dummy_layout, f)
        tmp = f.name

    zm = ZoneMapper(tmp)
    print(zm.get_zone([0.05, 0.1, 0.15, 0.4]))   # → ZONE_ENTRY
    print(zm.get_zone([0.72, 0.1, 0.85, 0.4]))   # → ZONE_BILLING
    print(zm.get_zone([0.45, 0.1, 0.55, 0.4]))   # → None (middle)
