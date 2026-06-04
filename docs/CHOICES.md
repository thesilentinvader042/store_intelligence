# CHOICES.md — Architectural Decision Record

## Decision 1: Detection model — YOLOv8-small over RT-DETR and MediaPipe

**Options considered:**
- YOLOv8-nano: fastest, ~3ms/frame on CPU, but misses partial occlusions and small bounding boxes in group entries
- YOLOv8-small: 2× slower than nano (~6ms/frame), but F1 is measurably better on crowded scenes in COCO benchmarks
- RT-DETR: transformer-based, best accuracy of the three, but 4× inference time and heavier GPU dependency
- MediaPipe Pose: tracks body keypoints rather than bounding boxes, which is interesting for zone dwell but doesn't emit the raw bbox we need for Re-ID cropping

**Decision:** YOLOv8-small with `model.track(persist=True)` from the ultralytics package.

**Reasoning:** The scoring rubric penalises missed ENTRY events more than it rewards marginal accuracy gains. YOLOv8-small gives ByteTrack enough signal to maintain track IDs through moderate occlusion (the group-entry edge case), while still processing a 20-minute 1080p clip in well under 10 minutes on CPU batch mode. RT-DETR would be the right choice at 40-store scale with GPU workers — documented in the Kafka section of DESIGN.md.

**What AI suggested:** Claude suggested evaluating YOLOv8-small vs RT-DETR by running both on a 60-second sample clip and comparing ENTRY event counts against a manual ground truth. I agreed this was the right methodology and added it as a TODO in the pipeline README. I disagreed with the suggestion to use MediaPipe because we need bounding-box crops for OSNet Re-ID, not skeleton keypoints.

---

## Decision 2: Re-ID approach — gallery embedding with spatial fallback

**Options considered:**
- Pure spatial heuristic: if a new bbox appears within 100px of where the last track ended, within 3 seconds, same person. Fast to implement, fragile on busy floors.
- OSNet gallery (torchreid): extract 512-d appearance embeddings. Cosine similarity > 0.75 → same person. Survives longer gaps and crowded re-entries.
- DeepSORT: bundles Re-ID into the tracker itself. Less flexible — harder to swap the embedding model.

**Decision:** OSNet gallery (primary) + spatial heuristic (fallback when torchreid unavailable). Both live in `tracker.py` with a graceful import fallback.

**Reasoning:** The held-out test set includes re-entry edge cases with gaps up to 4 minutes. The spatial heuristic (3-second window) would miss those entirely. OSNet's 512-d embedding is small enough to run on CPU (< 2ms per crop) and the gallery TTL of 5 minutes covers the gap range. The spatial fallback ensures the system still emits events even if torchreid isn't installed.

**What AI suggested:** Claude initially suggested DeepSORT because it "comes ready-made." I pushed back: DeepSORT couples the embedding to the tracker, making it impossible to swap to a store-specific fine-tuned model later. The gallery pattern in tracker.py is intentionally model-agnostic.

---

## Decision 3: Storage — SQLite for development, PostgreSQL path in production

**Options considered:**
- SQLite: zero-config, single file, ships in Python stdlib. No separate process needed.
- PostgreSQL: full ACID, `pg_isready` healthcheck, supports concurrent writers from multiple API workers.
- Redis only: fast, but not durable; no JOIN support for POS correlation.

**Decision:** SQLite with `DATABASE_URL` defaulting to `sqlite:///./store_intel.db` for local development. Docker Compose wires it to PostgreSQL. The SQLAlchemy abstraction means the app doesn't change — only the URL does.

**Reasoning:** For a hackathon, SQLite lets you run `pytest` without Docker and cuts setup time by ~20 minutes. The acceptance gate runs against PostgreSQL via Docker Compose. At 40-store scale, you'd need connection pooling (`PgBouncer`) and read replicas for the metrics queries — but that's a deployment concern, not a code concern.

**What changes at 40-store scale:** Partition the `events` table by `store_id` + month. Add a TimescaleDB extension for time-series queries on dwell and conversion rate. Migrate metrics computation to a nightly aggregation job and serve from a `metrics_snapshots` table rather than scanning raw events on every GET.
