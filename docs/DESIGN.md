# DESIGN.md — System Design & AI-Assisted Decisions

## System Overview

Store Intelligence ingests video clips through a detection pipeline, emits structured
events to a JSONL file (or Kafka topic — see below), and exposes a FastAPI layer
for real-time metrics, funnel analysis, anomaly detection, and system health.

The north-star metric is `conversion_rate = purchasing_visitors / total_unique_visitors`.
Every architectural decision is evaluated against whether it makes that number more
accurate or more actionable.

---

## AI-Assisted Decisions

### 1. Zone classification method

I asked Claude to compare three approaches for mapping bounding-box centroids to
store zones: a VLM (GPT-4V) prompted with zone descriptions, a k-NN classifier
trained on labelled frames, and rule-based polygon overlap from `store_layout.json`.

Claude's recommendation was polygon overlap. Reasoning: the store layout is fixed,
the polygons are already provided, and `cv2.pointPolygonTest` is deterministic and
runs in microseconds. A VLM would add latency (100–500ms per frame) with no accuracy
benefit on fixed physical zones. The k-NN approach requires labelled training data
we don't have.

I agreed. The VLM approach is documented as a useful alternative only for stores
where zone boundaries are ambiguous (e.g. open-plan layouts with no physical dividers).

### 2. Re-ID embedding model selection

I asked Claude whether to use OSNet (`osnet_x0_25`), a ResNet-50 Re-ID model, or
a CLIP ViT for the appearance gallery.

Claude suggested OSNet-x0_25 because it was designed specifically for person Re-ID
(trained on Market-1501, DukeMTMC), is the lightest torchreid model (0.6M params),
and produces 512-d embeddings in < 2ms on CPU. CLIP ViT is more general and performs
worse on the Re-ID task without fine-tuning. ResNet-50 is 10× heavier with marginal
accuracy gain at this scale.

I agreed, but added the spatial fallback after noticing that torchreid's install
footprint (~600MB with torch) would break the Docker build time budget. The fallback
means the system runs correctly even without the full Re-ID stack.

### 3. Anomaly detection threshold strategy

I asked Claude whether anomaly thresholds should be hard-coded, loaded from config,
or learned dynamically from historical data.

Claude recommended starting with hard-coded thresholds in environment variables,
explicitly documenting each one in `anomalies.py`, and adding a TODO to replace
`QUEUE_DEPTH_THRESHOLD` with a per-store learned value after 2 weeks of data
collection. The reasoning: dynamic thresholds need enough history to avoid
false positives on the first day of operation, and the scoring rubric doesn't
require ML-based thresholds — only that the detectors fire correctly on the test set.

I partially agreed: I kept thresholds in code constants (not env vars) for the
hackathon submission so the reviewer can read them inline, but noted in comments
where each one should eventually come from a `store_config` table.

---

## Kafka Integration — Future Architecture

This section describes how to add Kafka when scaling beyond a single-store, batch-mode setup.

### Why Kafka?

The current architecture writes events to a JSONL file and ingests via POST. This
works for batch processing of recorded clips. It breaks when you add:
- Live camera feeds (continuous event streams)
- Multiple stores ingesting simultaneously
- Dashboard websockets that need sub-second latency
- Audit trails and event replay for debugging

Kafka solves all four by acting as a durable, ordered, replayable event bus between
the detection pipeline and the API layer.

### Proposed topic structure

```
store.events.raw          # raw detections from pipeline/detect.py (high volume)
store.events.enriched     # after Re-ID + zone mapping (same volume, richer payload)
store.sessions            # assembled sessions from funnel.py (low volume)
store.anomalies           # anomaly events as they fire (very low volume)
```

Each topic uses `store_id` as the partition key so all events for one store go to
one partition, preserving per-store ordering.

### Migration plan (zero breaking changes to the API)

**Step 1 — Producer side (pipeline/detect.py)**

Replace the `EventWriter` JSONL append with a Kafka producer:

```python
# Before (current)
self.writer.write(evt)

# After (Kafka)
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
    key_serializer=str.encode,
    value_serializer=lambda v: json.dumps(v).encode(),
)

producer.send(
    topic="store.events.raw",
    key=self.store_id,
    value=evt.model_dump(mode="json"),
)
```

**Step 2 — Consumer side (new worker: app/kafka_consumer.py)**

A long-running worker that reads from `store.events.raw` and calls `ingest_events()`
exactly as the HTTP endpoint does:

```python
from kafka import KafkaConsumer
from db import SessionLocal
from ingestion import ingest_events
from models import IngestRequest

consumer = KafkaConsumer(
    "store.events.raw",
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
    group_id="store-intelligence-ingest",
    auto_offset_reset="earliest",
    value_deserializer=lambda m: json.loads(m.decode()),
)

for message in consumer:
    db = SessionLocal()
    try:
        payload = IngestRequest(events=[message.value])
        ingest_events(payload, db)
    finally:
        db.close()
```

The HTTP `POST /events/ingest` endpoint stays intact — both paths write to the
same database through the same `ingest_events()` function. Idempotency via
`event_id` deduplication prevents double-writes if both paths are active during migration.

**Step 3 — docker-compose.yml additions**

```yaml
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    healthcheck:
      test: ["CMD-SHELL", "kafka-topics --bootstrap-server localhost:9092 --list"]
      interval: 10s
      retries: 10

  consumer:
    build: .
    command: python app/kafka_consumer.py
    depends_on:
      kafka:
        condition: service_healthy
      db:
        condition: service_healthy
    environment:
      KAFKA_BOOTSTRAP: kafka:9092
      DATABASE_URL: postgresql://user:pass@db:5432/store_intel
```

### What Kafka enables that the current architecture can't do

| Capability | Current (JSONL → HTTP) | With Kafka |
|---|---|---|
| Live camera feed | No (batch only) | Yes |
| Multi-store parallel ingest | Manual orchestration | Topic partitioning |
| Event replay for debugging | Re-run pipeline script | Seek to offset |
| Dashboard websocket latency | Poll every N seconds | Consume directly |
| At-least-once delivery | Not guaranteed | Built-in |
| Dead-letter queue for bad events | Manual error file | Dedicated DLQ topic |

### Backpressure and ordering guarantees

The consumer uses `group_id="store-intelligence-ingest"` so Kafka tracks its
offset. If the API is down, events accumulate in the topic and are processed
when it comes back up — no events are lost. The partition-by-store_id strategy
guarantees that ENTRY always arrives before EXIT for the same visitor when
produced from a single pipeline process.
