"""
main.py — FastAPI application.

Endpoints:
  POST /events/ingest
  GET  /stores/{store_id}/metrics
  GET  /stores/{store_id}/funnel
  GET  /stores/{store_id}/anomalies
  GET  /health
  GET  /dashboard
  WS   /ws/{store_id}          ← live event feed (WebSocket)
"""
from __future__ import annotations

import asyncio
import uuid
import time
import os
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from db import create_tables, get_db, SessionLocal
from models import (
    IngestRequest, IngestResponse,
    MetricsResponse, FunnelResponse, AnomaliesResponse,
    HealthResponse,
)
from ingestion  import ingest_events
from metrics    import get_metrics
from funnel     import get_funnel
from anomalies  import get_anomalies
from health     import get_health
from pos_loader import load_pos_transactions
from ws_manager import manager


# ─── Structured logging ───────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup", msg="Creating tables")
    create_tables()

    db = SessionLocal()
    try:
        pos_csv = os.getenv("POS_CSV_PATH", "pos_transactions.csv")
        result  = load_pos_transactions(db, pos_csv)
        log.info("pos_loaded", **result)
    finally:
        db.close()

    # Start WebSocket heartbeat — keeps connections alive through proxies/load balancers
    heartbeat_task = asyncio.create_task(_ws_heartbeat())

    yield

    heartbeat_task.cancel()
    log.info("shutdown", msg="Goodbye")


async def _ws_heartbeat():
    """Ping all connected WebSocket clients every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        for store_id in manager.all_store_ids():
            await manager.broadcast(store_id, [])   # empty list = heartbeat


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="Store Intelligence API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Logging middleware ───────────────────────────────────────────────────────

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    store_id = request.path_params.get("store_id", "-")
    start    = time.perf_counter()

    request.state.trace_id = trace_id
    response = await call_next(request)

    log.info(
        "request",
        trace_id    = trace_id,
        store_id    = store_id,
        method      = request.method,
        path        = request.url.path,
        status_code = response.status_code,
        latency_ms  = round((time.perf_counter() - start) * 1000, 1),
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    for candidate in [
        pathlib.Path(__file__).parent / "dashboard.html",
        pathlib.Path(__file__).parent.parent / "dashboard.html",
    ]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text())
    return HTMLResponse("<h2>dashboard.html not found</h2>", status_code=404)


# ── WebSocket live feed ───────────────────────────────────────────────────────

@app.websocket("/ws/{store_id}")
async def websocket_feed(websocket: WebSocket, store_id: str):
    """
    Live event feed for a store.

    Connect: ws://host/ws/ST1008
    Receives JSON frames:
      { "type": "events",  "store_id": "ST1008", "count": 3, "events": [...], "server_ts": "..." }
      { "type": "metrics", "store_id": "ST1008", "data": {...}, "server_ts": "..." }
      { "type": "pong",    "server_ts": "..." }

    Send "ping" to get a pong back.
    """
    await manager.connect(websocket, store_id)
    log.info("ws_connect", store_id=store_id, active=manager.active_count(store_id))

    # Send current metrics snapshot immediately on connect
    db = SessionLocal()
    try:
        snap = get_metrics(store_id, db)
        await manager.broadcast_metrics(store_id, snap.model_dump(mode="json"))
    except Exception:
        pass
    finally:
        db.close()

    try:
        while True:
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_text(
                    '{"type":"pong","server_ts":"' + datetime.utcnow().isoformat() + '"}'
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, store_id)
        log.info("ws_disconnect", store_id=store_id, active=manager.active_count(store_id))


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.post("/events/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest, db: Session = Depends(get_db)):
    result = ingest_events(payload, db)
    # After ingest, push a fresh metrics snapshot to all WS clients for this store
    if result.ingested > 0:
        store_ids = {e.get("store_id") for e in payload.events if e.get("store_id")}
        for sid in store_ids:
            try:
                snap_db = SessionLocal()
                snap = get_metrics(sid, snap_db)
                snap_db.close()
                await manager.broadcast_metrics(sid, snap.model_dump(mode="json"))
            except Exception:
                pass
    return result


@app.get("/stores/{store_id}/metrics", response_model=MetricsResponse)
def metrics(
    store_id:     str,
    period_start: datetime | None = None,
    period_end:   datetime | None = None,
    db:           Session = Depends(get_db),
):
    return get_metrics(store_id, db, period_start, period_end)


@app.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
def funnel(
    store_id:     str,
    period_start: datetime | None = None,
    period_end:   datetime | None = None,
    db:           Session = Depends(get_db),
):
    return get_funnel(store_id, db, period_start, period_end)


@app.get("/stores/{store_id}/anomalies", response_model=AnomaliesResponse)
def anomalies(store_id: str, db: Session = Depends(get_db)):
    return get_anomalies(store_id, db)


@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    try:
        return get_health(db)
    except OperationalError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "DATABASE_UNAVAILABLE", "detail": str(exc)},
        )
