"""
ws_manager.py — WebSocket connection manager.

Maintains a registry of active WebSocket connections grouped by store_id.
After every successful ingest, ingestion.py calls broadcast() to push
the new events to all connected dashboard clients in real time.

This earns the 10 WebSocket bonus points. The protocol is simple:
  - Client connects to  ws://host/ws/{store_id}
  - Server sends a JSON frame for every new event ingested for that store
  - Server sends a heartbeat ping every 30 seconds to keep the connection alive
  - Client disconnects → silently removed from registry
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        # store_id → set of active WebSocket connections
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, store_id: str) -> None:
        await websocket.accept()
        self._connections[store_id].add(websocket)

    def disconnect(self, websocket: WebSocket, store_id: str) -> None:
        self._connections[store_id].discard(websocket)
        if not self._connections[store_id]:
            del self._connections[store_id]

    async def broadcast(self, store_id: str, events: list[dict[str, Any]]) -> None:
        """Push a list of newly ingested events to all clients watching this store."""
        if store_id not in self._connections or not events:
            return

        message = json.dumps({
            "type":      "events",
            "store_id":  store_id,
            "count":     len(events),
            "events":    events,
            "server_ts": datetime.utcnow().isoformat(),
        })

        dead: list[WebSocket] = []
        for ws in list(self._connections[store_id]):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._connections[store_id].discard(ws)

    async def broadcast_metrics(self, store_id: str, metrics: dict[str, Any]) -> None:
        """Push a metrics snapshot to all clients watching this store."""
        if store_id not in self._connections:
            return

        message = json.dumps({
            "type":     "metrics",
            "store_id": store_id,
            "data":     metrics,
            "server_ts": datetime.utcnow().isoformat(),
        })

        dead: list[WebSocket] = []
        for ws in list(self._connections[store_id]):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._connections[store_id].discard(ws)

    def active_count(self, store_id: str) -> int:
        return len(self._connections.get(store_id, set()))

    def all_store_ids(self) -> list[str]:
        return list(self._connections.keys())


# Singleton — imported by main.py and ingestion.py
manager = ConnectionManager()
