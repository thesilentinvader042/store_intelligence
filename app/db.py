"""
db.py — SQLAlchemy ORM models, session factory, and TimescaleDB setup.

Tables:
  events          — hypertable (TimescaleDB) partitioned by timestamp
  sessions        — one row per visitor visit
  pos_transactions— loaded from CSV at startup
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index,
    Integer, String, Text, create_engine, UniqueConstraint, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./store_intel.db")
IS_POSTGRES  = "postgresql" in DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if not IS_POSTGRES else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─── ORM models ───────────────────────────────────────────────────────────────

class EventRow(Base):
    __tablename__ = "events"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    event_id    = Column(String(64),  nullable=False, unique=True, index=True)
    event_type  = Column(String(32),  nullable=False, index=True)
    visitor_id  = Column(String(32),  nullable=False, index=True)
    store_id    = Column(String(64),  nullable=False, index=True)
    camera_id   = Column(String(64),  nullable=True)
    timestamp   = Column(DateTime,    nullable=False, index=True)
    is_staff    = Column(Boolean,     default=False)
    confidence  = Column(Float,       default=1.0)
    zone_id     = Column(String(64),  nullable=True)
    zone_name   = Column(String(128), nullable=True)
    queue_depth = Column(Integer,     nullable=True)
    dwell_secs  = Column(Float,       nullable=True)
    raw_json    = Column(Text,        nullable=True)

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_visitor",  "visitor_id", "store_id"),
    )


class SessionRow(Base):
    __tablename__ = "sessions"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id          = Column(String(32), nullable=False, index=True)
    store_id            = Column(String(64), nullable=False, index=True)
    entry_time          = Column(DateTime,   nullable=False)
    exit_time           = Column(DateTime,   nullable=True)
    dwell_seconds       = Column(Float,      nullable=True)
    zones_visited       = Column(Text,       nullable=True)
    visited_billing     = Column(Boolean,    default=False)
    billing_entry_time  = Column(DateTime,   nullable=True)
    converted           = Column(Boolean,    default=False)
    pos_transaction_id  = Column(String(64), nullable=True)
    is_staff            = Column(Boolean,    default=False)

    __table_args__ = (
        Index("ix_sessions_store_entry", "store_id", "entry_time"),
        UniqueConstraint("visitor_id", "entry_time", name="uq_session"),
    )


class PosTransactionRow(Base):
    __tablename__ = "pos_transactions"

    id             = Column(Integer,  primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), nullable=False, unique=True, index=True)
    store_id       = Column(String(64), nullable=False, index=True)
    timestamp      = Column(DateTime,   nullable=False, index=True)
    amount         = Column(Float,      nullable=True)
    lane_id        = Column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_pos_store_ts", "store_id", "timestamp"),
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    if IS_POSTGRES:
        _apply_timescale()


def _apply_timescale() -> None:
    """
    Convert the events table to a TimescaleDB hypertable partitioned by timestamp.
    This is idempotent — safe to call on every startup.
    create_hypertable(..., if_not_exists => TRUE) skips silently if already done.
    """
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                SELECT create_hypertable(
                    'events',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """))
            conn.commit()
        except Exception as e:
            # timescaledb extension not available (e.g. plain Postgres) — skip silently
            if "function create_hypertable" not in str(e).lower():
                pass  # already a hypertable or extension missing — not fatal


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
