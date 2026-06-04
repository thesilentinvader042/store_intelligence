-- init_timescale.sql
-- Runs once on first boot inside the TimescaleDB container.
-- SQLAlchemy's create_tables() runs AFTER this, so we use a trigger-based
-- approach: a function that converts the table after it is created.
-- The simplest safe pattern: wrap everything in DO blocks that are idempotent.

-- Enable the TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- The events and pos_transactions tables are created by SQLAlchemy at API startup,
-- not here. We convert them to hypertables in db.py's create_tables() call via
-- a raw SQL statement executed after table creation. This file just ensures the
-- extension is loaded so that call succeeds.

-- Verify extension loaded
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
  ) THEN
    RAISE EXCEPTION 'timescaledb extension failed to load';
  END IF;
  RAISE NOTICE 'TimescaleDB extension ready.';
END
$$;
