"""
pos_loader.py — Load pos_transactions.csv into the database at startup.

Expected CSV columns:
  transaction_id, store_id, timestamp, amount, lane_id

Idempotent: skips rows whose transaction_id already exists.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from db import PosTransactionRow


POS_CSV_DEFAULT = "pos_transactions.csv"


def load_pos_transactions(
    db:        DBSession,
    csv_path:  str | Path = POS_CSV_DEFAULT,
    verbose:   bool = True,
) -> dict[str, int]:
    path = Path(csv_path)
    if not path.exists():
        if verbose:
            print(f"[pos_loader] No POS file found at {path} — skipping.")
        return {"loaded": 0, "skipped": 0, "errors": 0}

    loaded = skipped = errors = 0

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tx_id = row.get("transaction_id", "").strip()
            if not tx_id:
                errors += 1
                continue

            # Skip duplicates
            exists = db.query(PosTransactionRow.id).filter_by(transaction_id=tx_id).first()
            if exists:
                skipped += 1
                continue

            try:
                ts_raw = row.get("timestamp", "").strip()
                ts     = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                errors += 1
                continue

            tx = PosTransactionRow(
                transaction_id = tx_id,
                store_id       = row.get("store_id", "").strip(),
                timestamp      = ts,
                amount         = _safe_float(row.get("amount")),
                lane_id        = row.get("lane_id", "").strip() or None,
            )
            db.add(tx)
            loaded += 1

    db.commit()

    if verbose:
        print(f"[pos_loader] loaded={loaded} skipped={skipped} errors={errors}")

    return {"loaded": loaded, "skipped": skipped, "errors": errors}


def _safe_float(val: Optional[str]) -> Optional[float]:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None
