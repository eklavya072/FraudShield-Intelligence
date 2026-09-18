"""Storage for scored transactions and the review queue.

SQLite instead of a list in memory so a restart doesn't lose review decisions,
and so the drift page has some history to work with.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "model" / "fraudshield.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_transactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_at         TEXT    NOT NULL,
    features          TEXT    NOT NULL,
    amount            REAL    NOT NULL,
    score             REAL    NOT NULL,
    threshold         REAL    NOT NULL,
    alerted           INTEGER NOT NULL,
    expected_loss     REAL    NOT NULL,
    source            TEXT    NOT NULL DEFAULT 'api',
    status            TEXT    NOT NULL DEFAULT 'pending',
    analyst_note      TEXT,
    decided_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerted_status
    ON scored_transactions (alerted, status, expected_loss DESC);
CREATE INDEX IF NOT EXISTS idx_scored_at ON scored_transactions (scored_at);
"""

VALID_DECISIONS = {"confirmed_fraud", "false_positive"}


class AlertStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("FRAUDSHIELD_DB", DEFAULT_DB))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record(
        self,
        features: dict,
        amount: float,
        score: float,
        threshold: float,
        source: str = "api",
    ) -> int:
        """Save one scored transaction."""
        alerted = score >= threshold
        expected_loss = float(score) * float(amount)
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO scored_transactions
                   (scored_at, features, amount, score, threshold, alerted,
                    expected_loss, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(UTC).isoformat(),
                    json.dumps(features),
                    float(amount),
                    float(score),
                    float(threshold),
                    int(alerted),
                    expected_loss,
                    source,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def queue(self, status: str = "pending", limit: int = 100) -> list[dict]:
        """Alerts by expected loss, biggest first.

        Sorting on probability x amount rather than probability means a 60%
        chance on $80,000 comes before a 98% chance on $40.
        """
        where = "alerted = 1"
        params: list = []
        if status != "all":
            where += " AND status = ?"
            params.append(status)
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM scored_transactions WHERE {where}
                    ORDER BY expected_loss DESC LIMIT ?""",
                params,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def decide(self, alert_id: int, decision: str, note: str | None = None) -> bool:
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE scored_transactions
                   SET status = ?, analyst_note = ?, decided_at = ?
                   WHERE id = ? AND alerted = 1""",
                (decision, note, datetime.now(UTC).isoformat(), alert_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def recent_features(self, limit: int = 5000) -> list[dict]:
        """Recent feature rows, for the drift check."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT features, score FROM scored_transactions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{**json.loads(r["features"]), "_score": r["score"]} for r in rows]

    def summary(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS scored,
                          COALESCE(SUM(alerted), 0) AS alerts,
                          COALESCE(SUM(alerted = 1 AND status = 'pending'), 0)
                              AS pending,
                          COALESCE(SUM(status = 'confirmed_fraud'), 0) AS confirmed,
                          COALESCE(SUM(status = 'false_positive'), 0) AS dismissed,
                          COALESCE(SUM(CASE WHEN alerted = 1 AND status = 'pending'
                                       THEN expected_loss ELSE 0 END), 0)
                              AS pending_exposure
                   FROM scored_transactions"""
            ).fetchone()

        summary = {key: row[key] for key in row.keys()}
        reviewed = summary["confirmed"] + summary["dismissed"]
        # Precision on alerts that were actually reviewed, rather than on the
        # test set.
        summary["realised_precision"] = summary["confirmed"] / reviewed if reviewed else None
        summary["reviewed"] = reviewed
        return summary

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        record = {key: row[key] for key in row.keys()}
        record["features"] = json.loads(record["features"])
        record["alerted"] = bool(record["alerted"])
        return record
