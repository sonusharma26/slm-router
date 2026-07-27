"""Append-only, hash-chained local event ledger (V2-101/V2-105)."""

from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    sequence: int
    event_id: str
    event_type: str
    schema_version: str
    occurred_at: str
    actor: str
    idempotency_key: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str


class SQLiteLedger:
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
          event_type TEXT NOT NULL, schema_version TEXT NOT NULL, occurred_at TEXT NOT NULL,
          actor TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
          previous_hash TEXT NOT NULL, event_hash TEXT UNIQUE NOT NULL);
        CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
        """)
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)",
            (self.SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor: str,
        idempotency_key: str,
        schema_version: str = "2.0",
    ) -> Event:
        existing = self.connection.execute(
            "SELECT * FROM events WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            return self._event(existing)
        last = self.connection.execute(
            "SELECT event_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = last[0] if last else "0" * 64
        occurred = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(
            "|".join(
                (previous, event_type, schema_version, occurred, actor, idempotency_key, canonical)
            ).encode()
        ).hexdigest()
        event_id = digest[:32]
        with self.connection:
            self.connection.execute(
                """INSERT INTO events(event_id,event_type,schema_version,occurred_at,
              actor,idempotency_key,payload,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    event_type,
                    schema_version,
                    occurred,
                    actor,
                    idempotency_key,
                    canonical,
                    previous,
                    digest,
                ),
            )
        return self.get(event_id)

    def get(self, event_id: str) -> Event:
        row = self.connection.execute(
            "SELECT * FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        if not row:
            raise KeyError(event_id)
        return self._event(row)

    def events(self, event_type: str | None = None) -> list[Event]:
        sql, params = (
            ("SELECT * FROM events ORDER BY sequence", ())
            if event_type is None
            else ("SELECT * FROM events WHERE event_type=? ORDER BY sequence", (event_type,))
        )
        return [self._event(row) for row in self.connection.execute(sql, params)]

    def verify(self) -> bool:
        previous = "0" * 64
        for event in self.events():
            canonical = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
            expected = hashlib.sha256(
                "|".join(
                    (
                        previous,
                        event.event_type,
                        event.schema_version,
                        event.occurred_at,
                        event.actor,
                        event.idempotency_key,
                        canonical,
                    )
                ).encode()
            ).hexdigest()
            if event.previous_hash != previous or event.event_hash != expected:
                return False
            previous = event.event_hash
        return True

    @staticmethod
    def _event(row: sqlite3.Row) -> Event:
        values = dict(row)
        values["payload"] = json.loads(values["payload"])
        return Event(**values)
