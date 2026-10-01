"""Append-only, hash-chained SQLite events; serialized writers and crash-safe claims."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from inference_control.util import canonical


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
    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
          event_type TEXT NOT NULL, schema_version TEXT NOT NULL, occurred_at TEXT NOT NULL,
          actor TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
          previous_hash TEXT NOT NULL, event_hash TEXT UNIQUE NOT NULL);
        CREATE INDEX IF NOT EXISTS event_type_sequence ON events(event_type, sequence);
        CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
        """)
        self.connection.execute("INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)",
                                (self.SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()))
        self.connection.commit()

    def _append_locked(self, event_type, payload, actor, idempotency_key, schema_version="2.0"):
        existing = self.connection.execute("SELECT * FROM events WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        if existing:
            return self._event(existing)
        last = self.connection.execute("SELECT event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        previous = last[0] if last else "0"*64
        occurred, data = datetime.now(timezone.utc).isoformat(), canonical(payload)
        event_hash = hashlib.sha256("|".join((previous, event_type, schema_version, occurred, actor,
                                             idempotency_key, data)).encode()).hexdigest()
        self.connection.execute("""INSERT INTO events(event_id,event_type,schema_version,occurred_at,
            actor,idempotency_key,payload,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?)""",
            (event_hash[:32], event_type, schema_version, occurred, actor, idempotency_key, data, previous, event_hash))
        return self.get(event_hash[:32])

    def append(self, event_type: str, payload: dict[str, Any], *, actor: str, idempotency_key: str,
               schema_version: str = "2.0") -> Event:
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                event = self._append_locked(event_type, payload, actor, idempotency_key, schema_version)
                self.connection.commit()
                return event
            except BaseException:
                self.connection.rollback()
                raise

    def append_batch(self, entries: list[tuple[str, dict, str]], *, actor: str) -> list[Event]:
        """Commit correlated records together (e.g. label + superseded map)."""
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                result = [self._append_locked(kind, payload, actor, key) for kind,payload,key in entries]
                self.connection.commit()
                return result
            except BaseException:
                self.connection.rollback()
                raise

    def get(self, event_id: str) -> Event:
        with self.lock:
            row = self.connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None: raise KeyError(event_id)
        return self._event(row)

    def events(self, event_type: str | None = None, *, after: int = 0) -> list[Event]:
        with self.lock:
            if event_type:
                rows = self.connection.execute("SELECT * FROM events WHERE event_type=? AND sequence>? ORDER BY sequence", (event_type,after)).fetchall()
            else:
                rows = self.connection.execute("SELECT * FROM events WHERE sequence>? ORDER BY sequence", (after,)).fetchall()
        return [self._event(r) for r in rows]

    def verify(self) -> bool:
        previous = "0"*64
        for event in self.events():
            expected = hashlib.sha256("|".join((previous,event.event_type,event.schema_version,event.occurred_at,
                                                event.actor,event.idempotency_key,canonical(event.payload))).encode()).hexdigest()
            if event.previous_hash != previous or event.event_hash != expected: return False
            previous = event.event_hash
        return True

    def close(self):
        with self.lock: self.connection.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    @staticmethod
    def _event(row):
        values = dict(row)
        values["payload"] = json.loads(values["payload"])
        return Event(**values)
