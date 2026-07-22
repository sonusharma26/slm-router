"""SQLite-backed trace store with idempotent inserts and oracle-matrix queries.

The append-only `runs` table is the queryable source of truth; metrics read from
here, observability backends (Langfuse/Phoenix) are for drill-down only.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from slm_router.trace.models import COLUMNS, SCHEMA_SQL, RunTrace


def _to_row(t: RunTrace) -> tuple:
    d = t.model_dump()
    out = []
    for c in COLUMNS:
        v = d[c]
        if isinstance(v, bool):  # sqlite stores bool as int
            v = int(v)
        out.append(v)
    return tuple(out)


def _from_row(row: sqlite3.Row) -> RunTrace:
    d = dict(row)
    if d.get("correct") is not None:
        d["correct"] = bool(d["correct"])
    return RunTrace.model_validate(d)


class TraceStore:
    def __init__(self, path: str | Path = "data/traces/router.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def insert_run(self, t: RunTrace) -> None:
        self.insert_many([t])

    def insert_many(self, traces: Iterable[RunTrace]) -> None:
        placeholders = ",".join("?" for _ in COLUMNS)
        cols = ",".join(COLUMNS)
        sql = f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({placeholders})"
        self.conn.executemany(sql, [_to_row(t) for t in traces])
        self.conn.commit()

    def has_run(self, item_id: str, model: str, config_hash: str, run_kind: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM runs WHERE item_id=? AND model=? AND config_hash=? AND run_kind=? LIMIT 1",
            (item_id, model, config_hash, run_kind),
        )
        return cur.fetchone() is not None

    def query(self, **filters) -> list[RunTrace]:
        clauses, params = [], []
        for k, v in filters.items():
            clauses.append(f"{k}=?")
            params.append(int(v) if isinstance(v, bool) else v)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self.conn.execute(f"SELECT * FROM runs{where}", params)
        return [_from_row(r) for r in cur.fetchall()]

    def get_oracle_matrix(
        self, dataset: str | None = None
    ) -> dict[str, dict[str, RunTrace]]:
        """item_id -> {model -> RunTrace} over run_kind='oracle'."""
        sql = "SELECT * FROM runs WHERE run_kind='oracle'"
        params: list = []
        if dataset:
            sql += " AND dataset=?"
            params.append(dataset)
        matrix: dict[str, dict[str, RunTrace]] = defaultdict(dict)
        for r in self.conn.execute(sql, params).fetchall():
            t = _from_row(r)
            matrix[t.item_id][t.model] = t
        return dict(matrix)

    def best_model_per_item(
        self, objective: str = "score", matrix: dict | None = None
    ) -> dict[str, str]:
        """Oracle choice per item. objective: 'score' or 'score_per_cost'."""
        matrix = matrix if matrix is not None else self.get_oracle_matrix()
        best: dict[str, str] = {}
        for item_id, by_model in matrix.items():
            def key(m: str) -> float:
                t = by_model[m]
                if objective == "score_per_cost":
                    return t.score / max(t.cost_usd, 1e-9)
                return t.score
            best[item_id] = max(by_model, key=key)
        return best

    def to_parquet(self, path: str | Path) -> None:
        import pandas as pd

        df = pd.read_sql_query("SELECT * FROM runs", self.conn)
        df.to_parquet(str(path))

    def close(self) -> None:
        self.conn.close()
