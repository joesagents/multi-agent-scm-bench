"""JSONL run records + SQLite index (`runs.db`).

Each (batch_id, team_id, scenario_id, seed, variant_id) cell writes:

- one JSONL file with the full RunRecord (header → ticks → result)
- one row in `runs.db` with summary metrics for fast aggregate queries

Layout::

    <store_root>/
        runs.db
        runs/<batch_id>/<team_id>/<scenario_id>__seed<seed>[__<variant_id>].jsonl

The JSONL format is line-delimited JSON; the first line is a `header`
record (run identity + scenario + bundle metadata), followed by one
`tick` record per simulated period, followed by one `result` record.
Replay (P2c follow-up) rehydrates RunRecord from this stream.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scm_bench.engine.environment import SimulationResult
from scm_bench.engine.state import TIER_NAMES
from scm_bench.metrics.composite import (
    bullwhip_ratio_variance,
)
from scm_bench.metrics.tokens import aggregate_tokens
from scm_bench.runner.harness import RunRecord, TickRecord

MIRROR_TEAM_ID = "__mirror_baseline__"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    batch_id     TEXT NOT NULL,
    team_id          TEXT NOT NULL,
    scenario_id      TEXT NOT NULL,
    seed             INTEGER NOT NULL,
    variant_id       TEXT NOT NULL DEFAULT '',
    jsonl_path       TEXT NOT NULL,
    horizon          INTEGER NOT NULL,
    total_cost       REAL,
    bullwhip_engine  REAL,
    bullwhip_ratio   REAL,
    chain_fill_rate  REAL,
    tokens_used      INTEGER,
    composite_score  REAL,
    created_at       TEXT NOT NULL,
    UNIQUE(batch_id, team_id, scenario_id, seed, variant_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_class ON runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_team  ON runs(batch_id, team_id);
CREATE INDEX IF NOT EXISTS idx_runs_cell  ON runs(batch_id, scenario_id, seed);
"""


@dataclass(frozen=True)
class RunRow:
    run_id: str
    batch_id: str
    team_id: str
    scenario_id: str
    seed: int
    variant_id: str
    jsonl_path: str
    horizon: int
    total_cost: float | None
    bullwhip_engine: float | None
    bullwhip_ratio: float | None
    chain_fill_rate: float | None
    tokens_used: int | None
    composite_score: float | None
    created_at: str


def _serialise_tick(tick: TickRecord) -> dict[str, Any]:
    return {
        "type": "tick",
        "t": tick.t,
        "consumer_demand": tick.consumer_demand,
        "observations": {
            role: obs.model_dump() for role, obs in tick.observations.items()
        },
        "decisions": {
            role: dec.model_dump() for role, dec in tick.decisions.items()
        },
        "messages_emitted": [msg.model_dump() for msg in tick.messages_emitted],
    }


def _serialise_result(result: SimulationResult) -> dict[str, Any]:
    return {"type": "result", **asdict(result)}


def _serialise_header(
    *,
    run_id: str,
    batch_id: str,
    team_id: str,
    scenario_id: str,
    seed: int,
    variant_id: str,
    horizon: int,
    bundle_meta: dict[str, Any] | None,
    sdk_version: str,
) -> dict[str, Any]:
    return {
        "type": "header",
        "run_id": run_id,
        "batch_id": batch_id,
        "team_id": team_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "variant_id": variant_id,
        "horizon": horizon,
        "sdk_version": sdk_version,
        "bundle_meta": bundle_meta or {},
        "created_at": _now(),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _factory_orders(record: RunRecord) -> list[int]:
    return [int(tick.decisions["factory"].order_qty) for tick in record.ticks]


def _consumer_demand(record: RunRecord) -> list[int]:
    return [int(tick.consumer_demand) for tick in record.ticks]


def _summary(record: RunRecord) -> dict[str, Any]:
    result = record.result
    if result is None:
        return {
            "total_cost": None,
            "bullwhip_engine": None,
            "bullwhip_ratio": None,
            "chain_fill_rate": None,
            "tokens_used": None,
        }
    factory_orders = _factory_orders(record)
    demand = _consumer_demand(record)
    tokens = aggregate_tokens(record)["chain"]
    return {
        "total_cost": float(result.total_cost),
        "bullwhip_engine": float(result.bullwhip),
        "bullwhip_ratio": float(bullwhip_ratio_variance(factory_orders, demand)),
        "chain_fill_rate": float(result.chain_fill_rate),
        "tokens_used": int(tokens),
    }


class RunStore:
    """JSONL writer + SQLite index for batch-run results."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.db"
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def jsonl_path_for(
        self,
        *,
        batch_id: str,
        team_id: str,
        scenario_id: str,
        seed: int,
        variant_id: str,
    ) -> Path:
        suffix = f"__{variant_id}" if variant_id else ""
        rel = (
            Path("runs")
            / batch_id
            / team_id
            / f"{scenario_id}__seed{seed}{suffix}.jsonl"
        )
        return self.root / rel

    def write_run(
        self,
        *,
        record: RunRecord,
        batch_id: str,
        team_id: str,
        variant_id: str = "",
        bundle_meta: dict[str, Any] | None = None,
        sdk_version: str,
    ) -> RunRow:
        path = self.jsonl_path_for(
            batch_id=batch_id,
            team_id=team_id,
            scenario_id=record.scenario_id,
            seed=record.seed,
            variant_id=variant_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        header = _serialise_header(
            run_id=record.run_id,
            batch_id=batch_id,
            team_id=team_id,
            scenario_id=record.scenario_id,
            seed=record.seed,
            variant_id=variant_id,
            horizon=record.horizon,
            bundle_meta=bundle_meta,
            sdk_version=sdk_version,
        )
        with path.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for tick in record.ticks:
                f.write(json.dumps(_serialise_tick(tick)) + "\n")
            if record.result is not None:
                f.write(json.dumps(_serialise_result(record.result)) + "\n")

        summary = _summary(record)
        row = RunRow(
            run_id=record.run_id,
            batch_id=batch_id,
            team_id=team_id,
            scenario_id=record.scenario_id,
            seed=record.seed,
            variant_id=variant_id,
            jsonl_path=str(path.relative_to(self.root)),
            horizon=record.horizon,
            total_cost=summary["total_cost"],
            bullwhip_engine=summary["bullwhip_engine"],
            bullwhip_ratio=summary["bullwhip_ratio"],
            chain_fill_rate=summary["chain_fill_rate"],
            tokens_used=summary["tokens_used"],
            composite_score=None,
            created_at=_now(),
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, batch_id, team_id, scenario_id, seed, variant_id,
                    jsonl_path, horizon, total_cost, bullwhip_engine,
                    bullwhip_ratio, chain_fill_rate, tokens_used,
                    composite_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.run_id, row.batch_id, row.team_id, row.scenario_id,
                    row.seed, row.variant_id, row.jsonl_path, row.horizon,
                    row.total_cost, row.bullwhip_engine, row.bullwhip_ratio,
                    row.chain_fill_rate, row.tokens_used, row.composite_score,
                    row.created_at,
                ),
            )
        return row

    def list_runs(
        self,
        *,
        batch_id: str | None = None,
        team_id: str | None = None,
        scenario_id: str | None = None,
    ) -> list[RunRow]:
        clauses, params = [], []
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if team_id is not None:
            clauses.append("team_id = ?")
            params.append(team_id)
        if scenario_id is not None:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        sql = "SELECT * FROM runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY team_id, scenario_id, seed, variant_id"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [RunRow(**dict(r)) for r in rows]

    def update_composite_score(self, run_id: str, composite_score: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET composite_score = ? WHERE run_id = ?",
                (composite_score, run_id),
            )

    def list_batches(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT batch_id FROM runs ORDER BY batch_id"
            ).fetchall()
        return [r["batch_id"] for r in rows]

    def read_jsonl(self, jsonl_path: Path | str) -> Iterable[dict[str, Any]]:
        path = Path(jsonl_path)
        if not path.is_absolute():
            path = self.root / path
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def import_jsonl(
        self,
        source_jsonl: Path | str,
        *,
        batch_id_override: str | None = None,
    ) -> RunRow:
        """Ingest a JSONL run file produced elsewhere (e.g. by the cluster worker).

        Reads header → ticks → result, copies the file into this store's
        canonical path, and inserts the summary row. Used to fold
        per-cell JSONL files from a distributed batch run into a
        local RunStore.

        If `batch_id_override` is given, the header's batch_id
        is rewritten — useful when relocating runs into a re-named
        batch-run.
        """
        src = Path(source_jsonl)
        lines = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        if not lines or lines[0].get("type") != "header":
            raise ValueError(f"{src}: missing or malformed header line")
        header = lines[0]
        ticks = [r for r in lines if r.get("type") == "tick"]
        results = [r for r in lines if r.get("type") == "result"]

        batch_id = batch_id_override or header["batch_id"]
        team_id = header["team_id"]
        scenario_id = header["scenario_id"]
        seed = int(header["seed"])
        variant_id = header.get("variant_id", "")

        dest = self.jsonl_path_for(
            batch_id=batch_id,
            team_id=team_id,
            scenario_id=scenario_id,
            seed=seed,
            variant_id=variant_id,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite the header if batch_id changed; otherwise copy bytes.
        if batch_id_override and batch_id_override != header["batch_id"]:
            header = {**header, "batch_id": batch_id_override}
            with dest.open("w") as f:
                f.write(json.dumps(header) + "\n")
                for tick in ticks:
                    f.write(json.dumps(tick) + "\n")
                for r in results:
                    f.write(json.dumps(r) + "\n")
        else:
            dest.write_text(src.read_text())

        # Recompute summary from ticks + result so the row matches the file.
        result_payload = results[-1] if results else None
        total_cost = float(result_payload["total_cost"]) if result_payload else None
        bullwhip_engine = (
            float(result_payload["bullwhip"]) if result_payload else None
        )
        chain_fill_rate = (
            float(result_payload["chain_fill_rate"]) if result_payload else None
        )
        if ticks and result_payload:
            factory_orders = [int(t["decisions"]["factory"]["order_qty"]) for t in ticks]
            demand = [int(t["consumer_demand"]) for t in ticks]
            bullwhip_ratio = float(bullwhip_ratio_variance(factory_orders, demand))
            tokens_used = sum(
                int(t["decisions"][role]["tokens_used"])
                for t in ticks
                for role in TIER_NAMES
            )
        else:
            bullwhip_ratio = None
            tokens_used = None

        row = RunRow(
            run_id=header["run_id"],
            batch_id=batch_id,
            team_id=team_id,
            scenario_id=scenario_id,
            seed=seed,
            variant_id=variant_id,
            jsonl_path=str(dest.relative_to(self.root)),
            horizon=int(header["horizon"]),
            total_cost=total_cost,
            bullwhip_engine=bullwhip_engine,
            bullwhip_ratio=bullwhip_ratio,
            chain_fill_rate=chain_fill_rate,
            tokens_used=tokens_used,
            composite_score=None,
            created_at=_now(),
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, batch_id, team_id, scenario_id, seed, variant_id,
                    jsonl_path, horizon, total_cost, bullwhip_engine,
                    bullwhip_ratio, chain_fill_rate, tokens_used,
                    composite_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.run_id, row.batch_id, row.team_id, row.scenario_id,
                    row.seed, row.variant_id, row.jsonl_path, row.horizon,
                    row.total_cost, row.bullwhip_engine, row.bullwhip_ratio,
                    row.chain_fill_rate, row.tokens_used, row.composite_score,
                    row.created_at,
                ),
            )
        return row


__all__ = ["MIRROR_TEAM_ID", "RunRow", "RunStore", "TIER_NAMES"]
