from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .models import Claim, ResearchRun, Source, utc_now
from .retrieval import BM25Index
from .text import stable_id


class ResearchMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists sources (
              id text primary key,
              title text not null,
              url text,
              kind text,
              text text not null,
              metadata_json text not null,
              created_at text not null
            );
            create table if not exists claims (
              id text primary key,
              run_id text,
              text text not null,
              citations_json text not null,
              source_ids_json text not null,
              confidence real not null,
              support_score real not null,
              stance text not null,
              quality_flags_json text not null,
              created_at text not null
            );
            create table if not exists runs (
              id text primary key,
              question text not null,
              answer text not null,
              metrics_json text not null,
              artifacts_json text not null,
              created_at text not null
            );
            create table if not exists mechanisms (
              id text primary key,
              statement text not null,
              model text,
              effect_size real,
              reproducibility real,
              novelty real,
              created_at text not null
            );
            create table if not exists experiments (
              id text primary key,           -- hash(model|task|probe|target): one row per unique spec
              model text, task text, probe text, target_json text,
              hypothesis text, effect_size real, control real,
              significant integer, reproduced integer, verdict text,
              observed_count integer not null default 1,
              drift_count integer not null default 0,
              code_version text,
              first_seen text not null, last_seen text not null
            );
            """
        )
        self.conn.commit()

    @staticmethod
    def _sign(x: float) -> int:
        return (x > 0) - (x < 0)

    @staticmethod
    def _limit(value: Any, default: int = 12) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, parsed)

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(ResearchMemory._json_ready(value), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return json.dumps(str(value))

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(key): ResearchMemory._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ResearchMemory._json_ready(item) for item in value]
        return str(value)

    @staticmethod
    def _rows(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _field(row: Any, name: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(name, default)
        return getattr(row, name, default)

    def record_experiments(self, model: str, task: str, hypotheses: list, results: list, code_version: str = "") -> dict:
        """Upsert experiments keyed by spec hash; count conclusion flips as drift.

        Scalable: one row per unique (model, task, probe, target). Re-running an
        experiment updates it in place and, if its significance or effect sign
        changed vs. the stored result, increments drift_count (e.g. after a model
        or code change). Returns {"recorded": n, "drifted": k}.
        """

        spec_to_hyp: dict[str, str] = {}
        for h in self._rows(hypotheses):
            statement = self._field(h, "statement", "")
            if not isinstance(statement, str):
                continue
            for eid in self._rows(self._field(h, "experiment_ids", [])):
                if isinstance(eid, str) and eid:
                    spec_to_hyp[eid] = statement
        recorded = drifted = 0
        for r in self._rows(results):
            if self._field(r, "status", "ran") != "ran":
                continue
            spec_id = self._field(r, "spec_id", "")
            probe = self._field(r, "probe", "")
            if not spec_id or not probe:
                continue
            is_drift = self.record_experiment(
                model,
                task,
                probe,
                self._field(r, "target", {}),
                spec_to_hyp.get(spec_id, "screen"),
                self._field(r, "effect_size", 0.0),
                self._field(r, "baseline", 0.0),
                self._field(r, "significant", False),
                self._field(r, "reproduced", False),
                code_version,
                commit=False,
            )
            recorded += 1
            drifted += is_drift
        self.conn.commit()
        return {"recorded": recorded, "drifted": drifted}

    def record_experiment(self, model, task, probe, target, hypothesis, effect_size, control,
                          significant, reproduced, code_version="", commit=True) -> int:
        """Upsert one experiment by spec hash; return 1 if its conclusion drifted."""

        effect = self._finite_float(effect_size)
        baseline = self._finite_float(control)
        key = stable_id("exp", f"{model}|{task}|{probe}|{self._safe_json(target)}")
        prior = self.conn.execute(
            "select effect_size, significant from experiments where id=?", (key,)
        ).fetchone()
        is_drift = 0
        if prior is not None and (
            self._sign(self._finite_float(prior["effect_size"])) != self._sign(effect)
            or bool(prior["significant"]) != bool(significant)
        ):
            is_drift = 1
        verdict = "good" if (significant and reproduced) else "weak"
        now = utc_now()
        if prior is None:
            self.conn.execute(
                "insert into experiments (id, model, task, probe, target_json, hypothesis, effect_size, control, "
                "significant, reproduced, verdict, observed_count, drift_count, code_version, first_seen, last_seen) "
                "values (?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?)",
                (key, str(model), str(task), str(probe), self._safe_json(target), str(hypothesis),
                 effect, baseline, int(bool(significant)), int(bool(reproduced)), verdict, is_drift, str(code_version), now, now),
            )
        else:
            self.conn.execute(
                "update experiments set effect_size=?, control=?, significant=?, reproduced=?, verdict=?, "
                "hypothesis=?, observed_count=observed_count+1, drift_count=drift_count+?, code_version=?, last_seen=? "
                "where id=?",
                (effect, baseline, int(bool(significant)), int(bool(reproduced)), verdict, str(hypothesis),
                 is_drift, str(code_version), now, key),
            )
        if commit:
            self.conn.commit()
        return is_drift

    def clear_experiments_and_mechanisms(self) -> None:
        """Wipe the experiment ledger + mechanisms before a presenter walkthrough."""

        self.conn.execute("delete from experiments")
        self.conn.execute("delete from mechanisms")
        self.conn.commit()

    def experiments_by_hypothesis(self, limit: int = 200) -> dict:
        rows = self.conn.execute(
            "select * from experiments order by last_seen desc limit ?", (self._limit(limit, 200),)
        ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["hypothesis"], []).append(dict(row))
        return grouped

    def record_mechanisms(self, model: str, mechanisms: list[dict]) -> int:
        """Persist confirmed mechanisms so findings compound across sessions."""

        rows = []
        if not isinstance(mechanisms, list):
            mechanisms = []
        for m in mechanisms:
            if not isinstance(m, dict):
                continue
            statement = m.get("statement", "")
            if not isinstance(statement, str) or not statement.strip():
                continue
            model_name = model if isinstance(model, str) else str(model)
            rows.append((
                stable_id("mech", f"{model_name}:{statement}"),
                statement.strip(),
                model_name,
                self._finite_float(m.get("effect_size", 0.0)),
                self._finite_float(m.get("reproducibility", 0.0)),
                self._finite_float(m.get("novelty", 0.0)),
                utc_now(),
            ))
        if rows:
            self.conn.executemany(
                "insert or replace into mechanisms (id, statement, model, effect_size, reproducibility, novelty, created_at) "
                "values (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def recent_mechanisms(self, limit: int = 12) -> list[dict]:
        rows = self.conn.execute(
            "select statement, model, effect_size, reproducibility, novelty, created_at "
            "from mechanisms order by created_at desc limit ?",
            (self._limit(limit, 12),),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_sources(self, sources: list[Source]) -> None:
        rows = []
        for source in self._rows(sources):
            source_id = self._field(source, "id", "")
            text = self._field(source, "text", "")
            if not source_id or not isinstance(text, str) or not text:
                continue
            rows.append(
                (
                    str(source_id),
                    str(self._field(source, "title", "")),
                    str(self._field(source, "url", "")),
                    str(self._field(source, "kind", "document")),
                    text,
                    self._safe_json(self._field(source, "metadata", {})),
                    str(self._field(source, "created_at", utc_now())),
                )
            )
        if not rows:
            return
        self.conn.executemany(
            """
            insert into sources (id, title, url, kind, text, metadata_json, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(id) do update set
              title=excluded.title,
              url=excluded.url,
              kind=excluded.kind,
              text=excluded.text,
              metadata_json=excluded.metadata_json
            """,
            rows,
        )
        self.conn.commit()

    def record_run(self, run: ResearchRun) -> None:
        run_id = self._field(run, "run_id", "")
        if not run_id:
            return
        created_at = str(self._field(run, "created_at", utc_now()))
        self.conn.execute(
            """
            insert or replace into runs (id, question, answer, metrics_json, artifacts_json, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                str(self._field(run, "question", "")),
                str(self._field(run, "answer", "")),
                self._safe_json(self._field(run, "metrics", {})),
                self._safe_json(self._field(run, "artifacts", {})),
                created_at,
            ),
        )
        claim_rows = []
        for claim in self._rows(self._field(run, "claims", [])):
            claim_id = self._field(claim, "id", "")
            text = self._field(claim, "text", "")
            if not claim_id or not isinstance(text, str) or not text:
                continue
            claim_rows.append(
                (
                    str(claim_id),
                    str(run_id),
                    text,
                    self._safe_json(self._field(claim, "citations", [])),
                    self._safe_json(self._field(claim, "source_ids", [])),
                    self._finite_float(self._field(claim, "confidence", 0.0)),
                    self._finite_float(self._field(claim, "support_score", 0.0)),
                    str(self._field(claim, "stance", "finding")),
                    self._safe_json(self._field(claim, "quality_flags", [])),
                    created_at,
                )
            )
        self.conn.executemany(
            """
            insert or replace into claims
            (id, run_id, text, citations_json, source_ids_json, confidence, support_score, stance, quality_flags_json, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            claim_rows,
        )
        self.conn.commit()

    def recall_sources(self, question: str, limit: int = 3) -> list[Source]:
        limit = self._limit(limit, 3)
        if limit <= 0:
            return []
        rows = self.conn.execute(
            """
            select c.text, c.confidence, c.support_score, c.created_at, r.question
            from claims c
            join runs r on r.id = c.run_id
            order by c.created_at desc
            limit 200
            """
        ).fetchall()
        if not rows:
            return []
        sources: list[Source] = []
        for row in rows:
            text = row["text"] if isinstance(row["text"], str) else ""
            created_at = row["created_at"] if isinstance(row["created_at"], str) else utc_now()
            question_text = row["question"] if isinstance(row["question"], str) else ""
            if not text.strip():
                continue
            sources.append(Source(
                id=stable_id("mem", f"{row['created_at']}:{row['text']}"),
                title=f"Memory from prior run: {question_text[:72]}",
                text=(
                    f"Prior claim: {text}\n"
                    f"Confidence: {self._finite_float(row['confidence']):.2f}; support: {self._finite_float(row['support_score']):.2f}; "
                    f"recorded_at: {created_at}"
                ),
                url=f"memory://{created_at}",
                kind="memory",
                created_at=utc_now(),
            ))
        if not sources:
            return []
        index = BM25Index.from_sources(sources)
        selected_ids = {chunk.source_id for chunk in index.search(question, limit=limit)}
        return [source for source in sources if source.id in selected_ids][:limit]
