from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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

    def record_experiments(self, model: str, task: str, hypotheses: list, results: list, code_version: str = "") -> dict:
        """Upsert experiments keyed by spec hash; count conclusion flips as drift.

        Scalable: one row per unique (model, task, probe, target). Re-running an
        experiment updates it in place and, if its significance or effect sign
        changed vs. the stored result, increments drift_count (e.g. after a model
        or code change). Returns {"recorded": n, "drifted": k}.
        """

        spec_to_hyp = {eid: h.statement for h in hypotheses for eid in getattr(h, "experiment_ids", [])}
        recorded = drifted = 0
        for r in results:
            if getattr(r, "status", "ran") != "ran":
                continue
            is_drift = self.record_experiment(
                model, task, r.probe, r.target, spec_to_hyp.get(r.spec_id, "screen"),
                r.effect_size, r.baseline, r.significant, r.reproduced, code_version, commit=False,
            )
            recorded += 1
            drifted += is_drift
        self.conn.commit()
        return {"recorded": recorded, "drifted": drifted}

    def record_experiment(self, model, task, probe, target, hypothesis, effect_size, control,
                          significant, reproduced, code_version="", commit=True) -> int:
        """Upsert one experiment by spec hash; return 1 if its conclusion drifted."""

        key = stable_id("exp", f"{model}|{task}|{probe}|{json.dumps(target, sort_keys=True)}")
        prior = self.conn.execute(
            "select effect_size, significant from experiments where id=?", (key,)
        ).fetchone()
        is_drift = 0
        if prior is not None and (
            self._sign(prior["effect_size"]) != self._sign(effect_size)
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
                (key, model, task, probe, json.dumps(target, sort_keys=True), hypothesis,
                 effect_size, control, int(significant), int(reproduced), verdict, is_drift, code_version, now, now),
            )
        else:
            self.conn.execute(
                "update experiments set effect_size=?, control=?, significant=?, reproduced=?, verdict=?, "
                "hypothesis=?, observed_count=observed_count+1, drift_count=drift_count+?, code_version=?, last_seen=? "
                "where id=?",
                (effect_size, control, int(significant), int(reproduced), verdict, hypothesis,
                 is_drift, code_version, now, key),
            )
        if commit:
            self.conn.commit()
        return is_drift

    def clear_experiments_and_mechanisms(self) -> None:
        """Wipe the experiment ledger + mechanisms (used to replay a demo cleanly)."""

        self.conn.execute("delete from experiments")
        self.conn.execute("delete from mechanisms")
        self.conn.commit()

    def experiments_by_hypothesis(self, limit: int = 200) -> dict:
        rows = self.conn.execute(
            "select * from experiments order by last_seen desc limit ?", (limit,)
        ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["hypothesis"], []).append(dict(row))
        return grouped

    def record_mechanisms(self, model: str, mechanisms: list[dict]) -> int:
        """Persist confirmed mechanisms so findings compound across sessions."""

        rows = []
        for m in mechanisms:
            statement = m.get("statement", "")
            if not statement:
                continue
            rows.append((
                stable_id("mech", f"{model}:{statement}"),
                statement,
                model,
                float(m.get("effect_size", 0.0)),
                float(m.get("reproducibility", 0.0)),
                float(m.get("novelty", 0.0)),
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
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_sources(self, sources: list[Source]) -> None:
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
            [
                (
                    source.id,
                    source.title,
                    source.url,
                    source.kind,
                    source.text,
                    json.dumps(source.metadata, sort_keys=True),
                    source.created_at,
                )
                for source in sources
            ],
        )
        self.conn.commit()

    def record_run(self, run: ResearchRun) -> None:
        self.conn.execute(
            """
            insert or replace into runs (id, question, answer, metrics_json, artifacts_json, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.question,
                run.answer,
                json.dumps(run.metrics, sort_keys=True),
                json.dumps(run.artifacts, sort_keys=True),
                run.created_at,
            ),
        )
        self.conn.executemany(
            """
            insert or replace into claims
            (id, run_id, text, citations_json, source_ids_json, confidence, support_score, stance, quality_flags_json, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    claim.id,
                    run.run_id,
                    claim.text,
                    json.dumps(claim.citations),
                    json.dumps(claim.source_ids),
                    claim.confidence,
                    claim.support_score,
                    claim.stance,
                    json.dumps(claim.quality_flags),
                    run.created_at,
                )
                for claim in run.claims
            ],
        )
        self.conn.commit()

    def recall_sources(self, question: str, limit: int = 3) -> list[Source]:
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
        sources = [
            Source(
                id=stable_id("mem", f"{row['created_at']}:{row['text']}"),
                title=f"Memory from prior run: {row['question'][:72]}",
                text=(
                    f"Prior claim: {row['text']}\n"
                    f"Confidence: {row['confidence']:.2f}; support: {row['support_score']:.2f}; "
                    f"recorded_at: {row['created_at']}"
                ),
                url=f"memory://{row['created_at']}",
                kind="memory",
                created_at=utc_now(),
            )
            for row in rows
        ]
        index = BM25Index.from_sources(sources)
        selected_ids = {chunk.source_id for chunk in index.search(question, limit=limit)}
        return [source for source in sources if source.id in selected_ids][:limit]

