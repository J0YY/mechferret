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
            """
        )
        self.conn.commit()

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

