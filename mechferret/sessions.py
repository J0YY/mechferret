"""Conversation transcript persistence + resume.

Each chat session is written to ``.mechferret/sessions/<id>.json`` (provider-
native messages + model + cost), so a crashed or quit session can be replayed
with ``/resume``. Reproducibility: the full conversation, not just the research
artifacts, is recoverable.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path(".mechferret/sessions")


def new_session_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass(slots=True)
class SessionMeta:
    id: str
    provider: str
    model: str
    turns: int
    usd: float
    updated_at: str


def save_session(session_id: str, provider: str, model: str, messages: list, cost: dict) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{session_id}.json"
    payload = {
        "id": session_id,
        "provider": provider,
        "model": model,
        "messages": messages,
        "cost": cost,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)  # atomic
    return path


def load_session(session_id: str) -> dict[str, Any]:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise KeyError(f"no session {session_id!r} (looked in {SESSIONS_DIR})")
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions(limit: int = 20) -> list[SessionMeta]:
    if not SESSIONS_DIR.exists():
        return []
    metas: list[SessionMeta] = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        msgs = d.get("messages", [])
        turns = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")
        metas.append(SessionMeta(
            id=d.get("id", path.stem),
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            turns=turns,
            usd=float(d.get("cost", {}).get("usd", 0.0)),
            updated_at=d.get("updated_at", ""),
        ))
    metas.sort(key=lambda m: m.updated_at, reverse=True)
    return metas[:limit]


def latest_session_id() -> str | None:
    metas = list_sessions(1)
    return metas[0].id if metas else None
