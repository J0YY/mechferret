"""Conversation transcript persistence + resume.

Each chat session is written to ``.mechferret/sessions/<id>.json`` (provider-
native messages + model + cost), so a crashed or quit session can be resumed
with ``/resume``. The full conversation, not just the research artifacts, is
recoverable.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path(".mechferret/sessions")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def new_session_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def is_valid_session_id(session_id: Any) -> bool:
    return isinstance(session_id, str) and SESSION_ID_PATTERN.fullmatch(session_id) is not None


def _session_path(session_id: str) -> Path:
    if not is_valid_session_id(session_id):
        raise ValueError("invalid session id")
    return SESSIONS_DIR / f"{session_id}.json"


def _safe_string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_limit(value: Any, default: int = 20) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


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
    path = _session_path(session_id)
    payload = {
        "id": session_id,
        "provider": provider,
        "model": model,
        "messages": _json_ready(messages if isinstance(messages, list) else []),
        "cost": _json_ready(cost if isinstance(cost, dict) else {}),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    tmp.replace(path)  # atomic
    return path


def load_session(session_id: str) -> dict[str, Any]:
    path = _session_path(session_id)
    if not path.exists():
        raise KeyError(f"no session {session_id!r} (looked in {SESSIONS_DIR})")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid session {session_id!r}: could not read JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid session {session_id!r}: expected JSON object")
    return payload


def list_sessions(limit: int = 20) -> list[SessionMeta]:
    limit = _safe_limit(limit)
    if not SESSIONS_DIR.is_dir():
        return []
    metas: list[SessionMeta] = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        msgs = d.get("messages", [])
        msgs = msgs if isinstance(msgs, list) else []
        cost = d.get("cost", {})
        cost = cost if isinstance(cost, dict) else {}
        session_id = _safe_string(d.get("id"), path.stem)
        if not is_valid_session_id(session_id):
            session_id = path.stem
        if not is_valid_session_id(session_id):
            continue
        turns = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")
        metas.append(SessionMeta(
            id=session_id,
            provider=_safe_string(d.get("provider")),
            model=_safe_string(d.get("model")),
            turns=turns,
            usd=_safe_float(cost.get("usd")),
            updated_at=_safe_string(d.get("updated_at")),
        ))
    metas.sort(key=lambda m: m.updated_at, reverse=True)
    return metas[:limit]


def latest_session_id() -> str | None:
    metas = list_sessions(1)
    return metas[0].id if metas else None
