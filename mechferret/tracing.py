from __future__ import annotations

import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen


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


class TraceRecorder:
    def __init__(self, run_id: str, out_dir: str | Path) -> None:
        self.run_id = run_id
        self.trace_id = uuid.uuid4().hex
        self.out_dir = Path(out_dir)
        self.path: Path | None = None
        self.endpoint = os.getenv("RAINDROP_ENDPOINT", "http://127.0.0.1:5899/v1/traces")
        self.raindrop_enabled = bool(os.getenv("RAINDROP_LOCAL_DEBUGGER") or os.getenv("MECHFERRET_RAINDROP"))
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.out_dir / "trace.jsonl"
        except OSError:
            self.path = None

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        span_id = uuid.uuid4().hex[:16]
        start = time.perf_counter()
        self.emit("start", name, span_id, attrs)
        try:
            yield
        except Exception as exc:
            attrs = {**attrs, "error": type(exc).__name__, "message": str(exc)}
            self.emit("error", name, span_id, attrs, elapsed_ms=(time.perf_counter() - start) * 1000)
            raise
        else:
            self.emit("end", name, span_id, attrs, elapsed_ms=(time.perf_counter() - start) * 1000)

    def event(self, name: str, **attrs: Any) -> None:
        self.emit("event", name, uuid.uuid4().hex[:16], attrs)

    def emit(self, phase: str, name: str, span_id: str, attrs: dict[str, Any], elapsed_ms: float | None = None) -> None:
        record = {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "span_id": span_id,
            "phase": phase,
            "name": name,
            "time_unix_ms": int(time.time() * 1000),
            "elapsed_ms": round(elapsed_ms or 0.0, 3),
            "attributes": _json_ready(attrs),
        }
        try:
            line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except (OSError, TypeError, ValueError):
            return
        if self.raindrop_enabled:
            self._mirror_to_raindrop(record)

    def _mirror_to_raindrop(self, record: dict[str, Any]) -> None:
        payload = json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [record]}]}]}, allow_nan=False).encode("utf-8")
        request = Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=0.2):
                pass
        except (OSError, URLError, TimeoutError):
            pass
