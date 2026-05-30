"""Minimal Model Context Protocol (MCP) stdio client.

Lets MechFerret pull tools from external MCP servers (a model server, a paper
DB, a circuit explorer) without baking each into ``tools.py``. Servers are
configured in ``.mechferret/mcp.json``; their tools are exposed to the agent as
``mcp__<server>__<tool>`` and flow through the same permission gate.

Conservative by design: if nothing is configured, every entry point returns
empty/no-op so the normal path is never affected. Connections are cached.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(".mechferret/mcp.json")
_PREFIX = "mcp__"
_connected: dict[str, "MCPClient"] | None = None
_specs: list[dict] | None = None
_lock = threading.Lock()


@dataclass(slots=True)
class ServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]


def load_servers() -> list[ServerConfig]:
    if not CONFIG_PATH.exists():
        return []
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    servers = []
    for name, cfg in payload.get("servers", {}).items():
        servers.append(ServerConfig(
            name=name,
            command=cfg.get("command", ""),
            args=list(cfg.get("args", [])),
            env=dict(cfg.get("env", {})),
        ))
    return servers


def add_server(name: str, command: str, args: list[str] | None = None) -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"servers": {}}
    if CONFIG_PATH.exists():
        try:
            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {"servers": {}}
    payload.setdefault("servers", {})[name] = {"command": command, "args": args or []}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reset()
    return CONFIG_PATH


class MCPClient:
    """Newline-delimited JSON-RPC 2.0 over a subprocess's stdio."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        import os

        env = {**os.environ, **cfg.env}
        self.proc = subprocess.Popen(
            [cfg.command, *cfg.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            bufsize=1,
        )
        self._id = 0
        self._initialize()

    def _rpc(self, method: str, params: dict | None = None, notify: bool = False) -> dict:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self._id += 1
            msg["id"] = self._id
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return {}
        # Read until we get a response with our id (skip notifications).
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server {self.cfg.name} closed the connection")
            data = json.loads(line)
            if data.get("id") == self._id:
                if "error" in data:
                    raise RuntimeError(f"MCP {method} error: {data['error']}")
                return data.get("result", {})

    def _initialize(self) -> None:
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mechferret", "version": "0.1.0"},
        })
        self._rpc("notifications/initialized", {}, notify=True)

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(result)

    def close(self) -> None:
        try:
            self.proc.terminate()
        except OSError:
            pass


def _connect_all() -> None:
    global _connected, _specs
    _connected = {}
    _specs = []
    for cfg in load_servers():
        if not cfg.command:
            continue
        try:
            client = MCPClient(cfg)
            _connected[cfg.name] = client
            for tool in client.list_tools():
                full = f"{_PREFIX}{cfg.name}__{tool['name']}"
                _specs.append({
                    "name": full,
                    "description": f"[mcp:{cfg.name}] " + tool.get("description", ""),
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}, "required": []},
                })
        except Exception:  # noqa: BLE001 - a bad server must not break the agent
            continue


def tool_specs() -> list[dict]:
    """Tool specs for all configured MCP servers (cached, [] if none)."""

    global _specs
    if _specs is None:
        with _lock:
            if _specs is None:
                _connect_all()
    return _specs or []


def call(full_name: str, args: dict) -> str:
    if _connected is None:
        tool_specs()
    if not full_name.startswith(_PREFIX) or _connected is None:
        return json.dumps({"error": f"no MCP tool {full_name}"})
    rest = full_name[len(_PREFIX):]
    server, _, tool = rest.partition("__")
    client = _connected.get(server)
    if not client:
        return json.dumps({"error": f"MCP server {server} not connected"})
    try:
        return client.call_tool(tool, args)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"MCP call failed: {exc}"})


def reset() -> None:
    global _connected, _specs
    with _lock:
        if _connected:
            for client in _connected.values():
                client.close()
        _connected = None
        _specs = None


def status() -> dict[str, Any]:
    servers = load_servers()
    return {
        "configured": [s.name for s in servers],
        "tool_count": len(tool_specs()),
    }
