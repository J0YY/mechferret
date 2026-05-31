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
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(".mechferret/mcp.json")
_PREFIX = "mcp__"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_connected: dict[str, "MCPClient"] | None = None
_specs: list[dict] | None = None
_lock = threading.Lock()


@dataclass(slots=True)
class ServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]


def _valid_name(value: Any) -> bool:
    return isinstance(value, str) and _NAME_PATTERN.fullmatch(value) is not None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, str)}


def _read_config_payload() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"servers": {}}
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"servers": {}}
    if not isinstance(payload, dict):
        return {"servers": {}}
    servers = payload.get("servers", {})
    if not isinstance(servers, dict):
        payload["servers"] = {}
    return payload


def load_servers() -> list[ServerConfig]:
    payload = _read_config_payload()
    servers = []
    for name, cfg in payload.get("servers", {}).items():
        if not _valid_name(name) or not isinstance(cfg, dict):
            continue
        command = cfg.get("command", "")
        if not isinstance(command, str):
            continue
        servers.append(ServerConfig(
            name=name,
            command=command.strip(),
            args=_string_list(cfg.get("args", [])),
            env=_string_dict(cfg.get("env", {})),
        ))
    return servers


def add_server(name: str, command: str, args: list[str] | None = None) -> Path:
    if not _valid_name(name):
        raise ValueError("invalid MCP server name")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("invalid MCP command")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_config_payload()
    servers = payload.setdefault("servers", {})
    servers[name] = {"command": command.strip(), "args": _string_list(args or [])}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
        import select

        while True:
            try:
                ready, _, _ = select.select([self.proc.stdout], [], [], 60)
                if not ready:
                    raise RuntimeError(f"MCP server {self.cfg.name} timed out waiting for {method}")
            except (OSError, ValueError):
                pass  # select unavailable for this stream/platform; fall back to blocking read
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
                if not isinstance(tool, dict):
                    continue
                tool_name = tool.get("name")
                if not _valid_name(tool_name):
                    continue
                full = f"{_PREFIX}{cfg.name}__{tool_name}"
                parameters = tool.get("inputSchema") or {"type": "object", "properties": {}, "required": []}
                if not isinstance(parameters, dict):
                    parameters = {"type": "object", "properties": {}, "required": []}
                _specs.append({
                    "name": full,
                    "description": f"[mcp:{cfg.name}] " + (tool.get("description") if isinstance(tool.get("description"), str) else ""),
                    "parameters": parameters,
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
    connected = _connected  # snapshot to avoid a concurrent reset() nulling it
    if not full_name.startswith(_PREFIX) or connected is None:
        return json.dumps({"error": f"no MCP tool {full_name}"})
    rest = full_name[len(_PREFIX):]
    server, _, tool = rest.partition("__")
    client = connected.get(server)
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
