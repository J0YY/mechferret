"""Generic SLURM cluster compute backend.

A sibling to ``modal_app.py`` for teams that already have a SLURM cluster
instead of (or in addition to) Modal. It runs the heavy work the standard way:
non-interactive ``ssh`` to a login host, ``srun`` with your resource flags,
inside your environment setup, in your remote project directory -- then copies
the resulting dossier back.

Everything is configured by environment variables (or ``~/.mechferret/cluster.json``)
so there is nothing host-specific baked in. The defaults mirror the common
``REMOTE_HOST`` / ``SLURM_*`` convention, but any SSH alias + SLURM partition
works.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_INTERP_MODEL

CONFIG_PATHS = (
    Path(os.getenv("MECHFERRET_CLUSTER_CONFIG", "")) if os.getenv("MECHFERRET_CLUSTER_CONFIG") else None,
    Path(".mechferret/cluster.json"),
    Path.home() / ".mechferret" / "cluster.json",
)


def _safe_string(value: Any, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _safe_int(value: Any, default: int, *, min_value: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= min_value else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _read_config_payload() -> dict[str, Any]:
    for path in CONFIG_PATHS:
        if path and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


@dataclass(slots=True)
class ClusterConfig:
    host: str = ""  # SSH alias or user@host (non-interactive SSH must already work)
    partition: str = ""
    gres: str = "gpu:1"  # e.g. gpu:1, gpu:a100:1
    cpus: int = 8
    mem: str = "32G"
    time: str = "02:00:00"
    remote_project_dir: str = ""  # where mechferret is installed on the cluster
    remote_setup: str = ""  # e.g. 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate mf'
    python: str = "python3"
    git_pull: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.host and self.remote_project_dir)


def load_cluster_config() -> ClusterConfig:
    """Env vars take precedence, then the first JSON config file found."""

    payload = _read_config_payload()

    def pick(env_name: str, key: str, default):
        env_value = os.getenv(env_name)
        if env_value not in (None, ""):
            return env_value
        value = payload.get(key)
        return value if value not in (None, "") else default

    return ClusterConfig(
        host=_safe_string(pick("REMOTE_HOST", "host", "")),
        partition=_safe_string(pick("SLURM_PARTITION", "partition", "")),
        gres=_safe_string(pick("SLURM_GRES", "gres", "gpu:1"), "gpu:1"),
        cpus=_safe_int(pick("SLURM_CPUS", "cpus", 8), 8),
        mem=_safe_string(pick("SLURM_MEM", "mem", "32G"), "32G"),
        time=_safe_string(pick("SLURM_TIME", "time", "02:00:00"), "02:00:00"),
        remote_project_dir=_safe_string(pick("REMOTE_PROJECT_DIR", "remote_project_dir", "")),
        remote_setup=_safe_string(pick("REMOTE_RUN_SETUP", "remote_setup", "")),
        python=_safe_string(pick("REMOTE_PYTHON", "python", "python3"), "python3"),
        git_pull=_safe_bool(pick("REMOTE_GIT_PULL", "git_pull", "")),
    )


def cluster_status(cfg: ClusterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_cluster_config()
    status = asdict(cfg)
    status["configured"] = cfg.configured
    status["ssh_ok"] = _ssh_reachable(cfg.host) if cfg.host else False
    return status


def _ssh_reachable(host: str, timeout: int = 8) -> bool:
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, "echo ok"],
            capture_output=True,
            text=True,
            timeout=timeout + 4,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def build_remote_command(
    cfg: ClusterConfig,
    skill: str | None,
    question: str,
    task: str | None,
    model: str,
    remote_out: str,
) -> str:
    """The inner command run on the compute node (under srun)."""

    parts = []
    remote_setup = _safe_string(getattr(cfg, "remote_setup", ""))
    remote_project_dir = _safe_string(getattr(cfg, "remote_project_dir", ""))
    python = _safe_string(getattr(cfg, "python", "python3"), "python3")
    if remote_setup:
        parts.append(remote_setup)
    parts.append(f"cd {shlex.quote(remote_project_dir)}")
    if _safe_bool(getattr(cfg, "git_pull", False)):
        parts.append("git pull --ff-only")
    discover = [python, "-m", "mechferret", "discover", "--backend", "transformer_lens", "--model", str(model or DEFAULT_INTERP_MODEL), "--out", str(remote_out)]
    if skill:
        discover += ["--skill", str(skill)]
    if task:
        discover += ["--task", str(task)]
    if question:
        discover.append(str(question))
    parts.append(" ".join(shlex.quote(token) for token in discover))
    return " && ".join(parts)


def build_srun_invocation(cfg: ClusterConfig, remote_command: str) -> list[str]:
    srun = ["srun"]
    partition = _safe_string(getattr(cfg, "partition", ""))
    gres = _safe_string(getattr(cfg, "gres", ""))
    if partition:
        srun += ["--partition", partition]
    if gres:
        srun += ["--gres", gres]
    srun += [
        "--cpus-per-task",
        str(_safe_int(getattr(cfg, "cpus", 8), 8)),
        "--mem",
        _safe_string(getattr(cfg, "mem", "32G"), "32G"),
        "--time",
        _safe_string(getattr(cfg, "time", "02:00:00"), "02:00:00"),
    ]
    srun += ["bash", "-lc", str(remote_command)]
    inner = " ".join(shlex.quote(token) for token in srun)
    return ["ssh", _safe_string(getattr(cfg, "host", "")), inner]


def dispatch_discovery_cluster(
    question: str = "",
    skill: str | None = None,
    task: str | None = None,
    model: str = DEFAULT_INTERP_MODEL,
    out_dir: str | Path = "runs/cluster",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the discovery loop on a SLURM cluster; fall back to local otherwise.

    Returns ``{"backend": "cluster"|"local", ...}``. With ``dry_run`` the SSH/srun
    command is returned without executing anything.
    """

    cfg = load_cluster_config()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    remote_out = f"{cfg.remote_project_dir.rstrip('/')}/runs/mechferret-cluster" if cfg.remote_project_dir else "runs/mechferret-cluster"
    remote_command = build_remote_command(cfg, skill, question, task, model, remote_out)
    invocation = build_srun_invocation(cfg, remote_command)
    printable = " ".join(shlex.quote(token) for token in invocation)

    if dry_run:
        return {"backend": "cluster", "dry_run": True, "command": printable, "out_dir": str(out_path)}
    if not cfg.configured:
        result = _local(question, skill, task, model, out_path)
        result["note"] = "Cluster not configured (set REMOTE_HOST + REMOTE_PROJECT_DIR); ran locally."
        return result
    try:
        proc = subprocess.run(invocation, capture_output=True, text=True, timeout=cfg_seconds(cfg.time))
        if proc.returncode != 0:
            result = _local(question, skill, task, model, out_path)
            result["note"] = f"Cluster srun failed (rc={proc.returncode}); ran locally. stderr: {proc.stderr[-400:]}"
            return result
        # Copy the dossier back.
        scp = subprocess.run(
            ["scp", "-q", f"{cfg.host}:{remote_out}/run.json", str(out_path / "run.json")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = {}
        run_json = out_path / "run.json"
        if scp.returncode == 0 and run_json.exists():
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        return {
            "backend": "cluster",
            "command": printable,
            "run": payload,
            "out_dir": str(out_path),
            "stdout_tail": proc.stdout[-400:],
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        result = _local(question, skill, task, model, out_path)
        result["note"] = f"Cluster dispatch error ({exc}); ran locally."
        return result


def cfg_seconds(hhmmss: str) -> int:
    try:
        h, m, s = (int(part) for part in hhmmss.split(":"))
        return h * 3600 + m * 60 + s
    except ValueError:
        return 7200


def _local(question, skill, task, model, out_path) -> dict[str, Any]:
    from .discovery import DiscoveryController

    run = DiscoveryController().run(
        question=question, skill=skill, task=task, model=model, backend="auto", out_dir=out_path
    )
    return {"backend": "local", "run": run.to_dict(), "out_dir": str(out_path)}
