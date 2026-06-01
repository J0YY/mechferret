"""Modal compute integration.

MechFerret can run locally through its fallback backend. When real measurements
are wanted, Modal supplies the GPU: the *entire* discovery loop runs remotely on
a GPU container with ``torch`` + ``transformer_lens`` installed, and the
resulting dossier comes back as JSON. This keeps a laptop in the loop only for
orchestration, never for heavy compute.

Three layers live here:

- ``modal_status`` / ``modal_available`` -- pure-Python detection used by the
  ``/modal`` CLI command; importable with or without Modal installed.
- ``run_discovery_remote`` / ``run_interp_remote`` -- the GPU functions.
- ``dispatch_discovery`` -- run the loop on Modal when configured. It fails
  closed by default so requested GPU work is not silently replaced by a local
  synthetic run; callers can explicitly opt into local fallback for demos.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:  # pragma: no cover - Modal is an optional dependency
    modal = None

GPU_TYPE = os.getenv("MECHFERRET_MODAL_GPU", "A10G")
APP_NAME = "mechferret-interp"


def modal_available() -> bool:
    return modal is not None


def modal_authenticated() -> bool:
    """True if a Modal token is configured (env or token file)."""

    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return True
    token_file = Path.home() / ".modal.toml"
    return token_file.exists()


def modal_status() -> dict[str, Any]:
    return {
        "installed": modal_available(),
        "authenticated": modal_authenticated(),
        "gpu": GPU_TYPE,
        "app": APP_NAME,
        "torch_local": importlib.util.find_spec("torch") is not None,
        "transformer_lens_local": importlib.util.find_spec("transformer_lens") is not None,
    }


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformer_lens>=2.0.0", "openai", "anthropic")
    )
    app = modal.App(APP_NAME, image=image)

    @app.function(gpu=GPU_TYPE, timeout=1800)
    def run_interp_remote(spec_dicts: list[dict], model: str | None = None) -> list[dict]:
        """Run a batch of experiment specs on a real model on a GPU."""

        from dataclasses import asdict

        from mechferret.interp.engine import InterpEngine
        from mechferret.models import ExperimentSpec

        engine = InterpEngine(model=model, backend="transformer_lens")
        specs = [ExperimentSpec(**spec) for spec in spec_dicts]
        return [asdict(engine.run_spec(spec)) for spec in specs]

    @app.function(gpu=GPU_TYPE, timeout=1800)
    def run_discovery_remote(
        question: str = "",
        skill: str | None = None,
        task: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Run the full discovery loop on a real model on a GPU."""

        import time

        from mechferret.discovery import DiscoveryController

        start = time.perf_counter()
        run = DiscoveryController("/tmp/mechferret/memory.sqlite").run(
            question=question,
            skill=skill,
            task=task,
            model=model,
            backend="transformer_lens",
            out_dir="/tmp/mechferret/run",
            include_memory=False,
        )
        payload = run.to_dict()
        payload.setdefault("metrics", {})["modal_gpu_seconds"] = round(time.perf_counter() - start, 2)
        return payload

    @app.local_entrypoint()
    def main(question: str = "", skill: str = "", model: str = "") -> None:
        result = run_discovery_remote.remote(question=question, skill=skill or None, model=model or None)
        print("Discoveries:", len(result.get("discoveries", [])))
        print("Readiness:", result.get("metrics", {}).get("readiness_score"))


def dispatch_discovery(
    question: str = "",
    skill: str | None = None,
    task: str | None = None,
    model: str | None = None,
    out_dir: str | Path = "runs/modal",
    allow_local_fallback: bool = False,
) -> dict[str, Any]:
    """Run the discovery loop on Modal if available.

    Returns ``{"ok": true, "backend": "modal", "run": <run dict>}`` when remote
    execution succeeds. Missing Modal setup or dispatch errors return a
    structured failure unless ``allow_local_fallback`` is true.
    """

    if not modal_available():
        return _modal_failure(
            "Modal is not installed.",
            failed_check="modal_installed",
            next_actions=[
                "Install Modal with `pip install -e '.[modal,interp]'`.",
                "Run `mechferret modal setup --json` for setup steps.",
            ],
            question=question,
            skill=skill,
            task=task,
            model=model,
            out_dir=out_dir,
            allow_local_fallback=allow_local_fallback,
        )
    if not modal_authenticated():
        return _modal_failure(
            "Modal is not authenticated.",
            failed_check="modal_authenticated",
            next_actions=[
                "Run `modal token new` to authenticate Modal.",
                "Run `mechferret modal status --json` to verify setup.",
            ],
            question=question,
            skill=skill,
            task=task,
            model=model,
            out_dir=out_dir,
            allow_local_fallback=allow_local_fallback,
        )
    if modal_available() and modal_authenticated():
        try:
            with app.run():  # ephemeral app run -- no separate `modal deploy` needed
                payload = run_discovery_remote.remote(
                    question=question, skill=skill, task=task, model=model
                )
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            import json

            (out_path / "run.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return {"ok": True, "backend": "modal", "run": payload, "out_dir": str(out_path)}
        except Exception as exc:  # pragma: no cover - network/credential failures
            return _modal_failure(
                f"Modal dispatch failed: {exc}",
                failed_check="modal_dispatch",
                next_actions=[
                    "Run `mechferret modal status --json` to inspect Modal setup.",
                    "Retry with `--local-fallback` only if a synthetic local run is intentional.",
                ],
                question=question,
                skill=skill,
                task=task,
                model=model,
                out_dir=out_dir,
                allow_local_fallback=allow_local_fallback,
            )
    return _modal_failure(
        "Modal is unavailable.",
        failed_check="modal_available",
        next_actions=["Run `mechferret modal setup --json` for setup steps."],
        question=question,
        skill=skill,
        task=task,
        model=model,
        out_dir=out_dir,
        allow_local_fallback=allow_local_fallback,
    )


def _modal_failure(
    error: str,
    *,
    failed_check: str,
    next_actions: list[str],
    question: str,
    skill: str | None,
    task: str | None,
    model: str | None,
    out_dir: str | Path,
    allow_local_fallback: bool,
) -> dict[str, Any]:
    if allow_local_fallback:
        fallback = _local_discovery(question, skill, task, model, out_dir)
        fallback["note"] = f"{error} Ran locally because local fallback was explicitly requested."
        fallback["remote_error"] = error
        fallback["failed_checks"] = [failed_check]
        return fallback
    return {
        "ok": False,
        "backend": "modal",
        "error": error,
        "failed_checks": [failed_check],
        "next_actions": next_actions,
        "out_dir": str(out_dir),
    }


def _local_discovery(question, skill, task, model, out_dir) -> dict[str, Any]:
    from .discovery import DiscoveryController

    run = DiscoveryController().run(
        question=question, skill=skill, task=task, model=model, backend="synthetic", out_dir=out_dir
    )
    return {"ok": True, "backend": "local", "run": run.to_dict(), "out_dir": str(out_dir)}
