from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from .config import PROVIDERS, default_config_path, load_config
from .memory import ResearchMemory
from .registry import all_items
from .sources import example_corpus_path

QUICKSTART_DEMO_QUESTION = (
    "What should a team build to win an autoresearch systems hackathon, "
    "and what reliability risks must the implementation address?"
)

_REDACTED = "[redacted]"
_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_SENSITIVE_FIELD_MARKERS = ("api_key", "token", "secret", "password")
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{8,}\b"),
)
_DOSSIER_ARTIFACTS = ("run", "report", "markdown", "graph", "evals")
_DISCOVERY_ARTIFACTS = ("experiments", "discoveries")
_SHARING_ARTIFACTS = ("paper", "manifest", "bundle", "pdf", "review")
_SETUP_ARTIFACTS = ("quickstart", "ci", "openvla")
_RUN_ARTIFACTS = _DOSSIER_ARTIFACTS + ("trace",)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _line_preview(value: Any, *, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", _text(value)).strip()
    if limit <= 3 or len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _positive_int(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _flag(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _path(value: Any, default: str | Path = "") -> Path:
    if isinstance(value, (str, Path)):
        return Path(value)
    return Path(default)


def _policy(value: Any, default: str = "latest") -> str:
    text = _text(value).strip().lower()
    return text if text in {"latest", "best", "ready"} else default


def _json_object_from_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def doctor() -> dict[str, Any]:
    config = load_config()
    from .skills import list_skills

    interp_real = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformer_lens") is not None
    )
    checks = [
        check("python_version", sys.version_info >= (3, 11), ".".join(map(str, sys.version_info[:3])), "Install Python 3.11 or newer."),
        check("example_corpus", example_corpus_path().exists(), str(example_corpus_path()), "Restore mechferret/seed_corpus."),
        check("registry_items", len(all_items()) >= 10, str(len(all_items())), "Restore the tool/task registry."),
        check("skills_available", len(list_skills()) >= 1, f"{len(list_skills())} skills", "Restore mechferret/skills."),
        check("interp_backend", True, "transformer_lens" if interp_real else "local fallback"),
        check("config_path", True, str(default_config_path())),
        check("paper_generator", _paper_generator_ok(), "evidence-bound local paper scaffold", "Fix mechferret.paper local scaffold generation."),
        _openvla_project_check(),
        check("openai_package", importlib.util.find_spec("openai") is not None, "optional", "Install `mechferret[openai]` for SDK-dependent extensions.", optional=True),
        check("anthropic_package", importlib.util.find_spec("anthropic") is not None, "optional", "Install `mechferret[anthropic]` for SDK-dependent extensions.", optional=True),
        check("modal_package", importlib.util.find_spec("modal") is not None, "optional", "Install modal for remote GPU execution.", optional=True),
        check("transformer_lens_package", interp_real, "optional", "Install torch and transformer-lens for real local model experiments.", optional=True),
        _openvla_manifest_check(),
        _latest_run_audit_check(),
    ]
    from .cluster import load_cluster_config

    cluster_cfg = load_cluster_config()
    checks.append(
        check(
            "cluster_configured",
            cluster_cfg.configured,
            cluster_cfg.host if cluster_cfg.configured else "unset (set REMOTE_HOST + REMOTE_PROJECT_DIR)",
            "Set REMOTE_HOST and REMOTE_PROJECT_DIR for cluster execution.",
            optional=True,
        )
    )
    for provider in sorted(PROVIDERS):
        env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        configured = bool(os.getenv(env_name) or config.providers.get(provider, None) and config.providers[provider].api_key)
        checks.append(check(f"{provider}_key", configured, "configured" if configured else "missing", f"Run `mechferret login {provider}`.", optional=True))
    required_actions = [
        item["action"]
        for item in checks
        if not item["passed"] and not item["optional"] and item.get("action")
    ]
    optional_actions = [
        item["action"]
        for item in checks
        if not item["passed"] and item["optional"] and item.get("action")
    ]
    return {
        "ok": all(item["passed"] or item["optional"] for item in checks),
        "passed": all(item["passed"] or item["optional"] for item in checks),
        "strict_passed": all(item["passed"] or item["optional"] for item in checks),
        "all_integrations_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "next_actions": required_actions,
        "optional_next_actions": optional_actions,
        "strict_next_actions": required_actions,
        "all_integrations_next_actions": required_actions + optional_actions,
    }


def check(name: str, passed: bool, detail: str, action: str = "", optional: bool = False) -> dict[str, Any]:
    payload = {"name": name, "passed": bool(passed), "detail": detail, "optional": optional}
    if action:
        payload["action"] = action
    return payload


def init_project_notes(project_root: str | Path = ".", *, force: bool = False) -> dict[str, Any]:
    """Create the project notes file read by the interactive agent."""

    root = Path(project_root)
    path = root / "MECHFERRET.md"
    if path.exists() and not force:
        return {
            "ok": True,
            "created": False,
            "path": str(path),
            "detected_stack": _detected_interp_stack(),
            "next_actions": ["Edit MECHFERRET.md as the project changes, or pass --force to rewrite the starter file."],
        }
    root.mkdir(parents=True, exist_ok=True)
    stack = _detected_interp_stack()
    path.write_text(_project_notes_template(stack), encoding="utf-8")
    return {
        "ok": True,
        "created": True,
        "path": str(path),
        "detected_stack": stack,
        "next_actions": [
            "Edit MECHFERRET.md with the model, task, artifact locations, and paper acceptance bar.",
            "Run `mechferret quickstart --run` to create a local dossier.",
            "Run `mechferret doctor --strict` before sharing the project.",
        ],
    }


def print_project_init(result: dict[str, Any]) -> None:
    if result.get("created"):
        stack = ", ".join(result.get("detected_stack") or ["no interp stack"])
        print(f"Project notes: {result['path']}")
        print(f"Detected stack: {stack}")
    else:
        print(f"Project notes already exist: {result['path']}")
    for action in result.get("next_actions", []):
        print(f"- {action}")


def print_doctor(*, strict: bool = False, all_integrations: bool = False) -> None:
    result = doctor()
    if all_integrations:
        passed = result["all_integrations_passed"]
    elif strict:
        passed = result["strict_passed"]
    else:
        passed = result["passed"]
    print(f"Doctor: {'PASS' if passed else 'WARN'}")
    for item in result["checks"]:
        marker = "ok" if item["passed"] else ("optional" if item["optional"] else "warn")
        print(f"{marker:8} {item['name']}: {item['detail']}")
    if all_integrations:
        actions = result["all_integrations_next_actions"]
    elif strict:
        actions = result["strict_next_actions"]
    else:
        actions = result["next_actions"]
    if actions:
        print("Next actions:")
        for action in actions:
            print(f"  - {action}")
    elif not strict and result["optional_next_actions"]:
        print("Optional next actions:")
        for action in result["optional_next_actions"][:4]:
            print(f"  - {action}")
        remaining = len(result["optional_next_actions"]) - 4
        if remaining > 0:
            print(f"  - ...and {remaining} more with `mechferret doctor --all-integrations`")


def selftest(
    *,
    run: bool = False,
    out_dir: str | Path = "runs/selftest",
    db_path: str | Path = ".mechferret/selftest.sqlite",
    runs_root: str | Path = "runs",
    notes_root: str | Path = ".",
    project_root: str | Path = "projects/openvla_sae",
    selection: str = "best",
    report_path: str | Path | None = None,
    command: str = "selftest",
) -> dict[str, Any]:
    command_name = _text(command).strip().lower()
    if command_name not in {"selftest", "support"}:
        command_name = "selftest"
    doc = doctor()
    guide = quickstart("all")
    status = project_status(
        runs_root=runs_root,
        db_path=db_path,
        notes_root=notes_root,
        project_root=project_root,
        selection=selection,
    )
    steps = [
        _ci_step("doctor_strict", doc["strict_passed"], "release-critical checks"),
        _ci_step("quickstart_guidance", bool(guide.get("sections")), f"{len(guide.get('sections', []))} workflow sections"),
        _ci_step("project_status", bool(status.get("state")), f"state={status.get('state', 'unknown')}", optional=True),
    ]
    artifacts: dict[str, Any] = {}
    demo_result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    if run:
        demo_result = run_quickstart("demo", out_dir=out_dir, db_path=db_path)
        artifacts.update(demo_result.get("artifacts", {}))
        run_json = _path(artifacts.get("run_json") or (_path(out_dir) / "run.json"))
        steps.append(_ci_step("demo_quickstart", bool(demo_result.get("ok")), artifacts.get("run_json", str(run_json))))
        verification = verify_run_artifacts(run_json)
        steps.append(_ci_step("verify_manifest", bool(verification.get("passed")), ", ".join(verification.get("failed_checks", [])) or "passed"))
    required_actions = [
        step["detail"]
        for step in steps
        if not step["ok"] and not step.get("optional") and step.get("detail")
    ]
    suggested_actions = []
    if not run:
        suggested_actions.append(f"Run `mechferret {command_name} --run` to execute the local demo and verify its artifacts.")
    result: dict[str, Any] = {
        "schema_version": 1,
        "ok": all(step["ok"] or step.get("optional", False) for step in steps),
        "command": command_name,
        "mode": "demo" if run else "core",
        "report": {
            "kind": command_name,
            "shareable": command_name == "support",
            "privacy": {
                "credential_values": "omitted",
                "environment_values": "omitted",
                "provider_credentials": "reported only as configured or missing",
            },
        },
        "steps": steps,
        "doctor": {
            "passed": doc["passed"],
            "strict_passed": doc["strict_passed"],
            "all_integrations_passed": doc["all_integrations_passed"],
        },
        "quickstart_sections": [section.get("name", "") for section in guide.get("sections", [])],
        "project_status": {
            "state": status.get("state", "unknown"),
            "run_selection": status.get("run_selection", selection),
            "readiness": status.get("readiness", {}),
            "readiness_summary": status.get("readiness_summary", []),
            "run_ready": status.get("run_ready", False),
            "share_ready": status.get("share_ready", False),
            "selected_run": status.get("selected_run", {}),
            "artifact_summary": status.get("artifact_summary", {}),
            "artifact_readiness": status.get("artifact_readiness", {}),
            "available_artifacts": status.get("available_artifacts", []),
            "missing_artifacts": status.get("missing_artifacts", []),
            "next_actions": status.get("next_actions", []),
            "suggested_next_actions": status.get("suggested_next_actions", []),
        },
        "artifacts": artifacts,
        "next_actions": _dedupe_actions(required_actions + doc.get("strict_next_actions", [])),
        "suggested_next_actions": _dedupe_actions(suggested_actions),
        "optional_next_actions": doc.get("optional_next_actions", []),
    }
    if demo_result is not None:
        result["demo"] = demo_result
    if verification is not None:
        result["verification"] = verification
    if report_path is not None:
        path = _path(report_path)
        artifacts["selftest_report"] = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    if command_name == "support":
        result = _redact_support_report(result)
    if report_path is not None:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def print_selftest(result: dict[str, Any]) -> None:
    label = "Support report" if result.get("command") == "support" else "Self-test"
    print(f"{label}: {'PASS' if result.get('ok') else 'WARN'}")
    print(f"Mode: {result.get('mode', 'core')}")
    redaction = _mapping(_mapping(result.get("report")).get("redaction"))
    if redaction.get("applied"):
        print(f"Redaction: applied ({_positive_int(redaction.get('total'), 0)} replacements)")
    for step in result.get("steps", []):
        marker = "ok" if step.get("ok") else ("info" if step.get("optional") else "fix")
        print(f"{marker:4} {step.get('name', '')}: {step.get('detail', '')}")
    status = _mapping(result.get("project_status"))
    if status:
        selection = _text(status.get("run_selection")) or "latest"
        summary = _mapping(status.get("artifact_summary"))
        if summary:
            print(f"Status selection: {selection} ({summary.get('found', 0)}/{summary.get('total', 0)} artifacts found)")
        else:
            print(f"Status selection: {selection}")
        readiness = _mapping(status.get("readiness"))
        if readiness:
            selected = _mapping(readiness.get("selected_run"))
            sharing = _mapping(readiness.get("sharing"))
            setup = _mapping(readiness.get("setup"))
            print(
                "Readiness: "
                f"run={'READY' if selected.get('ok') else 'BLOCKED'}; "
                f"share={'READY' if sharing.get('ok') else 'BLOCKED'}; "
                f"setup={'READY' if setup.get('ok') else 'BLOCKED'}"
            )
        readiness_summary = [item for item in status.get("readiness_summary", []) if isinstance(item, dict)]
        if readiness_summary:
            print("Readiness lanes:")
            for item in readiness_summary:
                name = _text(item.get("name")) or "lane"
                state = "READY" if item.get("ready") else "BLOCKED"
                reason = _text(item.get("reason"))
                print(f"  - {name}: {state}" + (f" ({reason})" if reason else ""))
    artifacts = result.get("artifacts", {})
    if artifacts:
        print("Artifacts:")
        for name, path in artifacts.items():
            if path:
                print(f"  - {name}: {path}")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"]:
            print(f"  - {action}")
    if status.get("next_actions"):
        print("Project status next actions:")
        for action in status["next_actions"][:6]:
            print(f"  - {action}")
    suggested_actions = _actions_not_repeated(
        result.get("suggested_next_actions", []) + status.get("suggested_next_actions", []),
        result.get("next_actions", []) + status.get("next_actions", []),
    )
    if suggested_actions:
        print("Suggested next actions:")
        for action in suggested_actions[:6]:
            print(f"  - {action}")


def _redact_support_report(value: Any) -> Any:
    counts = {"field": 0, "value": 0, "pattern": 0}
    redacted = _redact_value(value, replacements=_support_redaction_replacements(), counts=counts)
    if isinstance(redacted, dict):
        report = redacted.setdefault("report", {})
        if isinstance(report, dict):
            report["redaction"] = {
                "applied": True,
                "field_values": counts["field"],
                "configured_values": counts["value"],
                "credential_patterns": counts["pattern"],
                "total": sum(counts.values()),
            }
    return redacted


def _support_redaction_replacements() -> tuple[tuple[str, str], ...]:
    replacements: dict[str, str] = {}
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS):
            if isinstance(value, str) and len(value) >= 8:
                replacements[value] = _REDACTED
    config = load_config()
    for settings in config.providers.values():
        key = getattr(settings, "api_key", "")
        if isinstance(key, str) and len(key) >= 8:
            replacements[key] = _REDACTED
    try:
        home = str(Path.home())
    except RuntimeError:
        home = ""
    if len(home) > 1:
        replacements.setdefault(home, "~")
    return tuple(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _redact_value(
    value: Any,
    *,
    replacements: tuple[tuple[str, str], ...],
    counts: dict[str, int],
    key_name: str = "",
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_value(
                child,
                replacements=replacements,
                counts=counts,
                key_name=str(key),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, replacements=replacements, counts=counts, key_name=key_name) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, replacements=replacements, counts=counts, key_name=key_name) for item in value]
    if isinstance(value, str):
        lowered = key_name.lower()
        if any(marker in lowered for marker in _SENSITIVE_FIELD_MARKERS) and value:
            counts["field"] += 1
            return _REDACTED
        text = value
        for needle, replacement in replacements:
            if needle:
                matches = text.count(needle)
                counts["value"] += matches
                text = text.replace(needle, replacement)
        for pattern in _CREDENTIAL_PATTERNS:
            text, matches = pattern.subn(_REDACTED, text)
            counts["pattern"] += matches
        return text
    return value


def _detected_interp_stack() -> list[str]:
    return [
        module
        for module in ("torch", "transformer_lens", "sae_lens", "nnsight")
        if importlib.util.find_spec(module) is not None
    ]


def _project_notes_template(stack: list[str]) -> str:
    installed = ", ".join(stack) if stack else "none detected (install torch + transformer_lens for real experiments)"
    return f"""# MechFerret Project Notes

This file is read into the agent's system prompt each turn. Keep it short,
current, and specific to this research project.

## Stack
Installed: {installed}

## Research Target
- Model under study: gpt2
- Task or behavior:
- Dataset or prompt set:
- Main hypothesis:

## Evidence Rules
- Put run outputs under `runs/`.
- Log seeds, controls, model revision, and code revision for every causal claim.
- Do not promote a claim to the paper until `mechferret audit --strict` passes
  for the run artifact that supports it.

## Paper Acceptance Bar
- What result would be publishable?
- What controls or baselines are mandatory?
- What failure modes would invalidate the claim?
"""


def quickstart(mode: str = "all") -> dict[str, Any]:
    mode = _text(mode).strip().lower() or "all"
    if mode not in {"all", "demo", "openvla", "ci"}:
        raise ValueError(f"unknown quickstart mode: {mode}")
    sections = []
    if mode in {"all", "demo"}:
        sections.append(
            {
                "name": "local_demo",
                "goal": "Create a complete local dossier without API keys or GPUs.",
                "commands": [
                    "mechferret init",
                    "mechferret quickstart --run",
                    "mechferret status",
                    "mechferret support",
                    "mechferret open quickstart",
                    "mechferret open report --select best --browser",
                    "mechferret audit runs/demo/run.json --strict",
                    "mechferret verify runs/demo/run.json --strict",
                ],
            }
        )
    if mode in {"all", "openvla"}:
        sections.append(
            {
                "name": "openvla_sae",
                "goal": "Scaffold the OpenVLA SAE workflow and prepare a real manifest/cache/train run.",
                "commands": [
                    "mechferret sae openvla init",
                    "mechferret sae openvla status",
                    "mechferret sae openvla create-manifest --image-dir data/openvla_images --manifest data/openvla_sae_phase1.jsonl",
                    "mechferret sae openvla validate-manifest --manifest data/openvla_sae_phase1.jsonl",
                    "mechferret sae openvla commands",
                ],
            }
        )
    if mode in {"all", "ci"}:
        sections.append(
            {
                "name": "release_gates",
                "goal": "Run the checks that should stay green before publishing or demoing.",
                "commands": [
                    "mechferret quickstart --mode ci --run",
                    "mechferret doctor",
                    "mechferret quickstart --mode demo --run",
                    "python3 -m unittest discover -s tests -q",
                    "python3 -m compileall -q projects/openvla_sae/src mechferret tests",
                    "mechferret audit runs/demo/run.json --json",
                    "mechferret verify runs/demo/run.json --strict",
                    "mechferret bundle --select best",
                    "mechferret verify-bundle --select best --strict",
                    "mechferret doctor --strict",
                    "mechferret audit runs/demo/run.json --strict",
                    "mechferret doctor --all-integrations  # optional exhaustive integration audit",
                ],
            }
        )
    doc = doctor()
    return {
        "ok": doc["strict_passed"],
        "mode": mode,
        "doctor_passed": doc["passed"],
        "doctor_strict_passed": doc["strict_passed"],
        "doctor_all_integrations_passed": doc["all_integrations_passed"],
        "sections": sections,
        "next_actions": doc["next_actions"],
        "optional_next_actions": doc["optional_next_actions"],
    }


def print_quickstart(result: dict[str, Any]) -> None:
    print(f"Quickstart: {result['mode']}")
    print(f"Doctor: {'PASS' if result['doctor_passed'] else 'WARN'}")
    for section in result["sections"]:
        print()
        print(f"{section['name']}: {section['goal']}")
        for index, command in enumerate(section["commands"], 1):
            print(f"  {index}. {command}")
    actions = result["next_actions"] or result["optional_next_actions"][:4]
    if actions:
        print()
        print("Current environment notes:")
        for action in actions:
            print(f"  - {action}")


def run_quickstart(
    mode: str = "demo",
    *,
    out_dir: str | Path = "runs/demo",
    db_path: str | Path = ".mechferret/memory.sqlite",
    project_root: str | Path = "projects/openvla_sae",
    force: bool = False,
) -> dict[str, Any]:
    mode = _text(mode).strip().lower() or "demo"
    if mode == "all":
        mode = "demo"
    if mode == "demo":
        return _run_demo_quickstart(out_dir=out_dir, db_path=db_path)
    if mode == "openvla":
        return _run_openvla_quickstart(project_root=project_root, force=force)
    if mode == "ci":
        return _run_ci_quickstart(out_dir=out_dir, db_path=db_path)
    raise ValueError("--run supports --mode demo, --mode openvla, or --mode ci; omit --mode to run the local demo")


def print_quickstart_run(result: dict[str, Any]) -> None:
    print(f"Quickstart run: {'PASS' if result['ok'] else 'WARN'}")
    print(f"Mode: {result['mode']}")
    for step in result["steps"]:
        marker = "ok" if step["ok"] else "fix"
        print(f"{marker:4} {step['name']}: {step['detail']}")
    artifacts = result.get("artifacts", {})
    if artifacts:
        print("Artifacts:")
        for name, path in artifacts.items():
            if path:
                print(f"  - {name}: {path}")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"]:
            print(f"  - {action}")
    if result.get("suggested_next_actions"):
        print("Suggested next actions:")
        for action in result["suggested_next_actions"]:
            print(f"  - {action}")


def _artifact_summary(artifacts: dict[str, Any], *, include_setup: bool = True) -> dict[str, Any]:
    rows = {name: row for name, row in artifacts.items() if isinstance(row, dict)}
    found = [name for name, row in rows.items() if row.get("exists")]
    missing = [name for name, row in rows.items() if not row.get("exists")]

    def group(names: tuple[str, ...]) -> dict[str, Any]:
        present = [name for name in names if rows.get(name, {}).get("exists")]
        absent = [name for name in names if name in rows and not rows[name].get("exists")]
        return {
            "total": len([name for name in names if name in rows]),
            "found": len(present),
            "missing": len(absent),
            "found_artifacts": present,
            "missing_artifacts": absent,
        }

    groups = {
        "run": group(_RUN_ARTIFACTS),
        "dossier": group(_DOSSIER_ARTIFACTS),
        "discovery": group(_DISCOVERY_ARTIFACTS),
        "sharing": group(_SHARING_ARTIFACTS),
    }
    if include_setup:
        groups["setup"] = group(_SETUP_ARTIFACTS)

    return {
        "total": len(rows),
        "found": len(found),
        "missing": len(missing),
        "found_artifacts": found,
        "missing_artifacts": missing,
        "groups": groups,
    }


def _artifact_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    groups = _mapping(summary.get("groups"))

    def group_ready(name: str) -> dict[str, Any]:
        row = _mapping(groups.get(name))
        missing = _items(row.get("missing_artifacts"))
        return {
            "ok": not missing and _positive_int(row.get("total"), 0) > 0,
            "found": _positive_int(row.get("found"), 0),
            "total": _positive_int(row.get("total"), 0),
            "missing_artifacts": missing,
        }

    readiness = {
        "run": group_ready("run"),
        "sharing": group_ready("sharing"),
    }
    if "setup" in groups:
        readiness["setup"] = group_ready("setup")
    return readiness


def _compact_status_next_actions(actions: list[str], *, selection: str, allow_sharing: bool = False) -> list[str]:
    contextualized = [_contextualize_status_next_action(action, selection=selection) for action in actions]
    has_run_bound_paper_action = any("paper/main.tex" in action for action in contextualized)
    compacted: list[str] = []
    for action in contextualized:
        if not allow_sharing and _is_sharing_next_action(action):
            continue
        if (
            has_run_bound_paper_action
            and action.startswith("Run `mechferret paper --select ")
            and "generate a run-bound draft" in action
        ):
            continue
        compacted.append(action)
    return compacted


def _is_sharing_next_action(action: str) -> bool:
    return (
        action.startswith("Run `mechferret review-paper")
        or action.startswith("Run `mechferret bundle")
        or (action.startswith("Run `mechferret paper") and "--compile" in action)
    )


def _contextualize_status_next_action(action: str, *, selection: str) -> str:
    selection = _policy(selection)
    replacements = {
        "mechferret paper --provider local": f"mechferret paper --select {selection} --provider local",
        "mechferret verify`": f"mechferret verify --select {selection}`",
        "mechferret verify ": f"mechferret verify --select {selection} ",
        "mechferret bundle`": f"mechferret bundle --select {selection}`",
        "mechferret bundle.": f"mechferret bundle --select {selection}.",
        "mechferret bundle ": f"mechferret bundle --select {selection} ",
    }
    for command, replacement in replacements.items():
        if command in action:
            return action.replace(command, replacement)
    return action


def _artifact_index_next_actions(artifacts: dict[str, dict[str, Any]], *, selection: str, selected_run: str) -> list[str]:
    if not selected_run:
        actions: list[str] = []
        if not artifacts.get("run", {}).get("exists"):
            actions.append("Run `mechferret quickstart --run` to create a demo dossier.")
        if not artifacts.get("openvla", {}).get("exists"):
            actions.append("Run `mechferret quickstart --mode openvla --run` to scaffold OpenVLA artifacts.")
        return _dedupe_actions(actions)

    select_flag = f" --select {selection}" if selection else ""
    actions = []
    paper_exists = bool(artifacts.get("paper", {}).get("exists"))
    if paper_exists and not artifacts.get("review", {}).get("exists"):
        actions.append(f"Run `mechferret review-paper{select_flag}` with a configured provider to critique the run-bound paper.")
    if paper_exists and not artifacts.get("pdf", {}).get("exists"):
        actions.append(f"Run `mechferret paper{select_flag} --compile` to create a compiled PDF.")
    if paper_exists and not artifacts.get("bundle", {}).get("exists"):
        actions.append(f"Run `mechferret bundle{select_flag}` to package the selected dossier for sharing.")
    if not artifacts.get("quickstart", {}).get("exists"):
        actions.append("Run `mechferret quickstart --run` to create a fresh local quickstart dossier.")
    if not artifacts.get("ci", {}).get("exists"):
        actions.append("Run `mechferret quickstart --mode ci --run` to create a CI summary.")
    if not paper_exists:
        actions.append(f"Run `mechferret paper{select_flag}` to generate a run-bound draft from the selected dossier.")
    if not artifacts.get("manifest", {}).get("exists"):
        actions.append("Rerun the dossier with the current MechFerret version to create manifest.json.")
    if not artifacts.get("openvla", {}).get("exists"):
        actions.append("Run `mechferret quickstart --mode openvla --run` to scaffold OpenVLA artifacts.")
    return _dedupe_actions(actions)


def project_status(
    *,
    runs_root: str | Path = "runs",
    db_path: str | Path = ".mechferret/memory.sqlite",
    notes_root: str | Path = ".",
    project_root: str | Path = "projects/openvla_sae",
    selection: str = "latest",
) -> dict[str, Any]:
    selection = _policy(selection)
    runs_root = _path(runs_root, "runs")
    db_path = _path(db_path, ".mechferret/memory.sqlite")
    notes_root = _path(notes_root, ".")
    project_root = _path(project_root, "projects/openvla_sae")
    latest_run = _selected_run_json(runs_root, selection=selection)
    doc = doctor()
    artifacts = _artifact_index(runs_root=runs_root, project_root=project_root, selection=selection)
    notes_path = notes_root / "MECHFERRET.md"
    if latest_run is None:
        audit = None
        verification = None
        bundle_verification = None
        latest_summary = None
    else:
        from .audit import audit_run_artifact

        audit = audit_run_artifact(latest_run)
        verification = verify_run_artifacts(latest_run)
        bundle_artifact = _latest_run_artifact(latest_run, "bundle", "mechferret-bundle.zip")
        bundle_verification = verify_bundle_artifacts(bundle_artifact) if bundle_artifact and bundle_artifact.exists() else None
        latest_summary = summarize_run_artifact(latest_run)
    bundle_failed = bool(bundle_verification and not bundle_verification.get("passed"))
    if latest_run is not None and (not (audit and audit["passed"]) or not (verification and verification["passed"]) or bundle_failed):
        state = "needs_attention"
    elif not notes_path.exists():
        state = "needs_setup"
    elif latest_run is None:
        state = "needs_run"
    else:
        state = "ready" if audit and audit["passed"] and verification and verification["passed"] else "needs_attention"
    run_ready = bool(latest_run is not None and audit and audit["passed"] and verification and verification["passed"])
    memory_path = db_path
    memory = memory_summary(memory_path) if memory_path.exists() else {"runs": 0, "claims": 0, "sources": 0}
    available = [name for name, item in artifacts["artifacts"].items() if item["exists"]]
    missing = [name for name, item in artifacts["artifacts"].items() if not item["exists"]]
    artifact_summary = _artifact_summary(artifacts["artifacts"])
    artifact_readiness = _artifact_readiness(artifact_summary)
    artifact_groups = artifact_summary["groups"]
    run_group = artifact_groups["run"]
    setup_group = artifact_groups["setup"]
    sharing_group = artifact_groups["sharing"]
    setup_missing = [
        *(["project_notes"] if not notes_path.exists() else []),
        *setup_group["missing_artifacts"],
    ]
    sharing_ready = bool(run_ready and not sharing_group["missing_artifacts"] and bundle_verification and bundle_verification.get("passed"))
    readiness = {
        "project": {
            "ok": state == "ready" and doc["strict_passed"],
            "state": state,
            "doctor_strict_passed": doc["strict_passed"],
            "project_notes": notes_path.exists(),
        },
        "selected_run": {
            "ok": run_ready,
            "audit_passed": bool(audit and audit.get("passed")),
            "verify_passed": bool(verification and verification.get("passed")),
            "path": str(latest_run) if latest_run else "",
            "missing_artifacts": run_group["missing_artifacts"],
        },
        "sharing": {
            "ok": sharing_ready,
            "bundle_verified": bool(bundle_verification and bundle_verification.get("passed")),
            "missing_artifacts": sharing_group["missing_artifacts"],
        },
        "setup": {
            "ok": not setup_missing,
            "project_notes": notes_path.exists(),
            "missing": setup_missing,
            "missing_artifacts": setup_group["missing_artifacts"],
        },
    }
    readiness_summary = _status_readiness_summary(readiness)
    next_actions = _compact_status_next_actions(
        _dedupe_actions(
            [
                *(["Run `mechferret init` to create MECHFERRET.md project notes."] if not notes_path.exists() else []),
                *(["Run `mechferret quickstart --run` to create a local dossier."] if latest_run is None else []),
                *((audit or {}).get("next_actions", [])),
                *((verification or {}).get("next_actions", [])),
                *((bundle_verification or {}).get("next_actions", [])),
                *artifacts.get("next_actions", []),
                *doc.get("next_actions", []),
            ]
        ),
        selection=selection,
        allow_sharing=run_ready,
    )
    advisories = (audit or {}).get("advisories", [])
    advisory_actions = (audit or {}).get("advisory_actions", [])
    if not next_actions and "paper" not in available:
        next_actions.append("Run `mechferret paper` to generate a run-bound draft from the selected dossier.")
    if not next_actions and "review" not in available:
        next_actions.append("Run `mechferret review-paper --select best` with a configured provider to critique the selected draft.")
    selected_run = {
        "path": str(latest_run) if latest_run else "",
        "exists": latest_run is not None,
        "ok": latest_run is not None,
        "summary": latest_summary,
    }
    return {
        "state": state,
        "ok": state == "ready" and doc["strict_passed"],
        "project_notes": {"path": str(notes_path), "exists": notes_path.exists(), "ok": notes_path.exists()},
        "doctor": {
            "ok": doc["strict_passed"],
            "passed": doc["passed"],
            "strict_passed": doc["strict_passed"],
            "all_integrations_passed": doc["all_integrations_passed"],
        },
        "memory": memory,
        "run_selection": selection,
        "run_ready": run_ready,
        "share_ready": sharing_ready,
        "readiness": readiness,
        "readiness_summary": readiness_summary,
        "selected_run": selected_run,
        "latest_run": selected_run,
        "audit": audit,
        "verification": verification,
        "bundle_verification": bundle_verification,
        "advisories": advisories,
        "advisory_actions": advisory_actions,
        "artifacts": artifacts["artifacts"],
        "artifact_summary": artifact_summary,
        "artifact_readiness": artifact_readiness,
        "available_artifacts": available,
        "missing_artifacts": missing,
        "next_actions": next_actions,
        "suggested_next_actions": _status_suggested_next_actions(
            state=state,
            selection=selection,
            run_ready=run_ready,
            notes_exists=notes_path.exists(),
            latest_run_exists=latest_run is not None,
            available_artifacts=available,
        ),
    }


def print_project_status(result: dict[str, Any]) -> None:
    print(f"Project status: {result['state']}")
    notes = result["project_notes"]
    print(f"Project notes: {'found' if notes['exists'] else 'missing'} ({notes['path']})")
    doctor_state = "PASS" if result["doctor"]["strict_passed"] else "WARN"
    print(f"Doctor: {doctor_state}")
    memory = result["memory"]
    print(f"Memory: runs={memory['runs']} claims={memory['claims']} sources={memory['sources']}")
    latest = result["latest_run"]
    run_label = "Selected run" if result.get("run_selection", "latest") != "latest" else "Latest run"
    if latest["exists"]:
        summary = latest["summary"] or {}
        print(f"{run_label}: {latest['path']}")
        print(f"Question: {_line_preview(summary.get('question', ''))}")
        print(f"Readiness: {float(summary.get('readiness_score', 0)):.2f}")
        print(f"Run readiness: {'READY' if result.get('run_ready') else 'BLOCKED'}")
        print(f"Share readiness: {'READY' if result.get('share_ready') else 'BLOCKED'}")
        setup_ready = _mapping(result.get("readiness", {}).get("setup")).get("ok")
        print(f"Setup readiness: {'READY' if setup_ready else 'BLOCKED'}")
        audit = result.get("audit") or {}
        print(f"Audit: {'PASS' if audit.get('passed') else 'WARN'}")
        if audit.get("failed_checks"):
            print(f"Failed checks: {', '.join(audit['failed_checks'])}")
        verification = result.get("verification") or {}
        print(f"Verify: {'PASS' if verification.get('passed') else 'WARN'}")
        if verification.get("failed_checks"):
            print(f"Verification failures: {', '.join(verification['failed_checks'][:4])}")
        bundle_verification = result.get("bundle_verification")
        if bundle_verification:
            print(f"Bundle verify: {'PASS' if bundle_verification.get('passed') else 'WARN'}")
            if bundle_verification.get("failed_checks"):
                print(f"Bundle failures: {', '.join(bundle_verification['failed_checks'][:4])}")
        if result.get("advisories"):
            print("Advisories:")
            for item in result["advisories"][:4]:
                print(f"  - {item['name']}: {item.get('observed', '')}")
    else:
        print(f"{run_label}: missing")
        setup_ready = _mapping(result.get("readiness", {}).get("setup")).get("ok")
        print("Run readiness: BLOCKED")
        print("Share readiness: BLOCKED")
        print(f"Setup readiness: {'READY' if setup_ready else 'BLOCKED'}")
    available = ", ".join(result["available_artifacts"]) or "none"
    missing = ", ".join(result["missing_artifacts"]) or "none"
    artifact_summary = result.get("artifact_summary") or _artifact_summary(result.get("artifacts", {}))
    run_group = artifact_summary["groups"].get("run", {})
    setup_group = artifact_summary["groups"].get("setup", {})
    dossier = artifact_summary["groups"]["dossier"]
    discovery = artifact_summary["groups"]["discovery"]
    sharing = artifact_summary["groups"]["sharing"]
    print(
        "Artifact summary: "
        f"{artifact_summary['found']}/{artifact_summary['total']} found; "
        f"dossier {dossier['found']}/{dossier['total']}; "
        f"discovery {discovery['found']}/{discovery['total']}; "
        f"sharing {sharing['found']}/{sharing['total']}"
    )
    run_missing = ", ".join(run_group.get("missing_artifacts", [])) or "none"
    setup_missing = ", ".join(setup_group.get("missing_artifacts", [])) or "none"
    print(f"Run artifacts: {run_group.get('found', 0)}/{run_group.get('total', 0)} found; missing {run_missing}")
    print(f"Setup artifacts: {setup_group.get('found', 0)}/{setup_group.get('total', 0)} found; missing {setup_missing}")
    print(f"Artifacts found: {available}")
    print(f"Artifacts missing: {missing}")
    if result["next_actions"]:
        print("Next actions:")
        for action in result["next_actions"][:8]:
            print(f"  - {action}")
    suggested_actions = _actions_not_repeated(
        result.get("suggested_next_actions", []),
        result.get("next_actions", []),
    )
    if suggested_actions:
        print("Suggested next actions:")
        for action in suggested_actions[:6]:
            print(f"  - {action}")


def _status_readiness_summary(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    setup = _mapping(readiness.get("setup"))
    selected = _mapping(readiness.get("selected_run"))
    sharing = _mapping(readiness.get("sharing"))
    project = _mapping(readiness.get("project"))
    setup_missing = _string_list(setup.get("missing")) or _string_list(setup.get("missing_artifacts"))
    selected_missing = _string_list(selected.get("missing_artifacts"))
    sharing_missing = _string_list(sharing.get("missing_artifacts"))
    return [
        {
            "name": "setup",
            "ready": bool(setup.get("ok")),
            "status": "ready" if setup.get("ok") else "blocked",
            "reason": "Project notes and setup indexes are present." if setup.get("ok") else f"Missing setup items: {', '.join(setup_missing) or 'unknown'}.",
        },
        {
            "name": "run",
            "ready": bool(selected.get("ok")),
            "status": "ready" if selected.get("ok") else "blocked",
            "reason": "Selected run passed audit and artifact verification."
            if selected.get("ok")
            else f"Selected run is not ready; missing artifacts: {', '.join(selected_missing) or 'run artifact'}.",
        },
        {
            "name": "sharing",
            "ready": bool(sharing.get("ok")),
            "status": "ready" if sharing.get("ok") else "blocked",
            "reason": "Share bundle is present and verifies cleanly."
            if sharing.get("ok")
            else _sharing_readiness_reason(sharing, sharing_missing),
        },
        {
            "name": "project",
            "ready": bool(project.get("ok")),
            "status": str(project.get("state") or ("ready" if project.get("ok") else "blocked")),
            "reason": "Project is fully ready."
            if project.get("ok")
            else _project_readiness_reason(project, setup_missing),
        },
    ]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    return []


def _sharing_readiness_reason(sharing: dict[str, Any], missing: list[str]) -> str:
    if missing:
        return f"Sharing is missing artifacts: {', '.join(missing)}."
    if not sharing.get("bundle_verified"):
        return "Sharing bundle is missing or does not verify cleanly."
    return "Sharing requirements are incomplete."


def _project_readiness_reason(project: dict[str, Any], setup_missing: list[str]) -> str:
    if not project.get("doctor_strict_passed"):
        return "Doctor strict checks are not passing."
    if setup_missing:
        return f"Workspace setup is incomplete: {', '.join(setup_missing)}."
    state = str(project.get("state") or "blocked")
    return f"Project state is {state}."


def _status_suggested_next_actions(
    *,
    state: str,
    selection: str,
    run_ready: bool = False,
    notes_exists: bool,
    latest_run_exists: bool,
    available_artifacts: list[str],
) -> list[str]:
    select_flag = f" --select {selection}" if selection else ""
    suggestions: list[str] = []
    if not notes_exists:
        suggestions.append("Run `mechferret init` to create project notes.")
    if not latest_run_exists:
        suggestions.append("Run `mechferret quickstart --run` to create a local demo dossier.")
        return suggestions
    if "report" in available_artifacts:
        suggestions.append(f"Run `mechferret open report{select_flag} --browser` to inspect the report.")
    if "quickstart" in available_artifacts:
        suggestions.append("Run `mechferret open quickstart` to inspect the artifact index.")
    if (state == "ready" or run_ready) and "bundle" not in available_artifacts:
        suggestions.append(f"Run `mechferret bundle{select_flag}` to package the dossier for sharing.")
    if (state == "ready" or run_ready) and "review" not in available_artifacts:
        suggestions.append(f"Run `mechferret review-paper{select_flag}` with a configured provider to critique the draft.")
    return _dedupe_actions(suggestions)


def list_run_artifacts(
    *,
    runs_root: str | Path = "runs",
    limit: int = 10,
    include_audit: bool = True,
    selection: str = "best",
) -> dict[str, Any]:
    root = _path(runs_root, "runs")
    limit_value = _positive_int(limit, 10)
    selection = _policy(selection, "best")
    if not root.exists():
        return {
            "ok": True,
            "runs_root": str(root),
            "count": 0,
            "shown": 0,
            "runs": [],
            "selection": selection,
            "selected": None,
            "selected_path": "",
            "selected_rank": 0,
            "selected_visible": False,
            "next_actions": ["Run `mechferret quickstart --run` to create a dossier."],
        }
    candidates = _run_json_candidates(root)
    all_rows = [_run_list_entry(path, include_audit=include_audit) for path in candidates]
    rows = all_rows[:limit_value]
    if include_audit:
        selected_result = _select_run_from_rows(all_rows, policy=selection)
    elif selection == "ready":
        selected_result = select_run_artifact(runs_root=root, policy=selection)
    else:
        selected_result = _select_run_from_rows(all_rows, policy=selection)
    selected_path = _text(selected_result.get("path"))
    selected_rank = next(
        (index for index, row in enumerate(all_rows, 1) if isinstance(row, dict) and row.get("path") == selected_path),
        0,
    )
    rows = [
        {
            **row,
            "selected": bool(selected_path and row.get("path") == selected_path),
            "selection": selection if selected_path and row.get("path") == selected_path else "",
        }
        if isinstance(row, dict)
        else row
        for row in rows
    ]
    selected_visible = any(isinstance(row, dict) and row.get("selected") for row in rows)
    next_actions = [] if candidates else ["Run `mechferret quickstart --run` to create a dossier."]
    if candidates and not selected_result.get("run"):
        next_actions.extend(selected_result.get("next_actions", []))
    selection_failure = {
        "nearest_run": selected_result.get("nearest_run"),
        "nearest_path": selected_result.get("nearest_path", ""),
        "failed_checks": selected_result.get("failed_checks", []),
    } if selected_result.get("nearest_run") else None
    if selected_path and selected_rank and not selected_visible:
        next_actions.append(f"Increase `--limit` to at least {selected_rank} to show the selected {selection} run in the listing.")
    return {
        "ok": True,
        "runs_root": str(root),
        "count": len(candidates),
        "shown": len(rows),
        "runs": rows,
        "selection": selection,
        "selected": selected_result.get("run"),
        "selected_path": selected_path,
        "selected_rank": selected_rank,
        "selected_visible": selected_visible,
        "selection_failure": selection_failure,
        "next_actions": _dedupe_actions(next_actions),
    }


def print_run_list(result: dict[str, Any]) -> None:
    print(f"Runs: {result['count']} found (showing {result['shown']})")
    selected = result.get("selected")
    if selected:
        label = {"best": "Best run", "ready": "Ready run", "latest": "Latest run"}.get(result.get("selection"), "Selected run")
        print(f"{label}: {selected['path']}")
        if not result.get("selected_visible") and result.get("selected_rank"):
            print(f"Selected run is outside the current --limit; increase --limit to at least {result['selected_rank']} to show it.")
    elif result.get("selection_failure"):
        failure = _mapping(result.get("selection_failure"))
        nearest = _mapping(failure.get("nearest_run"))
        failed = ", ".join(_text(item) for item in _items(failure.get("failed_checks")) if _text(item)) or "unknown"
        print(f"Closest run: {failure.get('nearest_path', '')}")
        if nearest:
            print(f"Closest readiness: {float(nearest.get('readiness_score', 0)):.2f}")
        print(f"Blocking checks: {failed}")
    for index, row in enumerate(result["runs"], 1):
        if not row.get("ok", True):
            print(f"{index}. WARN {row['path']}")
            print(f"   error: {row.get('error', 'unreadable run artifact')}")
            continue
        audit = row.get("audit") or {}
        verdict = "PASS" if audit.get("passed") else ("WARN" if audit else "UNCK")
        selected_marker = f" [selected: {row.get('selection')}]" if row.get("selected") else ""
        artifacts = ", ".join(name for name, exists in row["artifacts"].items() if exists) or "none"
        print(f"{index}. {verdict}{selected_marker} readiness={float(row['readiness_score']):.2f} claims={row['claims']} artifacts={artifacts}")
        lanes = _mapping(row.get("artifact_readiness"))
        if lanes:
            print(
                "   lanes: "
                f"run={'READY' if _mapping(lanes.get('run')).get('ok') else 'BLOCKED'} "
                f"share={'READY' if _mapping(lanes.get('sharing')).get('ok') else 'BLOCKED'}"
            )
        print(f"   {row['run_id']}  {row['path']}")
        print(f"   {_line_preview(row.get('question'))}")
        if audit.get("failed_checks"):
            print(f"   failed: {', '.join(audit['failed_checks'])}")
        if audit.get("advisories"):
            print(f"   advisories: {', '.join(item.get('name', '') for item in audit['advisories'][:3])}")
    if result["next_actions"]:
        print("Next actions:")
        for action in result["next_actions"]:
            print(f"  - {action}")


def select_run_artifact(
    *,
    runs_root: str | Path = "runs",
    policy: str = "latest",
    require_artifact: str | None = None,
    _rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy = _policy(policy)
    artifact = _text(require_artifact).strip() or None
    row_list = _items(_rows) if _rows is not None else [_run_list_entry(path, include_audit=True) for path in _run_json_candidates(_path(runs_root, "runs"))]
    return _select_run_from_rows(row_list, policy=policy, require_artifact=artifact)


def _select_run_from_rows(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    require_artifact: str | None = None,
) -> dict[str, Any]:
    policy = _policy(policy)
    require_artifact = _text(require_artifact).strip() or None
    rows = [row for row in _items(rows) if isinstance(row, dict) and row.get("ok", True)]
    if require_artifact:
        rows = [row for row in rows if row.get("artifacts", {}).get(require_artifact)]
    if not rows:
        if require_artifact == "quickstart":
            action = (
                "No run with `quickstart` found. Run `mechferret quickstart --run` to create a fresh local "
                "quickstart dossier, or choose a different selection policy."
            )
        elif require_artifact:
            action = f"No run with `{require_artifact}` found. Generate that artifact or choose a different selection policy."
        else:
            action = "Run `mechferret quickstart --run` to create a dossier."
        return {
            "policy": policy,
            "run": None,
            "path": "",
            "next_actions": [action],
        }
    if policy == "latest":
        selected = max(rows, key=lambda row: row.get("mtime", 0))
    elif policy == "ready":
        ready = [row for row in rows if row.get("audit", {}).get("passed")]
        if not ready:
            nearest = max(rows, key=_run_selection_score)
            failed = [
                _text(item)
                for item in _items(_mapping(nearest.get("audit")).get("failed_checks"))
                if _text(item)
            ]
            failed_label = ", ".join(failed) if failed else "unknown"
            return {
                "policy": policy,
                "run": None,
                "path": "",
                "nearest_run": nearest,
                "nearest_path": nearest.get("path", ""),
                "failed_checks": failed,
                "next_actions": [
                    f"No audit-passing run found. Closest run is `{nearest.get('path', '')}`; fix: {failed_label}.",
                    f"Run `mechferret audit {nearest.get('path', '')} --strict` for the full gate report.",
                ],
            }
        selected = max(ready, key=_run_selection_score)
    else:
        selected = max(rows, key=_run_selection_score)
    return {"policy": policy, "run": selected, "path": selected["path"], "next_actions": []}


def bundle_run_artifacts(
    run_json: str | Path | None = None,
    *,
    runs_root: str | Path = "runs",
    selection: str = "latest",
    out: str | Path | None = None,
    notes_root: str | Path = ".",
    project_root: str | Path = "projects/openvla_sae",
) -> dict[str, Any]:
    from .audit import audit_run_artifact
    from .models import utc_now
    from .provenance import refresh_run_manifest, sha256_file

    selection = _policy(selection)
    target = _path(run_json) if run_json else _selected_run_json(runs_root, selection=selection)
    if target is None or not target.is_file():
        return {
            "ok": False,
            "path": "",
            "run_json": str(target) if target else "",
            "files": [],
            "missing": ["run.json"],
            "next_actions": ["Run `mechferret quickstart --run` to create a dossier before bundling."],
        }
    target = target.resolve()
    run_dir = target.parent
    bundle_path = _bundle_output_path(out, run_dir)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _json_object_from_file(target)
    artifacts = _mapping(payload.get("artifacts"))
    payload["artifacts"] = artifacts
    artifacts["bundle"] = str(bundle_path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    bundle_path.touch(exist_ok=True)
    refresh_run_manifest(target)

    payload = _json_object_from_file(target)
    candidates = _bundle_candidates(target, payload, notes_root=notes_root, project_root=project_root)
    audit = audit_run_artifact(target)
    status = project_status(runs_root=target.parent, notes_root=notes_root, project_root=project_root, selection=selection)
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    used_arcnames: set[str] = set()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, path, arcname in candidates:
            if not path.exists() or not path.is_file():
                missing.append(label)
                continue
            arc = _unique_arcname(arcname, used_arcnames)
            archive.write(path, arc)
            files.append(
                {
                    "label": label,
                    "path": str(path),
                    "arcname": arc,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "created_at": utc_now(),
            "run_id": payload.get("run_id", ""),
            "question": payload.get("question", ""),
            "run_json": str(target),
            "bundle": str(bundle_path),
            "selection": selection,
            "status_run_json": status.get("selected_run", {}).get("path", ""),
            "audit_passed": audit["passed"],
            "readiness_score": audit.get("readiness_score", 0),
            "advisories": audit.get("advisories", []),
            "files": files,
            "missing": missing,
        }
        audit_bytes = json.dumps(audit, indent=2, sort_keys=True, default=str).encode("utf-8")
        status_bytes = json.dumps(status, indent=2, sort_keys=True, default=str).encode("utf-8")
        readme_bytes = _bundle_readme(manifest, audit).encode("utf-8")
        manifest["metadata_files"] = [
            _bundle_embedded_file_row("audit", "audit.json", audit_bytes),
            _bundle_embedded_file_row("status", "status.json", status_bytes),
            _bundle_embedded_file_row("readme", "README.md", readme_bytes),
        ]
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
        archive.writestr("audit.json", audit_bytes)
        archive.writestr("status.json", status_bytes)
        archive.writestr("README.md", readme_bytes)
    refresh_run_manifest(target)
    bundle_verification = verify_bundle_artifacts(bundle_path)
    if bundle_verification.get("passed"):
        next_actions = [] if audit["passed"] else audit.get("next_actions", [])
    else:
        next_actions = bundle_verification.get("next_actions", [])
    result = {
        "ok": bool(bundle_verification.get("passed")),
        "created": True,
        "path": str(bundle_path),
        "run_json": str(target),
        "artifacts": {"bundle": str(bundle_path)},
        "files": files,
        "missing": missing,
        "missing_optional": _bundle_missing_optional(missing),
        "missing_optional_actions": _bundle_missing_optional_actions(missing),
        "manifest": manifest,
        "bundle_verification": bundle_verification,
        "next_actions": next_actions,
    }
    return result


def print_bundle_result(result: dict[str, Any]) -> None:
    print(f"Bundle: {'PASS' if result.get('ok') else 'WARN'}")
    if result.get("path"):
        print(f"Path: {result['path']}")
    if result.get("run_json"):
        print(f"Run: {result['run_json']}")
    if result.get("files") is not None:
        print(f"Files: {len(result.get('files', []))}")
    bundle_verification = result.get("bundle_verification")
    if bundle_verification:
        print(f"Bundle verify: {'PASS' if bundle_verification.get('passed') else 'WARN'}")
        if bundle_verification.get("failed_checks"):
            print(f"Bundle failures: {', '.join(bundle_verification['failed_checks'][:4])}")
    if result.get("missing_optional"):
        print("Missing optional context:")
        for item in result["missing_optional"][:8]:
            print(f"  - {item.get('name', '')}")
            if item.get("action"):
                print(f"    action: {item['action']}")
    elif result.get("missing"):
        print("Missing optional files:")
        for label in result["missing"][:8]:
            print(f"  - {label}")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"][:8]:
            print(f"  - {action}")


def _bundle_missing_optional(missing: list[str]) -> list[dict[str, Any]]:
    missing_set = set(missing)
    groups = [
        (
            {"quickstart_markdown", "quickstart_json"},
            "local quickstart index",
            "Run `mechferret quickstart --run` to include the local quickstart guide.",
        ),
        (
            {"ci_markdown", "ci_json"},
            "CI quickstart summary",
            "Run `mechferret quickstart --mode ci --run` to include release-gate notes.",
        ),
        (
            {"project_notes"},
            "project notes",
            "Run `mechferret init` to include MECHFERRET.md.",
        ),
        (
            {"openvla_quickstart", "openvla_quickstart_json"},
            "OpenVLA quickstart guide",
            "Run `mechferret quickstart --mode openvla --run` to include OpenVLA setup notes.",
        ),
    ]
    items: list[dict[str, Any]] = []
    grouped_labels: set[str] = set()
    for labels, name, action in groups:
        present = sorted(label for label in labels if label in missing_set)
        if not present:
            continue
        grouped_labels.update(present)
        items.append({"name": name, "labels": present, "action": action})
    for label in sorted(missing_set - grouped_labels):
        items.append({"name": label.replace("_", " "), "labels": [label], "action": ""})
    return items


def _bundle_missing_optional_actions(missing: list[str]) -> list[str]:
    return _dedupe_actions(
        [_text(item.get("action")) for item in _bundle_missing_optional(missing) if item.get("action")]
    )


def verify_bundle_artifacts(
    bundle_zip: str | Path | None = None,
    *,
    runs_root: str | Path = "runs",
    selection: str = "latest",
) -> dict[str, Any]:
    selection = _policy(selection)
    target: Path | None
    selected_lookup = not bool(bundle_zip)
    if bundle_zip:
        target = _path(bundle_zip)
    else:
        resolved = resolve_artifact("bundle", runs_root=runs_root, selection=selection)
        target = Path(resolved["path"]) if resolved.get("exists") and resolved.get("path") else None
    if target is None:
        return _bundle_verify_result(
            Path(""),
            [{"name": "bundle_exists", "passed": False, "observed": "missing", "threshold": "exists"}],
            [f"Run `mechferret bundle --select {selection}` to create a shareable archive before verifying it."],
        )
    recreate_command = _bundle_recreate_command(selection=selection, selected_lookup=selected_lookup)
    recreate_action = f"Recreate the bundle with `{recreate_command}`."
    checks: list[dict[str, Any]] = [
        {
            "name": "bundle_exists",
            "passed": target.is_file(),
            "observed": str(target) if target.is_file() else "missing",
            "threshold": "file exists",
        }
    ]
    if not target.is_file():
        return _bundle_verify_result(target, checks, [recreate_action])
    try:
        with zipfile.ZipFile(target) as archive:
            namelist = archive.namelist()
            names = set(namelist)
            counts: dict[str, int] = {}
            for name in namelist:
                counts[name] = counts.get(name, 0) + 1
            duplicates = sorted(name for name, count in counts.items() if count > 1)
            unsafe_names = sorted(name for name in names if not _bundle_arcname_safe(name))
            checks.append(
                {
                    "name": "bundle_entries_unique",
                    "passed": not duplicates,
                    "observed": ", ".join(duplicates[:8]) if duplicates else "unique",
                    "threshold": "no duplicate archive names",
                }
            )
            checks.append(
                {
                    "name": "bundle_entries_path_safe",
                    "passed": not unsafe_names,
                    "observed": ", ".join(unsafe_names[:8]) if unsafe_names else "safe",
                    "threshold": "relative paths without traversal",
                }
            )
            checks.append(
                {
                    "name": "bundle_manifest_exists",
                    "passed": "manifest.json" in names,
                    "observed": "present" if "manifest.json" in names else "missing",
                    "threshold": "manifest.json",
                }
            )
            if duplicates:
                return _bundle_verify_result(
                    target,
                    checks,
                    [f"Recreate the bundle so every archive entry name is unique: `{recreate_command}`."],
                )
            if "manifest.json" not in names:
                return _bundle_verify_result(target, checks, [f"Recreate the bundle so manifest.json is included: `{recreate_command}`."])
            try:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                checks.append(
                    {
                        "name": "bundle_manifest_parseable",
                        "passed": False,
                        "observed": str(exc),
                        "threshold": "valid JSON",
                    }
                )
                return _bundle_verify_result(target, checks, [f"Recreate the bundle so manifest.json is valid JSON: `{recreate_command}`."])
            checks.append(
                {
                    "name": "bundle_manifest_object",
                    "passed": isinstance(manifest, dict),
                    "observed": type(manifest).__name__,
                    "threshold": "object",
                }
            )
            if not isinstance(manifest, dict):
                return _bundle_verify_result(target, checks, [f"Recreate the bundle so manifest.json is a JSON object: `{recreate_command}`."])
            files = manifest.get("files", [])
            checks.append(
                {
                    "name": "bundle_manifest_files_parseable",
                    "passed": isinstance(files, list),
                    "observed": type(files).__name__,
                    "threshold": "list",
                }
            )
            if not isinstance(files, list):
                return _bundle_verify_result(target, checks, [f"Recreate the bundle so manifest.json files is a list: `{recreate_command}`."])
            declared_arcnames: set[str] = set()
            _verify_bundle_file_rows(archive, names, files, declared_arcnames, checks, entry_kind="file")
            _verify_bundle_required_labels(files, {"run_json", "run_manifest"}, checks, entry_kind="file")
            _verify_bundle_required_label_arcnames(
                files,
                {"run_json": "run/run.json", "run_manifest": "run/manifest.json"},
                checks,
                entry_kind="file",
            )
            metadata = manifest.get("metadata_files")
            checks.append(
                {
                    "name": "bundle_metadata_files_parseable",
                    "passed": isinstance(metadata, list),
                    "observed": type(metadata).__name__,
                    "threshold": "list",
                }
            )
            if isinstance(metadata, list):
                _verify_bundle_file_rows(archive, names, metadata, declared_arcnames, checks, entry_kind="metadata")
                _verify_bundle_required_labels(metadata, {"audit", "status", "readme"}, checks, entry_kind="metadata")
                _verify_bundle_required_label_arcnames(
                    metadata,
                    {"audit": "audit.json", "status": "status.json", "readme": "README.md"},
                    checks,
                    entry_kind="metadata",
                )
                expected_metadata = {"audit.json", "status.json", "README.md"}
                missing_metadata = sorted(expected_metadata - declared_arcnames)
                checks.append(
                    {
                        "name": "bundle_metadata_files_complete",
                        "passed": not missing_metadata,
                        "observed": ", ".join(missing_metadata) if missing_metadata else "complete",
                        "threshold": "audit.json, status.json, README.md",
                    }
                )
            _add_bundle_semantic_checks(archive, names, manifest, checks)
            metadata_entries = {"manifest.json"}
            undeclared = sorted(names - declared_arcnames - metadata_entries)
            checks.append(
                {
                    "name": "bundle_entries_declared",
                    "passed": not undeclared,
                    "observed": ", ".join(undeclared[:8]) if undeclared else "all declared",
                    "threshold": "manifest files plus metadata entries",
                }
            )
    except zipfile.BadZipFile as exc:
        checks.append(
            {
                "name": "bundle_zip_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "valid zip",
            }
        )
    return _bundle_verify_result(target, checks, [recreate_action])


def _add_bundle_semantic_checks(
    archive: zipfile.ZipFile,
    names: set[str],
    manifest: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    run_arcname = _bundle_manifest_arcname(manifest.get("files"), "run_json", "run/run.json")
    run_manifest_arcname = _bundle_manifest_arcname(manifest.get("files"), "run_manifest", "run/manifest.json")
    audit_arcname = _bundle_manifest_arcname(manifest.get("metadata_files"), "audit", "audit.json")
    readme_arcname = _bundle_manifest_arcname(manifest.get("metadata_files"), "readme", "README.md")
    status_arcname = _bundle_manifest_arcname(manifest.get("metadata_files"), "status", "status.json")
    run_payload = _read_bundle_json_entry(archive, names, run_arcname)
    run_manifest_payload = _read_bundle_json_entry(archive, names, run_manifest_arcname)
    audit_payload = _read_bundle_json_entry(archive, names, audit_arcname)
    readme_payload = _read_bundle_text_entry(archive, names, readme_arcname)
    status_payload = _read_bundle_json_entry(archive, names, status_arcname)

    checks.append(
        {
            "name": "bundle_run_json_parseable",
            "passed": isinstance(run_payload, dict),
            "observed": "object" if isinstance(run_payload, dict) else "missing or invalid",
            "threshold": run_arcname,
        }
    )
    checks.append(
        {
            "name": "bundle_run_manifest_parseable",
            "passed": isinstance(run_manifest_payload, dict),
            "observed": "object" if isinstance(run_manifest_payload, dict) else "missing or invalid",
            "threshold": run_manifest_arcname,
        }
    )
    checks.append(
        {
            "name": "bundle_audit_parseable",
            "passed": isinstance(audit_payload, dict),
            "observed": "object" if isinstance(audit_payload, dict) else "missing or invalid",
            "threshold": audit_arcname,
        }
    )
    checks.append(
        {
            "name": "bundle_readme_parseable",
            "passed": isinstance(readme_payload, str),
            "observed": "text" if isinstance(readme_payload, str) else "missing or invalid",
            "threshold": readme_arcname,
        }
    )
    checks.append(
        {
            "name": "bundle_status_parseable",
            "passed": isinstance(status_payload, dict),
            "observed": "object" if isinstance(status_payload, dict) else "missing or invalid",
            "threshold": status_arcname,
        }
    )
    manifest_run_id = str(manifest.get("run_id", ""))
    archived_run_id = str(run_payload.get("run_id", "")) if isinstance(run_payload, dict) else ""
    checks.append(
        {
            "name": "bundle_manifest_run_id_matches_run_json",
            "passed": bool(manifest_run_id) and manifest_run_id == archived_run_id,
            "observed": archived_run_id or "missing",
            "threshold": manifest_run_id or "manifest run_id",
        }
    )
    manifest_question = str(manifest.get("question", ""))
    archived_question = str(run_payload.get("question", "")) if isinstance(run_payload, dict) else ""
    checks.append(
        {
            "name": "bundle_manifest_question_matches_run_json",
            "passed": bool(manifest_question) and manifest_question == archived_question,
            "observed": "equal" if manifest_question == archived_question else "changed",
            "threshold": "run.json question",
        }
    )
    checks.append(
        {
            "name": "bundle_manifest_selection_supported",
            "passed": manifest.get("selection") in {"latest", "best", "ready"},
            "observed": manifest.get("selection", "missing"),
            "threshold": "latest, best, or ready",
        }
    )

    manifest_run_json = str(manifest.get("run_json", ""))
    status_run_json = str(manifest.get("status_run_json", ""))
    checks.append(
        {
            "name": "bundle_manifest_status_run_matches_run_json",
            "passed": bool(manifest_run_json) and status_run_json == manifest_run_json,
            "observed": status_run_json or "missing",
            "threshold": manifest_run_json or "manifest run_json",
        }
    )

    selected_run = ""
    if isinstance(status_payload, dict):
        selected = status_payload.get("selected_run", {})
        if isinstance(selected, dict):
            selected_run = str(selected.get("path", ""))
    checks.append(
        {
            "name": "bundle_status_selected_run_matches_manifest",
            "passed": bool(manifest_run_json) and selected_run == manifest_run_json,
            "observed": selected_run or "missing",
            "threshold": manifest_run_json or "manifest run_json",
        }
    )
    _add_archived_run_manifest_checks(archive, names, manifest, run_payload, run_manifest_payload, checks)
    _add_archived_audit_metadata_checks(manifest, names, run_payload, audit_payload, checks)
    _add_archived_readme_metadata_checks(manifest, audit_payload, readme_payload, checks)
    _add_archived_status_metadata_checks(manifest, run_payload, audit_payload, status_payload, checks)


def _bundle_manifest_arcname(rows: Any, label: str, fallback: str) -> str:
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict) and item.get("label") == label:
                arcname = item.get("arcname")
                if isinstance(arcname, str) and arcname and _bundle_arcname_safe(arcname):
                    return arcname
                return ""
    return fallback


def _read_bundle_json_entry(archive: zipfile.ZipFile, names: set[str], arcname: str) -> Any:
    if not arcname or not _bundle_arcname_safe(arcname) or arcname not in names:
        return None
    try:
        return json.loads(archive.read(arcname).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
        return None


def _read_bundle_text_entry(archive: zipfile.ZipFile, names: set[str], arcname: str) -> str | None:
    if not arcname or not _bundle_arcname_safe(arcname) or arcname not in names:
        return None
    try:
        return archive.read(arcname).decode("utf-8")
    except (UnicodeDecodeError, KeyError):
        return None


def _add_archived_readme_metadata_checks(
    bundle_manifest: dict[str, Any],
    audit_payload: Any,
    readme_payload: str | None,
    checks: list[dict[str, Any]],
) -> None:
    if not isinstance(audit_payload, dict) or not isinstance(readme_payload, str):
        return
    try:
        expected = _bundle_readme(bundle_manifest, audit_payload)
        expected_error = ""
    except Exception as exc:
        expected = ""
        expected_error = str(exc)
    checks.append(
        {
            "name": "bundle_readme_expected_parseable",
            "passed": not expected_error,
            "observed": expected_error or "loadable",
            "threshold": "manifest and audit can generate README.md",
        }
    )
    if expected_error:
        return
    checks.append(
        {
            "name": "bundle_readme_matches_manifest_audit",
            "passed": readme_payload == expected,
            "observed": "equal" if readme_payload == expected else "changed",
            "threshold": "README.md generated from archived manifest and audit metadata",
        }
    )


def _add_archived_status_metadata_checks(
    bundle_manifest: dict[str, Any],
    run_payload: Any,
    audit_payload: Any,
    status_payload: Any,
    checks: list[dict[str, Any]],
) -> None:
    if not isinstance(status_payload, dict) or not isinstance(run_payload, dict):
        return
    manifest_run_json = str(bundle_manifest.get("run_json", ""))
    checks.append(
        {
            "name": "bundle_status_selection_matches_manifest",
            "passed": status_payload.get("run_selection") == bundle_manifest.get("selection"),
            "observed": status_payload.get("run_selection", "missing"),
            "threshold": bundle_manifest.get("selection", "manifest selection"),
        }
    )

    expected_summary = _archived_run_summary_from_payload(run_payload)
    for key in ("selected_run", "latest_run"):
        row = status_payload.get(key)
        checks.append(
            {
                "name": f"bundle_status_{key}_parseable",
                "passed": isinstance(row, dict),
                "observed": type(row).__name__,
                "threshold": "object",
            }
        )
        if not isinstance(row, dict):
            continue
        checks.append(
            {
                "name": f"bundle_status_{key}_path_matches_manifest",
                "passed": bool(manifest_run_json) and row.get("path") == manifest_run_json,
                "observed": row.get("path", "missing"),
                "threshold": manifest_run_json or "manifest run_json",
            }
        )
        checks.append(
            {
                "name": f"bundle_status_{key}_summary_matches_run_json",
                "passed": _summary_contains_expected_fields(row.get("summary"), expected_summary),
                "observed": "equal" if _summary_contains_expected_fields(row.get("summary"), expected_summary) else "changed",
                "threshold": "summary generated from archived run.json",
            }
        )

    if isinstance(audit_payload, dict):
        checks.append(
            {
                "name": "bundle_status_audit_matches_audit_json",
                "passed": status_payload.get("audit") == audit_payload,
                "observed": "equal" if status_payload.get("audit") == audit_payload else "changed",
                "threshold": "archived audit.json",
            }
        )
        checks.append(
            {
                "name": "bundle_status_advisories_match_audit_json",
                "passed": status_payload.get("advisories", []) == audit_payload.get("advisories", []),
                "observed": "equal" if status_payload.get("advisories", []) == audit_payload.get("advisories", []) else "changed",
                "threshold": "audit.json advisories",
            }
        )
        checks.append(
            {
                "name": "bundle_status_advisory_actions_match_audit_json",
                "passed": status_payload.get("advisory_actions", []) == audit_payload.get("advisory_actions", []),
                "observed": "equal" if status_payload.get("advisory_actions", []) == audit_payload.get("advisory_actions", []) else "changed",
                "threshold": "audit.json advisory actions",
            }
        )


def _archived_run_summary_from_payload(run_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_payload.get("run_id", ""),
        "question": run_payload.get("question", ""),
        "readiness_score": run_payload.get("metrics", {}).get("readiness_score", 0),
        "claims": len(run_payload.get("claims", [])),
        "evidence": len(run_payload.get("evidence", [])),
        "gaps": run_payload.get("gaps", []),
        "artifacts": run_payload.get("artifacts", {}),
    }


def _summary_contains_expected_fields(summary: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    for key, value in expected.items():
        if summary.get(key) != value:
            return False
    return True


def _add_archived_audit_metadata_checks(
    bundle_manifest: dict[str, Any],
    names: set[str],
    run_payload: Any,
    audit_payload: Any,
    checks: list[dict[str, Any]],
) -> None:
    if not isinstance(run_payload, dict) or not isinstance(audit_payload, dict):
        return
    run_id = str(run_payload.get("run_id", ""))
    audit_run_id = str(audit_payload.get("run_id", ""))
    checks.append(
        {
            "name": "bundle_audit_run_id_matches_run_json",
            "passed": bool(run_id) and audit_run_id == run_id,
            "observed": audit_run_id or "missing",
            "threshold": run_id or "run.json run_id",
        }
    )
    run_question = str(run_payload.get("question", ""))
    audit_question = str(audit_payload.get("question", ""))
    checks.append(
        {
            "name": "bundle_audit_question_matches_run_json",
            "passed": bool(run_question) and audit_question == run_question,
            "observed": "equal" if audit_question == run_question else "changed",
            "threshold": "run.json question",
        }
    )
    checks.append(
        {
            "name": "bundle_manifest_audit_passed_matches_audit",
            "passed": bundle_manifest.get("audit_passed") == audit_payload.get("passed"),
            "observed": str(audit_payload.get("passed", "missing")),
            "threshold": str(bundle_manifest.get("audit_passed", "manifest audit_passed")),
        }
    )
    checks.append(
        {
            "name": "bundle_manifest_readiness_matches_audit",
            "passed": bundle_manifest.get("readiness_score") == audit_payload.get("readiness_score"),
            "observed": audit_payload.get("readiness_score", "missing"),
            "threshold": bundle_manifest.get("readiness_score", "manifest readiness_score"),
        }
    )
    checks.append(
        {
            "name": "bundle_manifest_advisories_match_audit",
            "passed": bundle_manifest.get("advisories", []) == audit_payload.get("advisories", []),
            "observed": "equal" if bundle_manifest.get("advisories", []) == audit_payload.get("advisories", []) else "changed",
            "threshold": "manifest advisories",
        }
    )

    audit_checks_raw = audit_payload.get("checks")
    audit_checks = audit_checks_raw if isinstance(audit_checks_raw, list) else []
    checks.append(
        {
            "name": "bundle_audit_checks_parseable",
            "passed": isinstance(audit_checks_raw, list),
            "observed": type(audit_checks_raw).__name__,
            "threshold": "list",
        }
    )
    if not isinstance(audit_checks_raw, list):
        return
    audit_check_names: list[str] = []
    audit_checks_by_name: dict[str, dict[str, Any]] = {}
    malformed = 0
    duplicates: list[str] = []
    for row in audit_checks:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row.get("name"):
            malformed += 1
            continue
        name = str(row["name"])
        if name in audit_checks_by_name:
            duplicates.append(name)
        audit_check_names.append(name)
        audit_checks_by_name[name] = row
    checks.append(
        {
            "name": "bundle_audit_check_rows_parseable",
            "passed": malformed == 0 and not duplicates,
            "observed": f"malformed={malformed}, duplicates={', '.join(sorted(set(duplicates))) or 'none'}",
            "threshold": "named unique check objects",
        }
    )

    expected, expected_error = _expected_archived_audit_metadata(bundle_manifest, names, run_payload)
    checks.append(
        {
            "name": "bundle_audit_expected_parseable",
            "passed": not expected_error,
            "observed": expected_error or "loadable",
            "threshold": "archived run.json can be audited",
        }
    )
    if expected_error:
        return

    expected_checks = expected["checks"]
    expected_checks_by_name = {str(row["name"]): row for row in expected_checks}
    missing = sorted(set(expected_checks_by_name) - set(audit_checks_by_name))
    unexpected = sorted(set(audit_checks_by_name) - set(expected_checks_by_name))
    checks.append(
        {
            "name": "bundle_audit_check_names_match_run_json",
            "passed": not missing and not unexpected,
            "observed": f"missing={', '.join(missing) or 'none'}; unexpected={', '.join(unexpected) or 'none'}",
            "threshold": "audit checks generated from archived run.json",
        }
    )
    changed: list[str] = []
    for name, expected_check in expected_checks_by_name.items():
        observed_check = audit_checks_by_name.get(name)
        if not isinstance(observed_check, dict):
            continue
        if name in {"paper_artifact_exists", "paper_artifact_structure", "manifest_integrity"}:
            if observed_check.get("passed") != expected_check.get("passed"):
                changed.append(name)
            continue
        if observed_check != expected_check:
            changed.append(name)
    checks.append(
        {
            "name": "bundle_audit_checks_match_run_json",
            "passed": not changed,
            "observed": ", ".join(sorted(changed)) if changed else "equal",
            "threshold": "run-derived audit check payloads",
        }
    )

    failed_checks_raw = audit_payload.get("failed_checks")
    failed_checks = failed_checks_raw if isinstance(failed_checks_raw, list) else []
    failed_parseable = isinstance(failed_checks_raw, list) and all(isinstance(item, str) for item in failed_checks)
    expected_failed = [name for name in audit_check_names if not bool(audit_checks_by_name.get(name, {}).get("passed"))]
    checks.append(
        {
            "name": "bundle_audit_failed_checks_parseable",
            "passed": failed_parseable,
            "observed": type(failed_checks_raw).__name__,
            "threshold": "list of strings",
        }
    )
    if failed_parseable:
        checks.append(
            {
                "name": "bundle_audit_failed_checks_match_checks",
                "passed": failed_checks == expected_failed,
                "observed": ", ".join(failed_checks) if failed_checks else "none",
                "threshold": ", ".join(expected_failed) if expected_failed else "none",
            }
        )
        checks.append(
            {
                "name": "bundle_audit_passed_matches_failed_checks",
                "passed": audit_payload.get("passed") == (not expected_failed),
                "observed": str(audit_payload.get("passed", "missing")),
                "threshold": str(not expected_failed),
            }
        )

    checks.append(
        {
            "name": "bundle_audit_readiness_matches_run_json",
            "passed": audit_payload.get("readiness_score") == expected["readiness_score"],
            "observed": audit_payload.get("readiness_score", "missing"),
            "threshold": expected["readiness_score"],
        }
    )
    checks.append(
        {
            "name": "bundle_audit_advisories_match_run_json",
            "passed": audit_payload.get("advisories", []) == expected["advisories"],
            "observed": "equal" if audit_payload.get("advisories", []) == expected["advisories"] else "changed",
            "threshold": "run-derived advisories",
        }
    )
    checks.append(
        {
            "name": "bundle_audit_advisory_actions_match_advisories",
            "passed": audit_payload.get("advisory_actions", []) == expected["advisory_actions"],
            "observed": "equal" if audit_payload.get("advisory_actions", []) == expected["advisory_actions"] else "changed",
            "threshold": "actions from run-derived advisories",
        }
    )
    checks.append(
        {
            "name": "bundle_audit_next_actions_match_failed_checks",
            "passed": audit_payload.get("next_actions", []) == expected["next_actions"],
            "observed": "equal" if audit_payload.get("next_actions", []) == expected["next_actions"] else "changed",
            "threshold": "actions from failed audit checks",
        }
    )


def _expected_archived_audit_metadata(
    bundle_manifest: dict[str, Any],
    names: set[str],
    run_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        from .audit import _actions_for_failed_checks, _add_alignment_gate, _add_seed_gate, _audit_advisories, _run_from_payload
        from .report import run_evals

        run = _run_from_payload(run_payload)
        evals = run_evals(run)
        checks = list(evals["checks"])
        _add_seed_gate(run, checks)
        should_have_paper = bool(run.discoveries) or run.metrics.get("readiness_score", 0) >= 0.7
        paper_arcname = _bundle_manifest_arcname(bundle_manifest.get("files"), "paper_tex", "")
        checks.append(
            {
                "name": "paper_artifact_exists",
                "passed": (not should_have_paper) or bool(paper_arcname and paper_arcname in names),
                "observed": "present" if paper_arcname and paper_arcname in names else "missing",
                "threshold": "required after confirmed discoveries or readiness>=0.70",
            }
        )
        if should_have_paper or (paper_arcname and paper_arcname in names):
            checks.append(
                {
                    "name": "paper_artifact_structure",
                    "passed": bool(paper_arcname and paper_arcname in names),
                    "observed": "present" if paper_arcname and paper_arcname in names else "missing",
                    "threshold": "article TeX with document body, Results, Experiment Ledger, Evidence Ledger, and Limitations sections",
                }
            )
        _add_alignment_gate(run, checks)
        checks.append(
            {
                "name": "manifest_integrity",
                "passed": True,
                "observed": "passed",
                "threshold": "verify_run_manifest passes",
            }
        )
        advisories = _audit_advisories(run)
        failed = [row["name"] for row in checks if not row["passed"]]
        return {
            "checks": checks,
            "failed_checks": failed,
            "next_actions": _actions_for_failed_checks(failed),
            "advisories": advisories,
            "advisory_actions": [item["action"] for item in advisories if item.get("action")],
            "readiness_score": evals.get("readiness_score", 0),
        }, ""
    except Exception as exc:
        return {}, str(exc)


def _add_archived_run_manifest_checks(
    archive: zipfile.ZipFile,
    names: set[str],
    bundle_manifest: dict[str, Any],
    run_payload: Any,
    run_manifest: Any,
    checks: list[dict[str, Any]],
) -> None:
    if not isinstance(run_payload, dict) or not isinstance(run_manifest, dict):
        return
    checks.append(
        {
            "name": "bundle_run_manifest_run_id_matches_run_json",
            "passed": run_manifest.get("run_id") == run_payload.get("run_id"),
            "observed": str(run_manifest.get("run_id", "")) or "missing",
            "threshold": str(run_payload.get("run_id", "")) or "run.json run_id",
        }
    )
    checks.append(
        {
            "name": "bundle_run_manifest_question_matches_run_json",
            "passed": run_manifest.get("question") == run_payload.get("question"),
            "observed": "equal" if run_manifest.get("question") == run_payload.get("question") else "changed",
            "threshold": "equal",
        }
    )
    checks.append(
        {
            "name": "bundle_run_manifest_schema_version_supported",
            "passed": run_manifest.get("schema_version") == 1,
            "observed": run_manifest.get("schema_version", "missing"),
            "threshold": 1,
        }
    )
    checks.append(
        {
            "name": "bundle_run_manifest_mode_matches_run_json",
            "passed": run_manifest.get("mode", "literature") == run_payload.get("mode", "literature"),
            "observed": f"{run_manifest.get('mode', 'literature')} / {run_payload.get('mode', 'literature')}",
            "threshold": "equal",
        }
    )
    checks.append(
        {
            "name": "bundle_run_manifest_provenance_matches_run_json",
            "passed": run_manifest.get("provenance", {}) == run_payload.get("provenance", {}),
            "observed": "equal" if run_manifest.get("provenance", {}) == run_payload.get("provenance", {}) else "changed",
            "threshold": "equal",
        }
    )
    ledger = run_manifest.get("run_ledger", {})
    checks.append(
        {
            "name": "bundle_run_manifest_ledger_parseable",
            "passed": isinstance(ledger, dict),
            "observed": type(ledger).__name__,
            "threshold": "object",
        }
    )
    if not isinstance(ledger, dict):
        return
    from .audit import _run_from_payload
    from .provenance import run_ledger_digest

    actual = run_ledger_digest(_run_from_payload(run_payload))
    declared_hash = ledger.get("sha256")
    hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
    checks.append(
        {
            "name": "bundle_run_manifest_ledger_sha256_declared",
            "passed": hash_valid,
            "observed": type(declared_hash).__name__ if not isinstance(declared_hash, str) else (declared_hash or "empty"),
            "threshold": "sha256 hex",
        }
    )
    if hash_valid:
        checks.append(
            {
                "name": "bundle_run_manifest_ledger_sha256",
                "passed": actual["sha256"] == declared_hash,
                "observed": actual["sha256"],
                "threshold": declared_hash,
            }
        )
    declared_bytes = ledger.get("bytes")
    bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
    checks.append(
        {
            "name": "bundle_run_manifest_ledger_bytes_declared",
            "passed": bytes_valid,
            "observed": type(declared_bytes).__name__ if type(declared_bytes) is not int else declared_bytes,
            "threshold": "non-negative integer",
        }
    )
    if bytes_valid:
        checks.append(
            {
                "name": "bundle_run_manifest_ledger_bytes",
                "passed": actual["bytes"] == declared_bytes,
                "observed": actual["bytes"],
                "threshold": declared_bytes,
            }
        )
    _add_archived_run_graph_checks(run_payload, checks)
    _add_archived_source_manifest_checks(run_payload, run_manifest, checks)
    _add_archived_artifact_manifest_checks(archive, names, bundle_manifest, run_payload, run_manifest, checks)


def _add_archived_run_graph_checks(run_payload: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    run_sources = run_payload.get("sources", [])
    if not isinstance(run_sources, list):
        run_sources = []
    from .provenance import _discovery_graph_checks, _evidence_graph_checks

    graph_checks = _evidence_graph_checks(run_payload, run_sources)
    graph_checks.extend(_discovery_graph_checks(run_payload, run_sources))
    for check in graph_checks:
        name = check.get("name", "")
        checks.append({**check, "name": f"bundle_{name}"})


def _add_archived_source_manifest_checks(run_payload: dict[str, Any], run_manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    run_sources_raw = run_payload.get("sources", [])
    manifest_sources_raw = run_manifest.get("sources", [])
    run_sources = run_sources_raw if isinstance(run_sources_raw, list) else []
    manifest_sources = manifest_sources_raw if isinstance(manifest_sources_raw, list) else []
    checks.append(
        {
            "name": "bundle_run_sources_parseable",
            "passed": isinstance(run_sources_raw, list),
            "observed": type(run_sources_raw).__name__,
            "threshold": "list",
        }
    )
    checks.append(
        {
            "name": "bundle_run_manifest_sources_parseable",
            "passed": isinstance(manifest_sources_raw, list),
            "observed": type(manifest_sources_raw).__name__,
            "threshold": "list",
        }
    )
    if not isinstance(run_sources_raw, list) or not isinstance(manifest_sources_raw, list):
        return
    checks.append(
        {
            "name": "bundle_run_manifest_source_count_matches_run_json",
            "passed": len(run_sources) == len(manifest_sources),
            "observed": f"{len(manifest_sources)} / {len(run_sources)}",
            "threshold": "equal",
        }
    )
    manifest_by_id = {row.get("id"): row for row in manifest_sources if isinstance(row, dict) and isinstance(row.get("id"), str) and row.get("id")}
    run_ids: set[str] = set()
    from .provenance import source_digest_from_payload

    for index, row in enumerate(run_sources):
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"bundle_run_source_parseable:{index}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id:
            checks.append(
                {
                    "name": f"bundle_run_source_id_declared:{index}",
                    "passed": False,
                    "observed": type(source_id).__name__ if not isinstance(source_id, str) else "empty",
                    "threshold": "non-empty string",
                }
            )
            continue
        run_ids.add(source_id)
        manifest_row = manifest_by_id.get(source_id)
        checks.append(
            {
                "name": f"bundle_run_manifest_source_tracked:{source_id}",
                "passed": isinstance(manifest_row, dict),
                "observed": "present" if isinstance(manifest_row, dict) else "missing",
                "threshold": "source id present in archived run manifest",
            }
        )
        if not isinstance(manifest_row, dict):
            continue
        expected = source_digest_from_payload(row)
        declared_hash = manifest_row.get("text_sha256")
        hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
        checks.append(
            {
                "name": f"bundle_run_manifest_source_text_sha256_declared:{source_id}",
                "passed": hash_valid,
                "observed": type(declared_hash).__name__ if not isinstance(declared_hash, str) else (declared_hash or "empty"),
                "threshold": "sha256 hex",
            }
        )
        declared_bytes = manifest_row.get("text_bytes")
        bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
        checks.append(
            {
                "name": f"bundle_run_manifest_source_text_bytes_declared:{source_id}",
                "passed": bytes_valid,
                "observed": type(declared_bytes).__name__ if type(declared_bytes) is not int else declared_bytes,
                "threshold": "non-negative integer",
            }
        )
        keys = ["title", "kind", "url", "metadata"]
        if hash_valid:
            keys.append("text_sha256")
        if bytes_valid:
            keys.append("text_bytes")
        if "created_at" in row:
            keys.append("created_at")
        for key in keys:
            checks.append(
                {
                    "name": f"bundle_run_manifest_source_{key}_matches:{source_id}",
                    "passed": manifest_row.get(key) == expected[key],
                    "observed": "equal" if manifest_row.get(key) == expected[key] else "changed",
                    "threshold": "equal",
                }
            )
    for source_id in sorted(manifest_by_id):
        if source_id in run_ids:
            continue
        checks.append(
            {
                "name": f"bundle_run_manifest_source_declared:{source_id}",
                "passed": False,
                "observed": source_id,
                "threshold": "present in archived run.json sources",
            }
        )


def _add_archived_artifact_manifest_checks(
    archive: zipfile.ZipFile,
    names: set[str],
    bundle_manifest: dict[str, Any],
    run_payload: dict[str, Any],
    run_manifest: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    artifacts = run_manifest.get("artifacts", {})
    checks.append(
        {
            "name": "bundle_run_manifest_artifacts_parseable",
            "passed": isinstance(artifacts, dict),
            "observed": type(artifacts).__name__,
            "threshold": "object",
        }
    )
    if not isinstance(artifacts, dict):
        return
    labels_by_artifact = {
        "json": "run_json",
        "html": "html_report",
        "markdown": "markdown_report",
        "graph": "evidence_graph",
        "evals": "evals",
        "trace": "trace",
        "experiments": "experiments",
        "discoveries": "discoveries",
        "paper": "paper_tex",
        "pdf": "paper_pdf",
        "review": "paper_review",
    }
    files = bundle_manifest.get("files", [])
    declared_run_artifacts = run_payload.get("artifacts", {}) if isinstance(run_payload.get("artifacts"), dict) else {}
    for artifact, label in labels_by_artifact.items():
        row = artifacts.get(artifact)
        arcname = _bundle_manifest_arcname(files, label, "")
        carried = bool(arcname and arcname in names)
        if carried and declared_run_artifacts.get(artifact):
            checks.append(
                {
                    "name": f"bundle_run_manifest_tracks_artifact:{artifact}",
                    "passed": isinstance(row, dict),
                    "observed": "present" if isinstance(row, dict) else "missing",
                    "threshold": "present in archived run manifest artifacts",
                }
            )
        if row is None:
            continue
        checks.append(
            {
                "name": f"bundle_run_manifest_artifact_parseable:{artifact}",
                "passed": isinstance(row, dict),
                "observed": type(row).__name__,
                "threshold": "object",
            }
        )
        if not isinstance(row, dict):
            continue
        if not carried:
            continue
        data = archive.read(arcname)
        declared_bytes = row.get("bytes")
        bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
        checks.append(
            {
                "name": f"bundle_run_manifest_artifact_bytes_declared:{artifact}",
                "passed": bytes_valid,
                "observed": type(declared_bytes).__name__ if type(declared_bytes) is not int else declared_bytes,
                "threshold": "non-negative integer",
            }
        )
        if bytes_valid:
            checks.append(
                {
                    "name": f"bundle_run_manifest_artifact_bytes:{artifact}",
                    "passed": declared_bytes == len(data),
                    "observed": len(data),
                    "threshold": declared_bytes,
                }
            )
        declared_hash = row.get("sha256")
        hash_required = not bool(row.get("mutable", False))
        hash_present = declared_hash is not None
        hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
        checks.append(
            {
                "name": f"bundle_run_manifest_artifact_sha256_declared:{artifact}",
                "passed": hash_valid if (hash_required or hash_present) else True,
                "observed": (
                    type(declared_hash).__name__
                    if (hash_required or hash_present) and not isinstance(declared_hash, str)
                    else (declared_hash or ("not required" if not hash_required else "empty"))
                ),
                "threshold": "sha256 hex for immutable artifacts",
            }
        )
        if hash_valid:
            actual_hash = _sha256_bytes(data)
            checks.append(
                {
                    "name": f"bundle_run_manifest_artifact_sha256:{artifact}",
                    "passed": actual_hash == declared_hash,
                    "observed": actual_hash,
                    "threshold": declared_hash,
                }
            )
        if artifact in {"html", "markdown", "graph", "evals", "experiments", "discoveries", "paper", "review", "pdf", "trace"}:
            _add_archived_sidecar_ledger_checks(artifact, data, run_payload, checks)


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _add_archived_sidecar_ledger_checks(artifact: str, data: bytes, run_payload: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    if artifact == "trace":
        checks.extend(_archived_trace_artifact_structure_checks(data, str(run_payload.get("run_id", ""))))
        return
    if artifact == "pdf":
        header = data[:5]
        checks.append(
            {
                "name": "bundle_pdf_artifact_parseable",
                "passed": True,
                "observed": "bytes",
                "threshold": "%PDF header",
            }
        )
        checks.append(
            {
                "name": "bundle_pdf_artifact_header",
                "passed": header.startswith(b"%PDF"),
                "observed": header.decode("latin1", errors="replace") or "empty",
                "threshold": "%PDF",
            }
        )
        return
    if artifact == "review":
        try:
            review = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            checks.append(
                {
                    "name": "bundle_review_artifact_parseable",
                    "passed": False,
                    "observed": str(exc),
                    "threshold": "UTF-8 text",
                }
            )
            return
        checks.append(
            {
                "name": "bundle_review_artifact_parseable",
                "passed": True,
                "observed": "text",
                "threshold": "UTF-8 text",
            }
        )
        checks.append(
            {
                "name": "bundle_review_artifact_nonempty",
                "passed": bool(review.strip()),
                "observed": "nonempty" if review.strip() else "empty",
                "threshold": "non-empty review text",
            }
        )
        return
    if artifact == "paper":
        try:
            tex = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            checks.append(
                {
                    "name": "bundle_paper_artifact_parseable",
                    "passed": False,
                    "observed": str(exc),
                    "threshold": "UTF-8 LaTeX",
                }
            )
            return
        checks.append(
            {
                "name": "bundle_paper_artifact_parseable",
                "passed": True,
                "observed": "text",
                "threshold": "UTF-8 LaTeX",
            }
        )
        from .provenance import PAPER_ARTIFACT_REQUIRED_MARKERS

        missing = [marker for marker in PAPER_ARTIFACT_REQUIRED_MARKERS if marker not in tex]
        checks.append(
            {
                "name": "bundle_paper_artifact_latex_structure",
                "passed": not missing,
                "observed": ", ".join(missing) if missing else "present",
                "threshold": ", ".join(PAPER_ARTIFACT_REQUIRED_MARKERS),
            }
        )
        return
    if artifact in {"html", "markdown"}:
        try:
            payload = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            checks.append(
                {
                    "name": f"bundle_{artifact}_sidecar_parseable",
                    "passed": False,
                    "observed": str(exc),
                    "threshold": "UTF-8 text",
                }
            )
            return
        expected, expected_error = _expected_archived_generated_sidecars(run_payload)
        checks.append(
            {
                "name": f"bundle_{artifact}_sidecar_parseable",
                "passed": True,
                "observed": "text",
                "threshold": "UTF-8 text",
            }
        )
        if expected_error:
            checks.append(
                {
                    "name": f"bundle_{artifact}_sidecar_expected_parseable",
                    "passed": False,
                    "observed": expected_error,
                    "threshold": "loadable archived run payload",
                }
            )
        else:
            from .provenance import _normalise_generated_report

            observed = _normalise_generated_report(artifact, payload)
            threshold = _normalise_generated_report(artifact, str(expected.get(artifact)))
            checks.append(
                {
                    "name": f"bundle_{artifact}_sidecar_matches_run",
                    "passed": observed == threshold,
                    "observed": "equal" if observed == threshold else "changed",
                    "threshold": f"generated {artifact} report from archived run.json",
                }
            )
        return
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        checks.append(
            {
                "name": f"bundle_{artifact}_sidecar_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "valid JSON",
            }
        )
        return
    if artifact in {"graph", "evals"}:
        expected, expected_error = _expected_archived_generated_sidecars(run_payload)
        checks.append(
            {
                "name": f"bundle_{artifact}_sidecar_parseable",
                "passed": isinstance(payload, dict),
                "observed": type(payload).__name__,
                "threshold": "object",
            }
        )
        if expected_error:
            checks.append(
                {
                    "name": f"bundle_{artifact}_sidecar_expected_parseable",
                    "passed": False,
                    "observed": expected_error,
                    "threshold": "loadable archived run payload",
                }
            )
        elif isinstance(payload, dict):
            checks.append(
                {
                    "name": f"bundle_{artifact}_sidecar_matches_run",
                    "passed": payload == expected.get(artifact),
                    "observed": "equal" if payload == expected.get(artifact) else "changed",
                    "threshold": f"generated {artifact}.json from archived run.json",
                }
            )
        return
    if artifact == "experiments":
        expected = run_payload.get("experiments", [])
        checks.append(
            {
                "name": "bundle_experiments_sidecar_parseable",
                "passed": isinstance(payload, list),
                "observed": type(payload).__name__,
                "threshold": "list",
            }
        )
        if isinstance(payload, list):
            checks.append(
                {
                    "name": "bundle_experiments_sidecar_matches_run",
                    "passed": payload == expected,
                    "observed": "equal" if payload == expected else "changed",
                    "threshold": "archived run.json experiments",
                }
            )
        return
    from .provenance import _discoveries_sidecar_payload_checks

    checks.extend(_discoveries_sidecar_payload_checks(payload, run_payload, prefix="bundle_"))


def _archived_trace_artifact_structure_checks(data: bytes, expected_run_id: str) -> list[dict[str, Any]]:
    try:
        lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        return [
            {
                "name": "bundle_trace_artifact_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "JSONL trace",
            }
        ]
    checks: list[dict[str, Any]] = [
        {
            "name": "bundle_trace_artifact_nonempty",
            "passed": bool(lines),
            "observed": len(lines),
            "threshold": "at least one event",
        }
    ]
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"{index}:{exc.msg}")
            continue
        if not isinstance(row, dict):
            parse_errors.append(f"{index}:{type(row).__name__}")
            continue
        rows.append(row)
    checks.append(
        {
            "name": "bundle_trace_artifact_parseable",
            "passed": not parse_errors,
            "observed": ", ".join(parse_errors[:5]) if parse_errors else "JSONL objects",
            "threshold": "one JSON object per line",
        }
    )
    if not rows:
        return checks
    phases = {"start", "end", "event", "error"}
    missing_fields = [
        str(index)
        for index, row in enumerate(rows)
        if not all(key in row for key in ("trace_id", "run_id", "span_id", "phase", "name", "time_unix_ms", "attributes"))
    ]
    checks.append(
        {
            "name": "bundle_trace_artifact_fields",
            "passed": not missing_fields,
            "observed": ", ".join(missing_fields[:5]) if missing_fields else "present",
            "threshold": "trace_id, run_id, span_id, phase, name, time_unix_ms, attributes",
        }
    )
    wrong_run_ids = sorted({str(row.get("run_id", "")) for row in rows if row.get("run_id") != expected_run_id})
    checks.append(
        {
            "name": "bundle_trace_artifact_run_id",
            "passed": bool(expected_run_id) and not wrong_run_ids,
            "observed": ", ".join(wrong_run_ids[:5]) if wrong_run_ids else (expected_run_id or "missing"),
            "threshold": expected_run_id or "run.json run_id",
        }
    )
    bad_phases = sorted({str(row.get("phase", "")) for row in rows if row.get("phase") not in phases})
    checks.append(
        {
            "name": "bundle_trace_artifact_phase",
            "passed": not bad_phases,
            "observed": ", ".join(bad_phases[:5]) if bad_phases else "valid",
            "threshold": ", ".join(sorted(phases)),
        }
    )
    return checks


def _expected_archived_generated_sidecars(run_payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        from .audit import _run_from_payload
        from .report import claim_graph, html_report, markdown_report, run_evals

        run = _run_from_payload(run_payload)
        return {
            "html": html_report(run),
            "markdown": markdown_report(run),
            "graph": claim_graph(run),
            "evals": run_evals(run),
        }, ""
    except Exception as exc:
        return {}, str(exc)


def _bundle_embedded_file_row(label: str, arcname: str, data: bytes) -> dict[str, Any]:
    return {
        "label": label,
        "arcname": arcname,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _verify_bundle_file_rows(
    archive: zipfile.ZipFile,
    names: set[str],
    rows: list[Any],
    declared_arcnames: set[str],
    checks: list[dict[str, Any]],
    *,
    entry_kind: str,
) -> None:
    prefix = f"bundle_{entry_kind}"
    label_counts: dict[str, int] = {}
    arcname_counts: dict[str, int] = {}
    malformed_labels = 0
    for item in rows:
        if not isinstance(item, dict):
            malformed_labels += 1
            continue
        label_value = item.get("label")
        if not isinstance(label_value, str) or not label_value:
            malformed_labels += 1
            continue
        label_counts[label_value] = label_counts.get(label_value, 0) + 1
        arcname_value = item.get("arcname")
        if isinstance(arcname_value, str) and arcname_value:
            arcname_counts[arcname_value] = arcname_counts.get(arcname_value, 0) + 1
    duplicate_labels = sorted(label for label, count in label_counts.items() if count > 1)
    duplicate_arcnames = sorted(arcname for arcname, count in arcname_counts.items() if count > 1)
    already_declared = sorted(set(arcname_counts) & declared_arcnames)
    checks.append(
        {
            "name": f"{prefix}_labels_declared",
            "passed": malformed_labels == 0,
            "observed": malformed_labels if malformed_labels else "all declared",
            "threshold": "non-empty string label per row",
        }
    )
    checks.append(
        {
            "name": f"{prefix}_labels_unique",
            "passed": not duplicate_labels,
            "observed": ", ".join(duplicate_labels[:8]) if duplicate_labels else "unique",
            "threshold": "one row per logical bundle label",
        }
    )
    checks.append(
        {
            "name": f"{prefix}_arcnames_unique",
            "passed": not duplicate_arcnames,
            "observed": ", ".join(duplicate_arcnames[:8]) if duplicate_arcnames else "unique",
            "threshold": "one row per archive path",
        }
    )
    checks.append(
        {
            "name": f"{prefix}_arcnames_unclaimed",
            "passed": not already_declared,
            "observed": ", ".join(already_declared[:8]) if already_declared else "unclaimed",
            "threshold": "not declared by an earlier manifest section",
        }
    )
    for item in rows:
        if not isinstance(item, dict):
            checks.append(
                {
                    "name": f"bundle_{entry_kind}_entry_parseable",
                    "passed": False,
                    "observed": type(item).__name__,
                    "threshold": "object",
                }
            )
            continue
        arcname_value = item.get("arcname")
        arcname = arcname_value if isinstance(arcname_value, str) else ""
        if arcname:
            declared_arcnames.add(arcname)
        label_value = item.get("label")
        label = label_value if isinstance(label_value, str) and label_value else (arcname or "unknown")
        path_safe = bool(arcname) and _bundle_arcname_safe(arcname)
        checks.append(
            {
                "name": f"{prefix}_path_safe:{label}",
                "passed": path_safe,
                "observed": arcname if isinstance(arcname_value, str) else type(arcname_value).__name__,
                "threshold": "relative path without traversal",
            }
        )
        present = bool(path_safe and arcname in names)
        checks.append(
            {
                "name": f"{prefix}_exists:{label}",
                "passed": present,
                "observed": arcname if present else "missing",
                "threshold": arcname or "arcname",
            }
        )
        declared_hash = item.get("sha256")
        hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
        checks.append(
            {
                "name": f"{prefix}_sha256_declared:{label}",
                "passed": hash_valid,
                "observed": declared_hash or "missing",
                "threshold": "sha256 hex",
            }
        )
        declared_bytes = item.get("bytes")
        bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
        checks.append(
            {
                "name": f"{prefix}_bytes_declared:{label}",
                "passed": bytes_valid,
                "observed": type(declared_bytes).__name__ if type(declared_bytes) is not int else declared_bytes,
                "threshold": "non-negative integer",
            }
        )
        if not present:
            continue
        data = archive.read(arcname)
        if bytes_valid:
            checks.append(
                {
                    "name": f"{prefix}_bytes:{label}",
                    "passed": declared_bytes == len(data),
                    "observed": len(data),
                    "threshold": declared_bytes,
                }
            )
        if hash_valid:
            actual = _sha256_bytes(data)
            checks.append(
                {
                    "name": f"{prefix}_sha256:{label}",
                    "passed": actual == declared_hash,
                    "observed": actual,
                    "threshold": declared_hash,
                }
            )


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _verify_bundle_required_labels(
    rows: list[Any],
    required: set[str],
    checks: list[dict[str, Any]],
    *,
    entry_kind: str,
) -> None:
    labels = {
        item.get("label")
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("label"), str) and item.get("label")
    }
    missing = sorted(required - labels)
    checks.append(
        {
            "name": f"bundle_{entry_kind}_labels_complete",
            "passed": not missing,
            "observed": ", ".join(missing) if missing else "complete",
            "threshold": ", ".join(sorted(required)),
        }
    )


def _verify_bundle_required_label_arcnames(
    rows: list[Any],
    expected: dict[str, str],
    checks: list[dict[str, Any]],
    *,
    entry_kind: str,
) -> None:
    by_label = {
        item.get("label"): item
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("label"), str) and item.get("label")
    }
    mismatched: list[str] = []
    for label, arcname in sorted(expected.items()):
        row = by_label.get(label)
        if not isinstance(row, dict) or row.get("arcname") != arcname:
            observed = row.get("arcname") if isinstance(row, dict) else "missing"
            mismatched.append(f"{label}={observed}")
    checks.append(
        {
            "name": f"bundle_{entry_kind}_label_arcnames_canonical",
            "passed": not mismatched,
            "observed": ", ".join(mismatched[:8]) if mismatched else "canonical",
            "threshold": ", ".join(f"{label}={arcname}" for label, arcname in sorted(expected.items())),
        }
    )


def _bundle_arcname_safe(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and "\\" not in name and ".." not in path.parts


def _bundle_recreate_command(*, selection: str, selected_lookup: bool) -> str:
    if selected_lookup:
        return f"mechferret bundle --select {_policy(selection)}"
    return "mechferret bundle"


def _bundle_verify_result(path: Path, checks: list[dict[str, Any]], next_actions: list[str]) -> dict[str, Any]:
    failed = [check["name"] for check in checks if not check.get("passed")]
    return {
        "path": str(path) if str(path) != "." else "",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "next_actions": [] if not failed else next_actions,
    }


def print_verify_bundle_result(result: dict[str, Any]) -> None:
    print(f"Bundle verify: {'PASS' if result.get('passed') else 'WARN'}")
    if result.get("path"):
        print(f"Bundle: {result['path']}")
    checks = [check for check in result.get("checks", []) if isinstance(check, dict)]
    failed = [check for check in checks if not check.get("passed")]
    if checks:
        print(f"Checks: {len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("Failed checks:")
        for check in failed[:12]:
            threshold = f" / {check.get('threshold')}" if check.get("threshold") is not None else ""
            print(f"  - {check.get('name', '')}: {check.get('observed')}{threshold}")
        if len(failed) > 12:
            print(f"  - ...and {len(failed) - 12} more; rerun with --json for the complete check list.")
    elif checks:
        print("Detailed checks omitted; rerun with --json for the complete check list.")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"][:8]:
            print(f"  - {action}")


def verify_run_artifacts(
    run_json: str | Path | None = None,
    *,
    runs_root: str | Path = "runs",
    repair: bool = False,
) -> dict[str, Any]:
    from .provenance import refresh_run_manifest, verify_run_manifest

    target = _path(run_json) if run_json else _latest_run_json(runs_root)
    if target is None or not target.is_file():
        return {
            "path": "",
            "manifest": "",
            "passed": False,
            "checks": [{"name": "run_json_exists", "passed": False, "observed": "missing"}],
            "failed_checks": ["run_json_exists"],
            "next_actions": ["Run `mechferret quickstart --run` to create a dossier before verifying artifacts."],
        }
    try:
        result = verify_run_manifest(target)
    except Exception as exc:  # noqa: BLE001 - public status/verify should report broken ledgers
        result = {
            "path": str(target),
            "manifest": "",
            "passed": False,
            "checks": [{"name": "run_json_parseable", "passed": False, "observed": str(exc), "threshold": "object run ledger"}],
            "failed_checks": ["run_json_parseable"],
            "next_actions": ["Regenerate or inspect the run artifact before verifying its manifest."],
        }
    if not _flag(repair) or result.get("passed"):
        return _annotate_manifest_repairability(target, result)
    failed = list(result.get("failed_checks", []))
    blocked = _manifest_repair_blockers(failed)
    if blocked:
        return {
            **result,
            "repair_attempted": False,
            "repair_blocked": True,
            "repair_blockers": blocked,
            "next_actions": [
                *result.get("next_actions", []),
                "Repair was not run because one or more failures require regenerating or manually inspecting the dossier.",
            ],
        }
    refresh_run_manifest(target)
    repaired = verify_run_manifest(target)
    return {
        **repaired,
        "repair_attempted": True,
        "repaired": repaired.get("passed", False),
        "before_failed_checks": failed,
    }


def _annotate_manifest_repairability(target: Path, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("passed"):
        return {**result, "repairable": False}
    failed = list(result.get("failed_checks", []))
    blockers = _manifest_repair_blockers(failed)
    if blockers:
        return {**result, "repairable": False, "repair_blockers": blockers}
    command = f"mechferret verify {shlex.quote(str(target))} --repair --strict"
    actions = list(result.get("next_actions", []))
    actions.append(f"Run `{command}` to refresh manifest.json without rerunning the dossier.")
    return {
        **result,
        "next_actions": _dedupe_actions(actions),
        "repairable": True,
        "repair_command": command,
    }


def _manifest_repair_blockers(failed_checks: list[str]) -> list[str]:
    repairable = _manifest_repairable_failures(failed_checks)
    return [name for name in failed_checks if name not in repairable]


def _manifest_repairable_failures(failed_checks: list[str]) -> set[str]:
    failed = set(failed_checks)
    repairable = {
        name
        for name in failed
        if name.startswith(
            (
                "manifest_tracks_declared_artifact:",
                "manifest_artifact_declared:",
                "artifact_bytes:",
                "artifact_bytes_declared:",
                "run_ledger_sha256_declared",
                "run_ledger_bytes_declared",
                "mode_matches_manifest",
                "provenance_matches_manifest",
            )
        )
    }
    if "manifest_schema_version_supported" in failed:
        repairable.add("manifest_schema_version_supported")
    for name in failed:
        if not name.startswith("artifact_exists:"):
            continue
        artifact = name.split(":", 1)[1]
        if f"manifest_artifact_declared:{artifact}" in failed:
            repairable.add(name)
    return repairable


def print_verify_result(result: dict[str, Any]) -> None:
    print(f"Verify: {'PASS' if result.get('passed') else 'WARN'}")
    if result.get("path"):
        print(f"Run: {result['path']}")
    if result.get("manifest"):
        print(f"Manifest: {result['manifest']}")
    if result.get("repair_attempted"):
        print(f"Manifest refreshed: {'yes' if result.get('repaired') else 'no'}")
    elif result.get("repair_blocked"):
        print("Manifest refreshed: blocked")
    checks = [check for check in result.get("checks", []) if isinstance(check, dict)]
    failed = [check for check in checks if not check.get("passed")]
    if checks:
        print(f"Checks: {len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("Failed checks:")
        for check in failed[:12]:
            threshold = f" / {check.get('threshold')}" if check.get("threshold") is not None else ""
            print(f"  - {check.get('name', '')}: {check.get('observed')}{threshold}")
        if len(failed) > 12:
            print(f"  - ...and {len(failed) - 12} more; rerun with --json for the complete check list.")
    elif checks:
        print("Detailed checks omitted; rerun with --json for the complete check list.")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"][:8]:
            print(f"  - {action}")


def resolve_artifact(
    target: str = "quickstart",
    *,
    runs_root: str | Path = "runs",
    project_root: str | Path = "projects/openvla_sae",
    selection: str = "latest",
) -> dict[str, Any]:
    requested = _text(target).strip() or "quickstart"
    selection = _policy(selection)
    runs_root = _path(runs_root, "runs")
    project_root = _path(project_root, "projects/openvla_sae")
    artifact_aliases = {
        "html": "report",
        "report_html": "report",
        "md": "markdown",
        "report_md": "markdown",
        "report_markdown": "markdown",
        "tex": "paper",
        "review_md": "review",
        "zip": "bundle",
        "run_manifest": "manifest",
    }
    requested = artifact_aliases.get(requested, requested)
    if requested in {"all", "artifacts"}:
        return _artifact_index(runs_root=runs_root, project_root=project_root, selection=selection)
    if requested == "openvla":
        path = _openvla_artifact_path(project_root)
        reason = "OpenVLA project scaffold" if path.name == "README.md" else "OpenVLA quickstart index"
        return _artifact_result(requested, path, reason, selection=selection, scope="workspace")

    selection_artifacts = {
        "quickstart": "quickstart",
        "ci": "ci",
        "report": "report",
        "pdf": "paper",
        "review": "paper",
        "manifest": "manifest",
        "markdown": "markdown",
        "graph": "graph",
        "evals": "evals",
        "trace": "trace",
        "experiments": "experiments",
        "discoveries": "discoveries",
    }
    require_artifact = selection_artifacts.get(requested) if selection != "latest" else None
    selection_result = None
    if require_artifact:
        selection_result = select_run_artifact(runs_root=runs_root, policy=selection, require_artifact=require_artifact)
        latest_run = Path(selection_result["path"]) if selection_result.get("path") else None
    elif selection != "latest":
        selection_result = select_run_artifact(runs_root=runs_root, policy=selection)
        latest_run = Path(selection_result["path"]) if selection_result.get("path") else None
    else:
        latest_run = _selected_run_json(runs_root, selection=selection, require_artifact=require_artifact)
    selection_actions = selection_result.get("next_actions", []) if selection_result else []
    if selection_result and not selection_result.get("path"):
        if requested == "paper":
            selection_actions = [
                *selection_actions,
                f"Run `mechferret paper --select {selection}` to generate a run-bound draft for the selected policy.",
            ]
        elif requested == "review":
            selection_actions = [
                *selection_actions,
                f"Run `mechferret paper --select {selection}` first, then `mechferret review-paper --select {selection}`.",
            ]
    reason_prefix = "latest" if selection == "latest" else f"{selection}-selected"
    selected_run = str(latest_run) if latest_run else ""
    if requested == "run":
        return _artifact_result(requested, latest_run, f"{reason_prefix} run artifact", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "quickstart":
        path = _latest_quickstart_index(runs_root, latest_run, allow_global_fallback=selection == "latest")
        return _artifact_result(requested, path, f"{reason_prefix} quickstart index", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "ci":
        path = _latest_ci_quickstart_index(runs_root, latest_run, allow_global_fallback=selection == "latest")
        return _artifact_result(requested, path, f"{reason_prefix} CI quickstart index", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "report":
        path = _latest_report(latest_run)
        return _artifact_result(requested, path, f"{reason_prefix} HTML report", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "paper":
        path = _latest_paper(latest_run)
        return _artifact_result(requested, path, f"{reason_prefix} paper scaffold", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "pdf":
        path = _latest_run_artifact(latest_run, "pdf", "paper/main.pdf")
        return _artifact_result(requested, path, f"{reason_prefix} compiled paper", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "review":
        path = _latest_run_artifact(latest_run, "review", "paper/review.md")
        return _artifact_result(requested, path, f"{reason_prefix} paper review", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "bundle":
        path = _latest_run_artifact(latest_run, "bundle", "mechferret-bundle.zip")
        return _artifact_result(requested, path, f"{reason_prefix} shareable research bundle", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    if requested == "manifest":
        path = _latest_run_artifact(latest_run, "manifest", "manifest.json")
        return _artifact_result(requested, path, f"{reason_prefix} run manifest", selection=selection, selected_run=selected_run, extra_next_actions=selection_actions)
    run_artifacts = {
        "markdown": ("markdown", "report.md"),
        "graph": ("graph", "graph.json"),
        "evals": ("evals", "evals.json"),
        "trace": ("trace", "trace.jsonl"),
        "experiments": ("experiments", "experiments.json"),
        "discoveries": ("discoveries", "discoveries.json"),
    }
    if requested in run_artifacts:
        key, default_name = run_artifacts[requested]
        return _artifact_result(
            requested,
            _latest_run_artifact(latest_run, key, default_name),
            f"{reason_prefix} {requested} artifact",
            selection=selection,
            selected_run=selected_run,
            extra_next_actions=selection_actions,
        )

    direct = _path(requested)
    return _artifact_result(requested, direct, "explicit path", selection=selection, scope="path")


def print_artifact_result(result: dict[str, Any]) -> None:
    if result.get("artifacts"):
        summary = result.get("artifact_summary") or _artifact_summary(result.get("artifacts", {}))
        artifact_readiness = result.get("artifact_readiness") or _artifact_readiness(summary)
        dossier = summary["groups"]["dossier"]
        discovery = summary["groups"]["discovery"]
        sharing = summary["groups"]["sharing"]
        print(f"Artifacts: {'found' if result['exists'] else 'missing'}")
        print(f"Target: {result['target']}")
        print(f"Reason: {result['reason']}")
        if result.get("selected_run"):
            print(f"Selected run: {result['selected_run']}")
        print(
            "Summary: "
            f"{summary['found']}/{summary['total']} found; "
            f"dossier {dossier['found']}/{dossier['total']}; "
            f"discovery {discovery['found']}/{discovery['total']}; "
            f"sharing {sharing['found']}/{sharing['total']}"
        )
        run_ready = _mapping(artifact_readiness.get("run"))
        share_ready = _mapping(artifact_readiness.get("sharing"))
        setup_ready = _mapping(artifact_readiness.get("setup"))
        complete = bool(result.get("complete"))
        print(
            "Readiness: "
            f"run={'READY' if run_ready.get('ok') else 'BLOCKED'}; "
            f"share={'READY' if share_ready.get('ok') else 'BLOCKED'}; "
            f"setup={'READY' if setup_ready.get('ok') else 'BLOCKED'}"
        )
        print(f"Complete: {'yes' if complete else 'no'}")
        for name, item in result["artifacts"].items():
            marker = "found" if item["exists"] else "missing"
            path = item["path"] or "(none)"
            scope = _text(item.get("scope"))
            scope_label = f" [{scope}]" if scope and scope != "run" else ""
            print(f"{marker:8} {name}{scope_label}: {path}")
            if not item["exists"] and item.get("next_actions"):
                print(f"         action: {item['next_actions'][0]}")
        if result.get("next_actions"):
            print("Next actions:")
            for action in result["next_actions"]:
                print(f"  - {action}")
        return
    print(f"Artifact: {'found' if result['exists'] else 'missing'}")
    print(f"Target: {result['target']}")
    print(f"Path: {result['path']}")
    print(f"Reason: {result['reason']}")
    if result.get("scope"):
        print(f"Scope: {result['scope']}")
    if result.get("selection"):
        print(f"Selection: {result['selection']}")
    if result.get("selected_run"):
        print(f"Selected run: {result['selected_run']}")
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"]:
            print(f"  - {action}")


def open_artifact(result: dict[str, Any]) -> bool:
    if result.get("artifacts"):
        for target in ("report", "quickstart", "bundle", "ci", "paper", "review", "pdf", "markdown", "graph", "evals", "trace", "openvla", "run"):
            item = result["artifacts"].get(target)
            if item and item.get("exists"):
                return open_artifact(item)
        return False
    if not result.get("exists"):
        return False
    import webbrowser

    return bool(webbrowser.open(Path(result["path"]).resolve().as_uri()))


def _run_demo_quickstart(*, out_dir: str | Path, db_path: str | Path) -> dict[str, Any]:
    from .audit import audit_run_artifact
    from .controller import MechFerret
    from .paper import write_paper_from_artifact
    from .provenance import refresh_run_manifest

    out = Path(out_dir)
    notes = init_project_notes(_quickstart_project_root(out))
    run = MechFerret(db_path).run(
        QUICKSTART_DEMO_QUESTION,
        source_paths=[str(example_corpus_path())],
        out_dir=out,
        provider="local",
        include_memory=False,
        allow_seed_corpus=True,
    )
    run_json = out / "run.json"
    paper = write_paper_from_artifact(run_json, out_dir=out / "paper", provider="local")
    refresh_run_manifest(run_json)
    audit = audit_run_artifact(run_json)
    steps = [
        {"name": "project_notes", "ok": Path(notes["path"]).exists(), "detail": notes["path"]},
        {"name": "demo", "ok": run_json.exists(), "detail": str(run_json)},
        {"name": "paper", "ok": bool(paper.get("tex")) and Path(paper["tex"]).exists(), "detail": paper.get("tex", "")},
        {"name": "audit", "ok": audit["passed"], "detail": ", ".join(audit.get("failed_checks", [])) or "passed"},
    ]
    result = {
        "mode": "demo",
        "ok": all(step["ok"] for step in steps),
        "run_id": run.run_id,
        "readiness_score": run.metrics.get("readiness_score", 0),
        "steps": steps,
        "project_notes": {
            "path": notes["path"],
            "created": bool(notes.get("created")),
            "detected_stack": notes.get("detected_stack", []),
        },
        "artifacts": {
            "run_json": str(run_json),
            "report": run.artifacts.get("html", ""),
            "paper": paper.get("tex", ""),
            "project_notes": notes["path"],
        },
        "audit": audit,
        "next_actions": audit.get("next_actions", []),
        "suggested_next_actions": [
            "Run `mechferret status` to see the selected dossier and remaining artifacts.",
            "Run `mechferret support` to write a redacted support report if anything looks off.",
            "Run `mechferret open report --select best --browser` to inspect the local report.",
            "Run `mechferret bundle --select best` to package the dossier for sharing.",
        ],
    }
    _write_quickstart_artifacts(out, result)
    return result


def _quickstart_project_root(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    parts = path.parts
    if "runs" not in parts:
        return path.parent if path.parent != Path("") else Path(".")
    index = parts.index("runs")
    if index == 0:
        return Path(".")
    return Path(*parts[:index])


def _run_openvla_quickstart(*, project_root: str | Path, force: bool) -> dict[str, Any]:
    from .openvla_sae import init_project, status

    root = Path(project_root)
    init = init_project(root, force=force)
    st = status(project_root=root)
    scaffold_ok = bool(init.get("ok") or st.get("ready_local"))
    scaffold_detail = f"{len(init['files_written'])} files written"
    if not init.get("ok") and st.get("ready_local"):
        scaffold_detail = "existing scaffold ready"
    steps = [
        {"name": "init", "ok": scaffold_ok, "detail": scaffold_detail},
        {"name": "status", "ok": st["ready_local"], "detail": f"{sum(st['files'].values())}/{len(st['files'])} files"},
    ]
    result = {
        "mode": "openvla",
        "ok": all(step["ok"] for step in steps),
        "steps": steps,
        "artifacts": {"project_root": str(root)},
        "status": st,
        "next_actions": [] if scaffold_ok and st.get("ready_local") else init.get("next_actions", []) or st.get("next_actions", []),
    }
    if root.exists():
        _write_openvla_quickstart_artifacts(root, result)
    return result


def _run_ci_quickstart(*, out_dir: str | Path, db_path: str | Path) -> dict[str, Any]:
    from .audit import audit_run_artifact

    out = Path(out_dir)
    steps: list[dict[str, Any]] = []

    doc = doctor()
    steps.append(_ci_step("doctor", doc["strict_passed"], "core doctor checks"))

    demo = _run_demo_quickstart(out_dir=out, db_path=db_path)
    steps.append(_ci_step("demo_quickstart", demo["ok"], demo.get("artifacts", {}).get("quickstart_markdown", str(out))))

    tests_dir = Path("tests")
    if tests_dir.exists():
        steps.append(_command_step("unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"]))
    else:
        steps.append(_ci_step("unit_tests", True, "skipped: no tests directory in this install"))

    compile_targets = [str(path) for path in (Path("projects/openvla_sae/src"), Path("mechferret"), tests_dir) if path.exists()]
    if not compile_targets:
        compile_targets = [str(Path(__file__).resolve().parent)]
    steps.append(_command_step("compileall", [sys.executable, "-m", "compileall", "-q", *compile_targets]))

    run_json = out / "run.json"
    audit_json = audit_run_artifact(run_json)
    steps.append(_ci_step("audit_json", "checks" in audit_json and "failed_checks" in audit_json, str(run_json)))

    selected_runs_root = out
    bundle = bundle_run_artifacts(runs_root=selected_runs_root, selection="best", out=out)
    bundle_path = Path(bundle.get("path", ""))
    steps.append(_ci_step("bundle_artifacts", bool(bundle.get("ok")) and bundle_path.exists(), str(bundle_path) if bundle_path else "missing"))

    verify_json = verify_run_artifacts(run_json)
    steps.append(_ci_step("verify_manifest", verify_json["passed"], ", ".join(verify_json.get("failed_checks", [])) or "passed"))

    bundle_verify = verify_bundle_artifacts(runs_root=selected_runs_root, selection="best")
    steps.append(_ci_step("verify_bundle", bundle_verify["passed"], ", ".join(bundle_verify.get("failed_checks", [])) or "passed"))

    strict_doc = doctor()
    steps.append(_ci_step("doctor_strict", strict_doc["strict_passed"], "release-critical checks"))

    strict_audit = audit_run_artifact(run_json)
    steps.append(_ci_step("audit_strict", strict_audit["passed"], ", ".join(strict_audit.get("failed_checks", [])) or "passed"))

    all_integrations = doctor()
    steps.append(
        _ci_step(
            "doctor_all_integrations",
            all_integrations["all_integrations_passed"],
            "optional exhaustive integration audit",
            optional=True,
        )
    )

    result = {
        "mode": "ci",
        "ok": all(step["ok"] or step.get("optional", False) for step in steps),
        "steps": steps,
        "artifacts": {
            "run_json": str(run_json),
            "demo_quickstart": demo.get("artifacts", {}).get("quickstart_markdown", ""),
            "paper": demo.get("artifacts", {}).get("paper", ""),
            "report": demo.get("artifacts", {}).get("report", ""),
            "bundle": str(bundle_path) if bundle_path else "",
        },
        "audit": strict_audit,
        "next_actions": [
            step["detail"]
            for step in steps
            if not step["ok"] and not step.get("optional") and step.get("detail")
        ],
        "optional_next_actions": all_integrations.get("all_integrations_next_actions", []),
    }
    _write_ci_quickstart_artifacts(out, result)
    return result


def _ci_step(name: str, ok: bool, detail: str, *, optional: bool = False) -> dict[str, Any]:
    step = {"name": name, "ok": bool(ok), "detail": detail}
    if optional:
        step["optional"] = True
    return step


def _command_step(name: str, command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return _ci_step(name, False, f"timed out: {' '.join(command)}")
    stdout = proc.stdout.strip().splitlines()
    stderr = proc.stderr.strip().splitlines()
    if proc.returncode == 0:
        ran = next((line for line in stderr if line.startswith("Ran ")), "")
        verdict = next((line for line in reversed(stderr) if line == "OK" or line.startswith("FAILED")), "")
        summary = f"{ran}; {verdict}" if ran and verdict else ((stderr or stdout or ["ok"])[-1])
    else:
        summary = ((stderr or stdout) or [f"exit {proc.returncode}: {' '.join(command)}"])[-1]
    return _ci_step(name, proc.returncode == 0, summary[:240])


def _write_quickstart_artifacts(out: Path, result: dict[str, Any]) -> None:
    json_path = out / "quickstart.json"
    md_path = out / "QUICKSTART.md"
    result["artifacts"]["quickstart_json"] = str(json_path)
    result["artifacts"]["quickstart_markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    status = "PASS" if result["ok"] else "WARN"
    steps = "\n".join(
        f"- [{'x' if step['ok'] else ' '}] {step['name']}: `{step['detail']}`"
        for step in result["steps"]
    )
    artifacts = "\n".join(
        f"- {name}: `{path}`"
        for name, path in result["artifacts"].items()
        if path
    )
    notes = _mapping(result.get("project_notes"))
    notes_path = _text(notes.get("path", ""))
    notes_status = "created" if notes.get("created") else "preserved"
    notes_section = f"- {notes_status}: `{notes_path}`" if notes_path else "- Missing."
    actions = "\n".join(f"- {action}" for action in result.get("next_actions", [])) or "- None. The local demo quickstart audit passed."
    suggested = "\n".join(f"- {action}" for action in result.get("suggested_next_actions", [])) or "- None."
    md_path.write_text(
        f"""# MechFerret Quickstart Run

Status: {status}

Run ID: `{result.get('run_id', '')}`
Readiness: {float(result.get('readiness_score', 0)):.3f}

## Steps
{steps}

## Artifacts
{artifacts}

## Project Notes
{notes_section}

## Next Actions
{actions}

## Suggested Next Actions
{suggested}
""",
        encoding="utf-8",
    )


def _write_ci_quickstart_artifacts(out: Path, result: dict[str, Any]) -> None:
    json_path = out / "ci_quickstart.json"
    md_path = out / "CI_QUICKSTART.md"
    result["artifacts"]["ci_json"] = str(json_path)
    result["artifacts"]["ci_markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    steps = "\n".join(
        f"- [{'x' if step['ok'] else ' '}] {step['name']}{' (optional)' if step.get('optional') else ''}: `{step['detail']}`"
        for step in result["steps"]
    )
    artifacts = "\n".join(
        f"- {name}: `{path}`"
        for name, path in result["artifacts"].items()
        if path
    )
    actions = "\n".join(f"- {action}" for action in result.get("next_actions", [])) or "- None. Release-critical offline gates passed."
    optional = "\n".join(f"- {action}" for action in result.get("optional_next_actions", [])[:8]) or "- None."
    md_path.write_text(
        f"""# MechFerret CI Quickstart

Status: {'PASS' if result['ok'] else 'WARN'}

## Release-Critical Steps
{steps}

## Artifacts
{artifacts}

## Required Next Actions
{actions}

## Optional Integration Notes
{optional}
""",
        encoding="utf-8",
    )


def _write_openvla_quickstart_artifacts(root: Path, result: dict[str, Any]) -> None:
    json_path = root / "quickstart.json"
    md_path = root / "QUICKSTART.md"
    result["artifacts"]["quickstart_json"] = str(json_path)
    result["artifacts"]["quickstart_markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    status_payload = result.get("status", {})
    steps = "\n".join(
        f"- [{'x' if step['ok'] else ' '}] {step['name']}: `{step['detail']}`"
        for step in result["steps"]
    )
    actions = "\n".join(f"- {action}" for action in result.get("next_actions", [])) or "- Add a manifest, cache activations, train an SAE, then run eval/features/dossier."
    md_path.write_text(
        f"""# OpenVLA SAE Quickstart

Status: {'PASS' if result['ok'] else 'WARN'}

Project root: `{root}`
Project files: {sum(status_payload.get('files', {}).values())}/{len(status_payload.get('files', {}))}

## Steps
{steps}

## Next Actions
{actions}

## Common Commands
```bash
mechferret sae openvla status --project-root {root}
mechferret sae openvla create-manifest --image-dir data/openvla_images --manifest data/openvla_sae_phase1.jsonl
mechferret sae openvla validate-manifest --manifest data/openvla_sae_phase1.jsonl
mechferret sae openvla commands --project-root {root}
```
""",
        encoding="utf-8",
    )


def _bundle_output_path(out: str | Path | None, run_dir: Path) -> Path:
    if out is None:
        return run_dir / "mechferret-bundle.zip"
    path = _path(out, run_dir / "mechferret-bundle.zip")
    if path.suffix.lower() != ".zip":
        return path / "mechferret-bundle.zip"
    return path


def _bundle_candidates(
    run_json: Path,
    payload: dict[str, Any],
    *,
    notes_root: str | Path,
    project_root: str | Path,
) -> list[tuple[str, Path, str]]:
    from .provenance import resolve_run_artifact_path

    run_dir = run_json.parent
    artifact_map = _mapping(payload.get("artifacts"))

    def candidate(label: str, artifact_key: str, default: str, arcname: str) -> tuple[str, Path, str]:
        artifact = artifact_map.get(artifact_key)
        path = resolve_run_artifact_path(run_dir, artifact) if isinstance(artifact, (str, Path)) and artifact else run_dir / default
        return label, path, arcname

    candidates = [
        ("run_json", run_json, "run/run.json"),
        candidate("html_report", "html", "report.html", "run/report.html"),
        candidate("markdown_report", "markdown", "report.md", "run/report.md"),
        candidate("evidence_graph", "graph", "graph.json", "run/graph.json"),
        candidate("evals", "evals", "evals.json", "run/evals.json"),
        candidate("trace", "trace", "trace.jsonl", "run/trace.jsonl"),
        candidate("experiments", "experiments", "experiments.json", "run/experiments.json"),
        candidate("discoveries", "discoveries", "discoveries.json", "run/discoveries.json"),
        candidate("paper_tex", "paper", "paper/main.tex", "paper/main.tex"),
        candidate("paper_pdf", "pdf", "paper/main.pdf", "paper/main.pdf"),
        candidate("paper_review", "review", "paper/review.md", "paper/review.md"),
        candidate("run_manifest", "manifest", "manifest.json", "run/manifest.json"),
        ("quickstart_markdown", run_dir / "QUICKSTART.md", "quickstart/QUICKSTART.md"),
        ("quickstart_json", run_dir / "quickstart.json", "quickstart/quickstart.json"),
        ("ci_markdown", run_dir / "CI_QUICKSTART.md", "quickstart/CI_QUICKSTART.md"),
        ("ci_json", run_dir / "ci_quickstart.json", "quickstart/ci_quickstart.json"),
        ("project_notes", _path(notes_root, ".") / "MECHFERRET.md", "project/MECHFERRET.md"),
        ("openvla_quickstart", _path(project_root, "projects/openvla_sae") / "QUICKSTART.md", "openvla/QUICKSTART.md"),
        ("openvla_quickstart_json", _path(project_root, "projects/openvla_sae") / "quickstart.json", "openvla/quickstart.json"),
    ]
    return candidates


def _unique_arcname(arcname: str, used: set[str]) -> str:
    path = Path(arcname)
    candidate = path.as_posix()
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = path.stem
    suffix = path.suffix
    parent = path.parent.as_posix()
    for index in range(2, 1000):
        name = f"{stem}-{index}{suffix}"
        candidate = name if parent == "." else f"{parent}/{name}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError(f"could not allocate bundle arcname for {arcname}")


def _bundle_readme(manifest: dict[str, Any], audit: dict[str, Any]) -> str:
    files = "\n".join(
        f"- `{_text(item.get('arcname'))}` ({_text(item.get('label'))})"
        for item in _items(manifest.get("files"))
        if isinstance(item, dict)
    )
    missing = "\n".join(f"- {_text(label)}" for label in _items(manifest.get("missing"))) or "- None"
    actions = "\n".join(f"- {_text(action)}" for action in _items(audit.get("next_actions", []))) or "- None"
    advisories = "\n".join(
        f"- `{_text(item.get('name'))}`: {_text(item.get('observed'))}. {_text(item.get('action'))}"
        for item in _items(audit.get("advisories", []))
        if isinstance(item, dict)
    ) or "- None"
    return f"""# MechFerret Research Bundle

Run: `{manifest.get('run_id', '')}`

Question: {manifest.get('question', '')}

Audit: {'PASS' if audit.get('passed') else 'WARN'}
Readiness: {_number(manifest.get('readiness_score', 0)):.3f}

## Files
{files or "- No run files were bundled."}

## Missing Optional Files
{missing}

## Next Actions
{actions}

## Advisories
{advisories}
"""


def _run_list_entry(path: Path, *, include_audit: bool) -> dict[str, Any]:
    payload = _json_object_from_file(path)
    if not payload:
        return {
            "ok": False,
            "path": str(path),
            "mtime": path.stat().st_mtime if path.exists() else 0,
            "error": "unreadable or non-object run artifact",
        }
    artifacts = _mapping(payload.get("artifacts"))
    metrics = _mapping(payload.get("metrics"))
    artifact_flags = _run_artifact_flags(path, artifacts)
    artifact_summary = _artifact_summary({name: {"exists": exists} for name, exists in artifact_flags.items()}, include_setup=False)
    entry = {
        "ok": True,
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "run_id": _text(payload.get("run_id", "")),
        "question": _text(payload.get("question", "")),
        "mode": _text(payload.get("mode", "")),
        "created_at": _text(payload.get("created_at", "")),
        "readiness_score": _number(metrics.get("readiness_score", 0)),
        "claims": len(_items(payload.get("claims", []))),
        "evidence": len(_items(payload.get("evidence", []))),
        "gaps": len(_items(payload.get("gaps", []))),
        "artifacts": artifact_flags,
        "artifact_summary": artifact_summary,
        "artifact_readiness": _artifact_readiness(artifact_summary),
    }
    if include_audit:
        from .audit import audit_run_artifact

        try:
            audit = audit_run_artifact(path)
            entry["audit"] = {
                "passed": audit["passed"],
                "failed_checks": audit.get("failed_checks", []),
                "advisories": audit.get("advisories", []),
                "readiness_score": audit.get("readiness_score", entry["readiness_score"]),
            }
        except Exception as exc:  # noqa: BLE001 - listing should keep scanning other runs
            entry["audit"] = {"passed": False, "failed_checks": ["audit_error"], "error": str(exc)}
    return entry


def _openvla_artifact_path(project_root: Path) -> Path:
    quickstart_path = project_root / "QUICKSTART.md"
    if quickstart_path.exists():
        return quickstart_path
    try:
        from .openvla_sae import status

        scaffold_status = status(project_root=project_root)
    except Exception:
        return quickstart_path
    if scaffold_status.get("ready_local") and (project_root / "README.md").exists():
        return project_root / "README.md"
    return quickstart_path


def _run_artifact_flags(run_json: Path, artifacts: dict[str, Any]) -> dict[str, bool]:
    return {
        "run": True,
        "quickstart": _payload_artifact_exists(run_json, artifacts, "quickstart_markdown", "QUICKSTART.md"),
        "ci": _payload_artifact_exists(run_json, artifacts, "ci_markdown", "CI_QUICKSTART.md"),
        "report": _payload_artifact_exists(run_json, artifacts, "html", "report.html"),
        "markdown": _payload_artifact_exists(run_json, artifacts, "markdown", "report.md"),
        "graph": _payload_artifact_exists(run_json, artifacts, "graph", "graph.json"),
        "evals": _payload_artifact_exists(run_json, artifacts, "evals", "evals.json"),
        "experiments": _payload_artifact_exists(run_json, artifacts, "experiments", "experiments.json"),
        "discoveries": _payload_artifact_exists(run_json, artifacts, "discoveries", "discoveries.json"),
        "paper": _payload_artifact_exists(run_json, artifacts, "paper", "paper/main.tex"),
        "pdf": _payload_artifact_exists(run_json, artifacts, "pdf", "paper/main.pdf"),
        "review": _payload_artifact_exists(run_json, artifacts, "review", "paper/review.md"),
        "bundle": _payload_artifact_exists(run_json, artifacts, "bundle", "mechferret-bundle.zip"),
        "manifest": _payload_artifact_exists(run_json, artifacts, "manifest", "manifest.json"),
        "trace": _payload_artifact_exists(run_json, artifacts, "trace", "trace.jsonl"),
    }


def _run_json_candidates(root: Path) -> list[Path]:
    root = _path(root, "runs")
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.name == "run.json" else []
    return sorted((path for path in root.rglob("run.json") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)


def _run_selection_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    audit = _mapping(row.get("audit"))
    artifact_count = sum(1 for exists in _mapping(row.get("artifacts", {})).values() if exists)
    readiness = _number(audit.get("readiness_score", row.get("readiness_score", 0)))
    advisory_penalty = _advisory_penalty(_items(audit.get("advisories", [])))
    return (
        1.0 if audit.get("passed") else 0.0,
        readiness,
        -advisory_penalty,
        float(artifact_count),
        _number(row.get("mtime", 0)),
    )


def _advisory_penalty(advisories: list[dict[str, Any]]) -> float:
    weights = {"warning": 1.0, "info": 0.25}
    return sum(weights.get(_text(item.get("severity", "info")), 0.5) for item in advisories if isinstance(item, dict))


def _payload_artifact_exists(run_json: Path, artifacts: dict[str, Any], key: str, default_name: str) -> bool:
    from .provenance import resolve_run_artifact_path

    artifact = artifacts.get(key)
    if isinstance(artifact, (str, Path)) and artifact:
        path = resolve_run_artifact_path(run_json.parent, artifact)
        if path.exists():
            return True
    return (run_json.parent / default_name).exists()


def _artifact_result(
    target: str,
    path: Path | None,
    reason: str,
    *,
    selection: str | None = None,
    selected_run: str = "",
    scope: str = "run",
    extra_next_actions: list[str] | None = None,
) -> dict[str, Any]:
    exists = bool(path and path.exists())
    next_actions = []
    if not exists:
        next_actions.extend(_items(extra_next_actions))
        select_flag = f" --select {selection}" if selection else ""
        if target == "paper" and selected_run:
            next_actions.append(f"Run `mechferret paper{select_flag}` to generate a run-bound draft from the selected dossier.")
        elif target == "review" and selected_run:
            next_actions.append(f"Run `mechferret review-paper{select_flag}` with a configured provider to critique the run-bound paper.")
        elif target == "bundle" and selected_run:
            next_actions.append(f"Run `mechferret bundle{select_flag}` to package the selected dossier for sharing.")
        elif target == "pdf" and selected_run:
            next_actions.append(f"Run `mechferret paper{select_flag} --compile` to create a compiled PDF.")
        elif target == "quickstart":
            if not any("mechferret quickstart --run" in action for action in next_actions):
                next_actions.append("Run `mechferret quickstart --run` to create a fresh local quickstart dossier.")
        elif target in {"report", "run", "markdown", "graph", "evals", "trace"} and not selected_run:
            next_actions.append("Run `mechferret quickstart --run` to create a demo dossier.")
        elif target in {"report", "markdown", "graph", "evals", "trace"}:
            next_actions.append("Rerun the selected dossier to regenerate missing run artifacts.")
        elif target == "review":
            next_actions.append("Run `mechferret review-paper` with a configured provider to critique the run-bound paper.")
        elif target == "bundle":
            next_actions.append("Run `mechferret bundle` to package the selected dossier for sharing.")
        elif target == "pdf":
            next_actions.append("Run `mechferret paper --compile` to create a compiled PDF.")
        elif target == "ci":
            next_actions.append("Run `mechferret quickstart --mode ci --run` to create a CI summary.")
        elif target == "openvla":
            next_actions.append("Run `mechferret quickstart --mode openvla --run` to scaffold OpenVLA artifacts.")
        elif target in {"experiments", "discoveries"}:
            next_actions.append("Run `mechferret discover --skill ioi-circuit` to create discovery artifacts.")
        elif target == "manifest":
            next_actions.append("Rerun the dossier with the current MechFerret version to create manifest.json.")
        elif reason == "explicit path":
            next_actions.append("Check the path, or run `mechferret open all` to list known artifacts.")
    result = {
        "target": target,
        "path": str(path) if path else "",
        "exists": exists,
        "ok": exists,
        "reason": reason,
        "scope": scope,
        "next_actions": _dedupe_actions(next_actions),
    }
    if selection is not None:
        result["selection"] = selection
    if selected_run:
        result["selected_run"] = selected_run
    return result


def _artifact_index(*, runs_root: str | Path, project_root: str | Path, selection: str = "latest") -> dict[str, Any]:
    selection = _policy(selection)
    runs_root = _path(runs_root, "runs")
    project_root = _path(project_root, "projects/openvla_sae")
    latest_run = _selected_run_json(runs_root, selection=selection)
    reason_prefix = "latest" if selection == "latest" else f"{selection}-selected"
    selected_run = str(latest_run) if latest_run else ""
    openvla_path = _openvla_artifact_path(project_root)
    openvla_reason = "OpenVLA project scaffold" if openvla_path.name == "README.md" else "OpenVLA quickstart index"
    quickstart_path = _latest_quickstart_index(runs_root, latest_run, allow_global_fallback=True)
    ci_path = _latest_ci_quickstart_index(runs_root, latest_run, allow_global_fallback=True)
    quickstart_reason = f"{reason_prefix} quickstart index"
    if quickstart_path is not None and latest_run is not None and quickstart_path.parent != latest_run.parent:
        quickstart_reason = "workspace quickstart index"
    ci_reason = f"{reason_prefix} CI quickstart index"
    if ci_path is not None and latest_run is not None and ci_path.parent != latest_run.parent:
        ci_reason = "workspace CI quickstart index"

    def artifact(target: str, path: Path | None, reason: str, *, scope: str = "run") -> dict[str, Any]:
        return _artifact_result(target, path, reason, selection=selection, selected_run=selected_run, scope=scope)

    artifacts = {
        "quickstart": artifact("quickstart", quickstart_path, quickstart_reason, scope="workspace"),
        "ci": artifact("ci", ci_path, ci_reason, scope="workspace"),
        "report": artifact("report", _latest_report(latest_run), f"{reason_prefix} HTML report"),
        "markdown": artifact("markdown", _latest_run_artifact(latest_run, "markdown", "report.md"), f"{reason_prefix} Markdown report"),
        "graph": artifact("graph", _latest_run_artifact(latest_run, "graph", "graph.json"), f"{reason_prefix} evidence graph"),
        "evals": artifact("evals", _latest_run_artifact(latest_run, "evals", "evals.json"), f"{reason_prefix} self-checks"),
        "trace": artifact("trace", _latest_run_artifact(latest_run, "trace", "trace.jsonl"), f"{reason_prefix} trace"),
        "experiments": artifact("experiments", _latest_run_artifact(latest_run, "experiments", "experiments.json"), f"{reason_prefix} experiment records"),
        "discoveries": artifact("discoveries", _latest_run_artifact(latest_run, "discoveries", "discoveries.json"), f"{reason_prefix} discoveries"),
        "paper": artifact("paper", _latest_run_paper(latest_run), f"{reason_prefix} run paper scaffold"),
        "review": artifact("review", _latest_run_artifact(latest_run, "review", "paper/review.md"), f"{reason_prefix} paper review"),
        "bundle": artifact("bundle", _latest_run_artifact(latest_run, "bundle", "mechferret-bundle.zip"), f"{reason_prefix} shareable research bundle"),
        "manifest": artifact("manifest", _latest_run_artifact(latest_run, "manifest", "manifest.json"), f"{reason_prefix} run manifest"),
        "pdf": artifact("pdf", _latest_run_artifact(latest_run, "pdf", "paper/main.pdf"), f"{reason_prefix} compiled paper"),
        "run": artifact("run", latest_run, f"{reason_prefix} run artifact"),
        "openvla": artifact("openvla", openvla_path, openvla_reason, scope="workspace"),
    }
    summary = _artifact_summary(artifacts)
    artifact_readiness = _artifact_readiness(summary)
    run_ready = bool(_mapping(artifact_readiness.get("run")).get("ok"))
    share_ready = bool(_mapping(artifact_readiness.get("sharing")).get("ok"))
    setup_ready = bool(_mapping(artifact_readiness.get("setup")).get("ok"))
    complete = run_ready and share_ready and setup_ready
    return {
        "target": "all",
        "path": "",
        "exists": any(item["exists"] for item in artifacts.values()),
        "ok": any(item["exists"] for item in artifacts.values()),
        "complete": complete,
        "run_ready": run_ready,
        "share_ready": share_ready,
        "setup_ready": setup_ready,
        "reason": "artifact index",
        "selection": selection,
        "selected_run": selected_run,
        "artifacts": artifacts,
        "artifact_summary": summary,
        "artifact_readiness": artifact_readiness,
        "next_actions": _artifact_index_next_actions(artifacts, selection=selection, selected_run=selected_run),
    }


def _selected_run_json(runs_root: str | Path, *, selection: str = "latest", require_artifact: str | None = None) -> Path | None:
    selection = _policy(selection)
    if selection == "latest":
        return _latest_run_json(runs_root)
    selected = select_run_artifact(runs_root=runs_root, policy=selection, require_artifact=require_artifact)
    return Path(selected["path"]) if selected.get("path") else None


def _latest_run_json(runs_root: str | Path) -> Path | None:
    from .audit import latest_run_json

    root = _path(runs_root, "runs")
    if root.is_file():
        return root if root.name == "run.json" else None
    return latest_run_json(root)


def _latest_quickstart_index(runs_root: str | Path, latest_run: Path | None, *, allow_global_fallback: bool = True) -> Path | None:
    if latest_run is not None:
        sibling = latest_run.parent / "QUICKSTART.md"
        if sibling.exists():
            return sibling
    if not allow_global_fallback:
        return None
    root = _path(runs_root, "runs")
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("QUICKSTART.md") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_ci_quickstart_index(runs_root: str | Path, latest_run: Path | None, *, allow_global_fallback: bool = True) -> Path | None:
    if latest_run is not None:
        sibling = latest_run.parent / "CI_QUICKSTART.md"
        if sibling.exists():
            return sibling
    if not allow_global_fallback:
        return None
    root = _path(runs_root, "runs")
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("CI_QUICKSTART.md") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_report(latest_run: Path | None) -> Path | None:
    return _latest_run_artifact(latest_run, "html", "report.html")


def _latest_run_artifact(latest_run: Path | None, key: str, default_name: str) -> Path | None:
    from .provenance import resolve_run_artifact_path

    if latest_run is None:
        return None
    payload = _json_object_from_file(latest_run)
    artifact = _mapping(payload.get("artifacts")).get(key)
    if isinstance(artifact, (str, Path)) and artifact:
        path = resolve_run_artifact_path(latest_run.parent, artifact)
        if path.exists():
            return path
    sibling = latest_run.parent / default_name
    return sibling if sibling.exists() else None


def _latest_paper(latest_run: Path | None) -> Path | None:
    return _latest_run_paper(latest_run)


def _latest_run_paper(latest_run: Path | None) -> Path | None:
    return _latest_run_artifact(latest_run, "paper", "paper/main.tex")


def _paper_generator_ok() -> bool:
    from .models import ResearchPlan, ResearchRun
    from .paper import latex_from_run

    run = ResearchRun(
        run_id="doctor",
        question="Doctor scaffold?",
        created_at="1970-01-01T00:00:00+00:00",
        plan=ResearchPlan("Doctor scaffold?", [], ""),
        sources=[],
        evidence=[],
        claims=[],
        contradictions=[],
        gaps=[],
        answer="",
        metrics={},
    )
    tex = latex_from_run(run)
    return (
        "\\documentclass{article}" in tex
        and "\\section{Evidence Ledger}" in tex
        and "\\section{Experiment Ledger}" in tex
        and "TODO:" not in tex
        and "Confirmed Mechanisms" not in tex
    )


def _dedupe_actions(actions: list[str]) -> list[str]:
    result = []
    seen = set()
    for action in actions:
        if not action or action in seen:
            continue
        seen.add(action)
        result.append(action)
    return result


def _actions_not_repeated(actions: list[str], prior_actions: list[str]) -> list[str]:
    prior_keys = {_action_key(action) for action in prior_actions if action}
    result: list[str] = []
    seen = set(prior_keys)
    for action in actions:
        key = _action_key(action)
        if not action or key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _action_key(action: str) -> str:
    match = re.search(r"`([^`]+)`", action)
    command = match.group(1) if match else action
    return re.sub(r"\s+", " ", command.strip())


def _openvla_project_check() -> dict[str, Any]:
    from .openvla_sae import REQUIRED_FILES, status

    st = status()
    present = sum(1 for ok in st["files"].values() if ok)
    total = len(REQUIRED_FILES)
    passed = st["ready_local"] or st["template_available"]
    detail = f"{present}/{total} files"
    if not st["ready_local"] and st["template_available"]:
        detail += "; packaged template available"
    return check(
        "openvla_sae_project",
        passed,
        detail,
        "Run `mechferret sae openvla init` to scaffold the OpenVLA SAE workflow.",
    )


def _openvla_manifest_check() -> dict[str, Any]:
    from .openvla_sae import status

    manifest = status()["manifest"]
    exists = bool(manifest["exists"])
    valid_rows = int(manifest["valid_rows"])
    return check(
        "openvla_manifest",
        exists and valid_rows > 0 and not manifest["errors"] and not manifest["missing_images"],
        f"{manifest['path']} ({valid_rows} valid rows)",
        "Create a real OpenVLA manifest with `mechferret sae openvla create-manifest`.",
        optional=True,
    )


def _latest_run_audit_check() -> dict[str, Any]:
    from .audit import audit_run_artifact, latest_run_json

    latest = latest_run_json()
    if latest is None:
        return check("latest_run_audit", False, "no runs/**/run.json", "Run `mechferret demo` or `mechferret discover` to create a dossier.", optional=True)
    try:
        audit = audit_run_artifact(latest)
    except Exception as exc:  # noqa: BLE001 - doctor should keep reporting other checks
        return check("latest_run_audit", False, f"{latest}: {str(exc)[:120]}", "Inspect or remove the broken latest run artifact.", optional=True)
    failed = [item["name"] for item in audit["checks"] if not item["passed"]]
    detail = f"{latest}: {'passed' if not failed else 'failed ' + ', '.join(failed[:4])}"
    return check(
        "latest_run_audit",
        not failed,
        detail,
        "Run `mechferret audit` and address failed dossier gates.",
        optional=True,
    )


def memory_summary(db_path: str | Path) -> dict[str, int]:
    memory = ResearchMemory(_path(db_path, ".mechferret/memory.sqlite"))
    try:
        runs = memory.conn.execute("select count(*) from runs").fetchone()[0]
        claims = memory.conn.execute("select count(*) from claims").fetchone()[0]
        sources = memory.conn.execute("select count(*) from sources").fetchone()[0]
        return {"runs": runs, "claims": claims, "sources": sources}
    finally:
        memory.close()


def memory_recent(db_path: str | Path, limit: int = 10) -> list[dict[str, Any]]:
    memory = ResearchMemory(_path(db_path, ".mechferret/memory.sqlite"))
    try:
        rows = memory.conn.execute(
            "select id, question, metrics_json, artifacts_json, created_at from runs order by created_at desc limit ?",
            (ResearchMemory._limit(limit, 10),),
        ).fetchall()
        recent = []
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"])
            except (TypeError, json.JSONDecodeError):
                metrics = {}
            try:
                artifacts = json.loads(row["artifacts_json"])
            except (TypeError, json.JSONDecodeError):
                artifacts = {}
            recent.append(
                {
                    "id": row["id"] if isinstance(row["id"], str) else "",
                    "question": row["question"] if isinstance(row["question"], str) else "",
                    "metrics": metrics if isinstance(metrics, dict) else {},
                    "artifacts": artifacts if isinstance(artifacts, dict) else {},
                    "created_at": row["created_at"] if isinstance(row["created_at"], str) else "",
                }
            )
        return recent
    finally:
        memory.close()


def memory_clear(db_path: str | Path) -> None:
    path = _path(db_path, ".mechferret/memory.sqlite")
    if path.exists():
        path.unlink()


def summarize_run_artifact(path: str | Path) -> dict[str, Any]:
    run_json = _path(path)
    payload = _json_object_from_file(run_json)
    metrics = _mapping(payload.get("metrics"))
    artifacts = _mapping(payload.get("artifacts", {}))
    artifact_flags = _run_artifact_flags(run_json, artifacts) if payload else {"run": bool(run_json.exists())}
    artifact_summary = _artifact_summary({name: {"exists": exists} for name, exists in artifact_flags.items()}, include_setup=False)
    artifact_readiness = _artifact_readiness(artifact_summary)
    audit: dict[str, Any] = {}
    next_actions: list[str] = []
    if payload:
        from .audit import audit_run_artifact

        try:
            audit_result = audit_run_artifact(run_json)
            audit = {
                "passed": audit_result.get("passed", False),
                "failed_checks": audit_result.get("failed_checks", []),
                "advisories": audit_result.get("advisories", []),
                "readiness_score": audit_result.get("readiness_score", _number(metrics.get("readiness_score", 0))),
            }
            next_actions.extend(audit_result.get("next_actions", []))
        except Exception as exc:  # noqa: BLE001 - summaries should still render malformed runs
            audit = {"passed": False, "failed_checks": ["audit_error"], "error": str(exc)}
    if payload and not _mapping(artifact_readiness.get("sharing")).get("ok"):
        next_actions.append("Run `mechferret open all` to inspect missing share artifacts.")
    return {
        "run_id": _text(payload.get("run_id", "")),
        "question": _text(payload.get("question", "")),
        "readiness_score": _number(metrics.get("readiness_score", 0)),
        "claims": len(_items(payload.get("claims", []))),
        "evidence": len(_items(payload.get("evidence", []))),
        "gaps": [_text(item) for item in _items(payload.get("gaps", [])) if _text(item)],
        "artifacts": artifacts,
        "artifact_presence": artifact_flags,
        "artifact_summary": artifact_summary,
        "artifact_readiness": artifact_readiness,
        "audit": audit,
        "next_actions": _dedupe_actions(next_actions),
    }
