from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .models import ResearchRun, Source, utc_now

MANIFEST_NAME = "manifest.json"
MUTABLE_ARTIFACTS = {"json", "trace", "manifest", "paper", "pdf", "review", "bundle"}
PAPER_ARTIFACT_REQUIRED_MARKERS = (
    "\\documentclass",
    "\\begin{document}",
    "\\end{document}",
    "\\section{Results}",
    "\\section{Experiment Ledger}",
    "\\section{Evidence Ledger}",
    "\\section{Limitations}",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {_string(key) or str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) or value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else 0.0
    try:
        return str(value)
    except Exception:
        return ""


def source_digest(source: Source) -> dict[str, Any]:
    text = _string(getattr(source, "text", ""))
    metadata = _dict(getattr(source, "metadata", {}))
    return {
        "id": _string(getattr(source, "id", "")),
        "title": _string(getattr(source, "title", "")),
        "kind": _string(getattr(source, "kind", "document")) or "document",
        "url": _string(getattr(source, "url", "")),
        "text_sha256": sha256_text(text),
        "text_bytes": len(text.encode("utf-8")),
        "created_at": _string(getattr(source, "created_at", "")),
        "metadata": metadata,
    }


def write_run_manifest(run: ResearchRun, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    manifest_path = out / MANIFEST_NAME
    if not isinstance(getattr(run, "artifacts", None), dict):
        run.artifacts = {}
    run.artifacts["manifest"] = str(manifest_path)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": _string(getattr(run, "run_id", "")),
        "question": _string(getattr(run, "question", "")),
        "mode": _string(getattr(run, "mode", "literature")) or "literature",
        "provenance": _dict(getattr(run, "provenance", {})),
        "run_ledger": run_ledger_digest(run),
        "sources": [digest for source in _list(getattr(run, "sources", [])) if (digest := source_digest(source)).get("id")],
        "artifacts": artifact_digests(run.artifacts, base_dir=out),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def refresh_run_manifest(run_json: str | Path) -> dict[str, Any]:
    """Regenerate manifest.json after tools add artifacts to an existing run."""

    target = Path(run_json)
    payload = json.loads(target.read_text(encoding="utf-8"))
    from .audit import load_run_artifact

    manifest_path = target.parent / MANIFEST_NAME
    artifacts = payload.setdefault("artifacts", {})
    if isinstance(artifacts, dict):
        artifacts["manifest"] = str(manifest_path)
    else:
        payload["artifacts"] = {"manifest": str(manifest_path)}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    run = load_run_artifact(target)
    _assert_refresh_preserves_run_ledger(target, run)
    manifest = write_run_manifest(run, target.parent)
    return manifest


def artifact_digests(artifacts: dict[str, str], *, base_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    base = Path(base_dir) if base_dir is not None else None
    artifact_rows = artifacts if isinstance(artifacts, dict) else {}
    for name, raw_path in sorted(artifact_rows.items(), key=lambda item: str(item[0])):
        if name == "manifest":
            continue
        if not isinstance(raw_path, (str, Path)) or not raw_path:
            continue
        path = resolve_run_artifact_path(base, raw_path) if base is not None else Path(raw_path)
        row: dict[str, Any] = {
            "path": str(raw_path),
            "exists": path.exists(),
            "mutable": name in MUTABLE_ARTIFACTS,
        }
        if path.exists() and path.is_file():
            row["bytes"] = path.stat().st_size
            if name not in MUTABLE_ARTIFACTS:
                row["sha256"] = sha256_file(path)
        rows[name] = row
    return rows


def verify_run_manifest(run_json: str | Path) -> dict[str, Any]:
    target = Path(run_json)
    if not target.exists():
        return _verify_result(target, "", [{"name": "run_json_exists", "passed": False, "observed": "missing"}])
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _verify_result(target, "", [{"name": "run_json_parseable", "passed": False, "observed": str(exc)}])
    manifest_path = _manifest_path(target, payload)
    if manifest_path is None:
        return _verify_result(
            target,
            "",
            [{"name": "manifest_exists", "passed": False, "observed": "missing", "threshold": "run artifacts.manifest or sibling manifest.json"}],
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _verify_result(target, str(manifest_path), [{"name": "manifest_parseable", "passed": False, "observed": str(exc)}])

    checks: list[dict[str, Any]] = [
        {
            "name": "manifest_schema_version_supported",
            "passed": manifest.get("schema_version") == 1,
            "observed": manifest.get("schema_version", "missing"),
            "threshold": 1,
        },
        {
            "name": "run_id_matches_manifest",
            "passed": payload.get("run_id") == manifest.get("run_id"),
            "observed": f"{payload.get('run_id', '')} / {manifest.get('run_id', '')}",
            "threshold": "equal",
        },
        {
            "name": "question_matches_manifest",
            "passed": payload.get("question") == manifest.get("question"),
            "observed": "equal" if payload.get("question") == manifest.get("question") else "changed",
            "threshold": "equal",
        },
        {
            "name": "mode_matches_manifest",
            "passed": payload.get("mode", "literature") == manifest.get("mode", "literature"),
            "observed": f"{payload.get('mode', 'literature')} / {manifest.get('mode', 'literature')}",
            "threshold": "equal",
        },
        {
            "name": "provenance_matches_manifest",
            "passed": payload.get("provenance", {}) == manifest.get("provenance", {}),
            "observed": "equal" if payload.get("provenance", {}) == manifest.get("provenance", {}) else "changed",
            "threshold": "equal",
        },
    ]
    checks.extend(_run_ledger_checks(target, manifest))
    manifest_sources_raw = manifest.get("sources", [])
    run_sources_raw = payload.get("sources", [])
    manifest_sources = manifest_sources_raw if isinstance(manifest_sources_raw, list) else []
    run_sources = run_sources_raw if isinstance(run_sources_raw, list) else []
    if not isinstance(run_sources_raw, list):
        checks.append(
            {
                "name": "run_sources_parseable",
                "passed": False,
                "observed": type(run_sources_raw).__name__,
                "threshold": "list",
            }
        )
    if not isinstance(manifest_sources_raw, list):
        checks.append(
            {
                "name": "manifest_sources_parseable",
                "passed": False,
                "observed": type(manifest_sources_raw).__name__,
                "threshold": "list",
            }
        )
    if isinstance(run_sources_raw, list) and isinstance(manifest_sources_raw, list):
        checks.append(
            {
                "name": "source_count_matches_manifest",
                "passed": len(run_sources) == len(manifest_sources),
                "observed": f"{len(run_sources)} / {len(manifest_sources)}",
                "threshold": "equal",
            }
        )
        checks.extend(_source_manifest_checks(run_sources, manifest_sources))
    checks.extend(_evidence_graph_checks(payload, run_sources))
    checks.extend(_discovery_graph_checks(payload, run_sources))
    manifest_artifacts_raw = manifest.get("artifacts", {})
    manifest_artifacts = manifest_artifacts_raw if isinstance(manifest_artifacts_raw, dict) else {}
    if not isinstance(manifest_artifacts_raw, dict):
        checks.append(
            {
                "name": "manifest_artifacts_parseable",
                "passed": False,
                "observed": type(manifest_artifacts_raw).__name__,
                "threshold": "object",
            }
        )
    for name, row in sorted(manifest_artifacts.items()):
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"manifest_artifact_parseable:{name}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        raw_path = row.get("path", "")
        path_valid = isinstance(raw_path, str) and bool(raw_path)
        checks.append(
            {
                "name": f"artifact_path_declared:{name}",
                "passed": path_valid,
                "observed": type(raw_path).__name__ if not isinstance(raw_path, str) else (raw_path or "empty"),
                "threshold": "non-empty string",
            }
        )
        path = resolve_run_artifact_path(target.parent, raw_path) if path_valid else Path("")
        exists = path_valid and path.exists()
        checks.append(
            {
                "name": f"artifact_exists:{name}",
                "passed": exists,
                "observed": str(path) if exists else "missing",
                "threshold": "exists",
            }
        )
        if exists and path.is_file():
            declared_bytes = row.get("bytes")
            bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
            checks.append(
                {
                    "name": f"artifact_bytes_declared:{name}",
                    "passed": bytes_valid,
                    "observed": type(declared_bytes).__name__ if type(declared_bytes) is not int else declared_bytes,
                    "threshold": "non-negative integer",
                }
            )
            if bytes_valid:
                actual_bytes = path.stat().st_size
                checks.append(
                    {
                        "name": f"artifact_bytes:{name}",
                        "passed": actual_bytes == declared_bytes,
                        "observed": actual_bytes,
                        "threshold": declared_bytes,
                    }
                )
        if exists and not row.get("mutable"):
            declared_hash = row.get("sha256")
            hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
            checks.append(
                {
                    "name": f"artifact_sha256_declared:{name}",
                    "passed": hash_valid,
                    "observed": type(declared_hash).__name__ if not isinstance(declared_hash, str) else (declared_hash or "empty"),
                    "threshold": "sha256 hex",
                }
            )
            if not hash_valid:
                continue
            actual = sha256_file(path)
            checks.append(
                {
                    "name": f"artifact_sha256:{name}",
                    "passed": actual == declared_hash,
                    "observed": actual,
                    "threshold": declared_hash,
                }
            )
    declared_artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    for name, raw_path in sorted(declared_artifacts.items()):
        if not raw_path:
            continue
        path = resolve_run_artifact_path(target.parent, raw_path)
        checks.append(
            {
                "name": f"declared_artifact_exists:{name}",
                "passed": path.exists(),
                "observed": str(path) if path.exists() else "missing",
                "threshold": "exists",
            }
        )
        if name == "manifest":
            continue
        row = manifest_artifacts.get(name)
        row_path = row.get("path", "") if isinstance(row, dict) else ""
        checks.append(
            {
                "name": f"manifest_tracks_declared_artifact:{name}",
                "passed": isinstance(row, dict) and isinstance(row_path, str) and _paths_equivalent(row_path, raw_path, base_dir=target.parent),
                "observed": row_path if isinstance(row_path, str) and row_path else ("missing" if not isinstance(row, dict) else type(row_path).__name__),
                "threshold": str(raw_path),
            }
        )
    checks.extend(_sidecar_ledger_checks(target, payload, declared_artifacts, manifest_artifacts))
    declared_names = {name for name, raw_path in declared_artifacts.items() if raw_path and name != "manifest"}
    for name in sorted(manifest_artifacts):
        if name in declared_names:
            continue
        checks.append(
            {
                "name": f"manifest_artifact_declared:{name}",
                "passed": False,
                "observed": name,
                "threshold": "present in run.json artifacts",
            }
        )
    return _verify_result(target, str(manifest_path), checks)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def run_ledger_digest(run: ResearchRun) -> dict[str, Any]:
    return run_ledger_digest_from_payload(run.to_dict())


def run_ledger_digest_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _json_safe(_run_ledger_payload(payload if isinstance(payload, dict) else {}))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def _run_ledger_checks(run_json: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    manifest_ledger_raw = manifest.get("run_ledger", {})
    manifest_ledger = manifest_ledger_raw if isinstance(manifest_ledger_raw, dict) else {}
    if not isinstance(manifest_ledger_raw, dict):
        checks.append(
            {
                "name": "run_ledger_parseable",
                "passed": False,
                "observed": type(manifest_ledger_raw).__name__,
                "threshold": "object",
            }
        )
        return checks
    try:
        from .audit import load_run_artifact

        actual = run_ledger_digest(load_run_artifact(run_json))
    except Exception as exc:
        checks.append(
            {
                "name": "run_ledger_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "loadable run artifact",
            }
        )
        return checks
    declared_hash = manifest_ledger.get("sha256")
    hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
    checks.append(
        {
            "name": "run_ledger_sha256_declared",
            "passed": hash_valid,
            "observed": type(declared_hash).__name__ if not isinstance(declared_hash, str) else (declared_hash or "empty"),
            "threshold": "sha256 hex",
        }
    )
    if hash_valid:
        checks.append(
            {
                "name": "run_ledger_sha256",
                "passed": actual["sha256"] == declared_hash,
                "observed": actual["sha256"],
                "threshold": declared_hash,
            }
        )
    declared_bytes = manifest_ledger.get("bytes")
    bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
    checks.append(
        {
            "name": "run_ledger_bytes_declared",
            "passed": bytes_valid,
            "observed": type(declared_bytes).__name__ if type(declared_bytes) is not int else declared_bytes,
            "threshold": "non-negative integer",
        }
    )
    if bytes_valid:
        checks.append(
            {
                "name": "run_ledger_bytes",
                "passed": actual["bytes"] == declared_bytes,
                "observed": actual["bytes"],
                "threshold": declared_bytes,
            }
        )
    return checks


def _assert_refresh_preserves_run_ledger(run_json: Path, run: ResearchRun) -> None:
    manifest_path = _manifest_path(run_json, run.to_dict())
    if manifest_path is None:
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    ledger = manifest.get("run_ledger")
    if not isinstance(ledger, dict):
        return
    declared_hash = ledger.get("sha256")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        return
    actual = run_ledger_digest(run)
    if actual["sha256"] != declared_hash:
        raise ValueError("run ledger changed after manifest creation; inspect the dossier instead of refreshing the manifest")


def _run_ledger_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "artifacts"}


def _source_manifest_checks(run_sources: list[Any], manifest_sources: list[Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    run_source_ids = _row_ids(run_sources)
    manifest_source_ids = _row_ids(manifest_sources)
    checks.append(_unique_ids_check("run_source_ids_unique", run_source_ids))
    checks.append(_unique_ids_check("manifest_source_ids_unique", manifest_source_ids))
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(manifest_sources):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"manifest_source_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        source_id = row.get("id")
        if isinstance(source_id, str) and source_id:
            manifest_by_id[source_id] = row
    run_ids: set[str] = set()
    for index, row in enumerate(run_sources):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"run_source_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        source_id = row.get("id")
        source_id_valid = isinstance(source_id, str) and bool(source_id)
        checks.append(
            {
                "name": f"source_id_declared:{label}",
                "passed": source_id_valid,
                "observed": type(source_id).__name__ if not isinstance(source_id, str) else (source_id or "empty"),
                "threshold": "non-empty string",
            }
        )
        if not source_id_valid:
            continue
        run_ids.add(source_id)
        manifest_row = manifest_by_id.get(source_id)
        checks.append(
            {
                "name": f"source_tracked:{source_id}",
                "passed": isinstance(manifest_row, dict),
                "observed": "present" if isinstance(manifest_row, dict) else "missing",
                "threshold": "present in manifest sources",
            }
        )
        if not isinstance(manifest_row, dict):
            continue
        expected = source_digest_from_payload(row)
        declared_hash = manifest_row.get("text_sha256")
        hash_valid = isinstance(declared_hash, str) and _is_sha256_hex(declared_hash)
        checks.append(
            {
                "name": f"source_text_sha256_declared:{source_id}",
                "passed": hash_valid,
                "observed": type(declared_hash).__name__ if not isinstance(declared_hash, str) else (declared_hash or "empty"),
                "threshold": "sha256 hex",
            }
        )
        declared_bytes = manifest_row.get("text_bytes")
        bytes_valid = type(declared_bytes) is int and declared_bytes >= 0
        checks.append(
            {
                "name": f"source_text_bytes_declared:{source_id}",
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
            observed = manifest_row.get(key)
            checks.append(
                {
                    "name": f"source_{key}_matches:{source_id}",
                    "passed": observed == expected[key],
                    "observed": "equal" if observed == expected[key] else _compact_observed(observed),
                    "threshold": _compact_observed(expected[key]),
                }
            )
    for source_id in sorted(manifest_by_id):
        if source_id in run_ids:
            continue
        checks.append(
            {
                "name": f"manifest_source_declared:{source_id}",
                "passed": False,
                "observed": source_id,
                "threshold": "present in run.json sources",
            }
        )
    return checks


def _evidence_graph_checks(payload: dict[str, Any], run_sources: list[Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    evidence_raw = payload.get("evidence", [])
    claims_raw = payload.get("claims", [])
    evidence = evidence_raw if isinstance(evidence_raw, list) else []
    claims = claims_raw if isinstance(claims_raw, list) else []
    if not isinstance(evidence_raw, list):
        checks.append(
            {
                "name": "run_evidence_parseable",
                "passed": False,
                "observed": type(evidence_raw).__name__,
                "threshold": "list",
            }
        )
    if not isinstance(claims_raw, list):
        checks.append(
            {
                "name": "run_claims_parseable",
                "passed": False,
                "observed": type(claims_raw).__name__,
                "threshold": "list",
            }
        )
    if not isinstance(evidence_raw, list) or not isinstance(claims_raw, list):
        return checks

    source_ids = set(_row_ids(run_sources))
    evidence_ids = _row_ids(evidence)
    evidence_id_set = set(evidence_ids)
    checks.append(_unique_ids_check("evidence_ids_unique", evidence_ids))
    checks.append(_unique_ids_check("claim_ids_unique", _row_ids(claims)))
    for index, row in enumerate(evidence):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"run_evidence_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        evidence_id = row.get("id")
        evidence_id_valid = isinstance(evidence_id, str) and bool(evidence_id)
        checks.append(
            {
                "name": f"evidence_id_declared:{label}",
                "passed": evidence_id_valid,
                "observed": type(evidence_id).__name__ if not isinstance(evidence_id, str) else (evidence_id or "empty"),
                "threshold": "non-empty string",
            }
        )
        evidence_label = evidence_id if evidence_id_valid else label
        source_id = row.get("source_id")
        source_id_valid = isinstance(source_id, str) and bool(source_id)
        checks.append(
            {
                "name": f"evidence_source_id_declared:{evidence_label}",
                "passed": source_id_valid,
                "observed": type(source_id).__name__ if not isinstance(source_id, str) else (source_id or "empty"),
                "threshold": "non-empty string",
            }
        )
        if source_id_valid:
            checks.append(
                {
                    "name": f"evidence_source_tracked:{evidence_label}",
                    "passed": source_id in source_ids,
                    "observed": source_id,
                    "threshold": "source id present in run sources",
                }
            )

    for index, row in enumerate(claims):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"run_claim_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        claim_id = row.get("id")
        claim_id_valid = isinstance(claim_id, str) and bool(claim_id)
        checks.append(
            {
                "name": f"claim_id_declared:{label}",
                "passed": claim_id_valid,
                "observed": type(claim_id).__name__ if not isinstance(claim_id, str) else (claim_id or "empty"),
                "threshold": "non-empty string",
            }
        )
        claim_label = claim_id if claim_id_valid else label
        citations = row.get("citations", [])
        source_refs = row.get("source_ids", [])
        checks.append(
            {
                "name": f"claim_citations_parseable:{claim_label}",
                "passed": isinstance(citations, list),
                "observed": type(citations).__name__,
                "threshold": "list",
            }
        )
        checks.append(
            {
                "name": f"claim_source_ids_parseable:{claim_label}",
                "passed": isinstance(source_refs, list),
                "observed": type(source_refs).__name__,
                "threshold": "list",
            }
        )
        if isinstance(citations, list):
            for citation_index, citation in enumerate(citations):
                citation_valid = isinstance(citation, str) and bool(citation)
                suffix = citation if citation_valid else str(citation_index)
                checks.append(
                    {
                        "name": f"claim_citation_declared:{claim_label}:{suffix}",
                        "passed": citation_valid,
                        "observed": type(citation).__name__ if not isinstance(citation, str) else (citation or "empty"),
                        "threshold": "non-empty string",
                    }
                )
                if citation_valid:
                    checks.append(
                        {
                            "name": f"claim_citation_tracked:{claim_label}:{citation}",
                            "passed": citation in evidence_id_set,
                            "observed": citation,
                            "threshold": "evidence id present in run evidence",
                        }
                    )
        if isinstance(source_refs, list):
            for source_index, source_id in enumerate(source_refs):
                source_id_valid = isinstance(source_id, str) and bool(source_id)
                suffix = source_id if source_id_valid else str(source_index)
                checks.append(
                    {
                        "name": f"claim_source_id_declared:{claim_label}:{suffix}",
                        "passed": source_id_valid,
                        "observed": type(source_id).__name__ if not isinstance(source_id, str) else (source_id or "empty"),
                        "threshold": "non-empty string",
                    }
                )
                if source_id_valid:
                    checks.append(
                        {
                            "name": f"claim_source_tracked:{claim_label}:{source_id}",
                            "passed": source_id in source_ids,
                            "observed": source_id,
                            "threshold": "source id present in run sources",
                        }
                    )
    return checks


def _sidecar_ledger_checks(
    run_json: Path,
    payload: dict[str, Any],
    artifacts: dict[str, Any],
    manifest_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    generated_sidecars, expected_error = _expected_generated_sidecars(payload)
    for artifact in ("markdown", "html"):
        if not _manifest_tracks_artifact(manifest_artifacts, artifact):
            continue
        path = _sidecar_artifact_path(run_json.parent, artifacts, artifact)
        if path is None or not path.is_file():
            continue
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError as exc:
            checks.append(
                {
                    "name": f"{artifact}_sidecar_parseable",
                    "passed": False,
                    "observed": str(exc),
                    "threshold": "UTF-8 text",
                }
            )
            continue
        checks.append(
            {
                "name": f"{artifact}_sidecar_parseable",
                "passed": True,
                "observed": "text",
                "threshold": "UTF-8 text",
            }
        )
        expected = generated_sidecars.get(artifact)
        if expected_error:
            checks.append(
                {
                    "name": f"{artifact}_sidecar_expected_parseable",
                    "passed": False,
                    "observed": expected_error,
                    "threshold": "loadable run artifact",
                }
            )
        else:
            observed = _normalise_generated_report(artifact, actual)
            threshold = _normalise_generated_report(artifact, str(expected))
            checks.append(
                {
                    "name": f"{artifact}_sidecar_matches_run",
                    "passed": observed == threshold,
                    "observed": "equal" if observed == threshold else "changed",
                    "threshold": f"generated {artifact} report from run.json",
                }
            )

    paper_path = _sidecar_artifact_path(run_json.parent, artifacts, "paper")
    if _manifest_tracks_artifact(manifest_artifacts, "paper") and paper_path is not None and paper_path.is_file():
        checks.extend(_paper_artifact_structure_checks(paper_path, prefix=""))
    review_path = _sidecar_artifact_path(run_json.parent, artifacts, "review")
    if _manifest_tracks_artifact(manifest_artifacts, "review") and review_path is not None and review_path.is_file():
        checks.extend(_review_artifact_structure_checks(review_path, prefix=""))
    pdf_path = _sidecar_artifact_path(run_json.parent, artifacts, "pdf")
    if _manifest_tracks_artifact(manifest_artifacts, "pdf") and pdf_path is not None and pdf_path.is_file():
        checks.extend(_pdf_artifact_structure_checks(pdf_path, prefix=""))
    trace_path = _sidecar_artifact_path(run_json.parent, artifacts, "trace")
    if _manifest_tracks_artifact(manifest_artifacts, "trace") and trace_path is not None and trace_path.is_file():
        checks.extend(_trace_artifact_structure_checks(trace_path, str(payload.get("run_id", "")), prefix=""))

    for artifact in ("graph", "evals"):
        if not _manifest_tracks_artifact(manifest_artifacts, artifact):
            continue
        path = _sidecar_artifact_path(run_json.parent, artifacts, artifact)
        if path is None or not path.is_file():
            continue
        payload_result, parse_error = _read_json_file(path)
        if parse_error:
            checks.append(
                {
                    "name": f"{artifact}_sidecar_parseable",
                    "passed": False,
                    "observed": parse_error,
                    "threshold": "valid JSON",
                }
            )
            continue
        checks.append(
            {
                "name": f"{artifact}_sidecar_parseable",
                "passed": isinstance(payload_result, dict),
                "observed": type(payload_result).__name__,
                "threshold": "object",
            }
        )
        expected = generated_sidecars.get(artifact)
        if expected_error:
            checks.append(
                {
                    "name": f"{artifact}_sidecar_expected_parseable",
                    "passed": False,
                    "observed": expected_error,
                    "threshold": "loadable run artifact",
                }
            )
        elif isinstance(payload_result, dict):
            checks.append(
                {
                    "name": f"{artifact}_sidecar_matches_run",
                    "passed": payload_result == expected,
                    "observed": "equal" if payload_result == expected else "changed",
                    "threshold": f"generated {artifact}.json from run.json",
                }
            )

    experiments_path = _sidecar_artifact_path(run_json.parent, artifacts, "experiments")
    if _manifest_tracks_artifact(manifest_artifacts, "experiments") and experiments_path is not None and experiments_path.is_file():
        experiments_payload, parse_error = _read_json_file(experiments_path)
        if parse_error:
            checks.append(
                {
                    "name": "experiments_sidecar_parseable",
                    "passed": False,
                    "observed": parse_error,
                    "threshold": "JSON list",
                }
            )
        else:
            expected = payload.get("experiments", [])
            checks.append(
                {
                    "name": "experiments_sidecar_parseable",
                    "passed": isinstance(experiments_payload, list),
                    "observed": type(experiments_payload).__name__,
                    "threshold": "list",
                }
            )
            if isinstance(experiments_payload, list):
                checks.append(
                    {
                        "name": "experiments_sidecar_matches_run",
                        "passed": experiments_payload == expected,
                        "observed": "equal" if experiments_payload == expected else "changed",
                        "threshold": "run.json experiments",
                    }
                )

    discoveries_path = _sidecar_artifact_path(run_json.parent, artifacts, "discoveries")
    if _manifest_tracks_artifact(manifest_artifacts, "discoveries") and discoveries_path is not None and discoveries_path.is_file():
        discoveries_payload, parse_error = _read_json_file(discoveries_path)
        if parse_error:
            checks.append(
                {
                    "name": "discoveries_sidecar_parseable",
                    "passed": False,
                    "observed": parse_error,
                    "threshold": "JSON object",
                }
            )
        else:
            checks.extend(_discoveries_sidecar_payload_checks(discoveries_payload, payload, prefix=""))
    return checks


def _manifest_tracks_artifact(manifest_artifacts: dict[str, Any], name: str) -> bool:
    return isinstance(manifest_artifacts.get(name), dict)


def _paper_artifact_structure_checks(path: Path, *, prefix: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {
            "name": f"{prefix}paper_artifact_main_tex",
            "passed": path.name == "main.tex",
            "observed": path.name,
            "threshold": "main.tex",
        }
    ]
    try:
        tex = path.read_text(encoding="utf-8")
    except OSError as exc:
        checks.append(
            {
                "name": f"{prefix}paper_artifact_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "UTF-8 LaTeX",
            }
        )
        return checks
    checks.append(
        {
            "name": f"{prefix}paper_artifact_parseable",
            "passed": True,
            "observed": "text",
            "threshold": "UTF-8 LaTeX",
        }
    )
    missing = [marker for marker in PAPER_ARTIFACT_REQUIRED_MARKERS if marker not in tex]
    checks.append(
        {
            "name": f"{prefix}paper_artifact_latex_structure",
            "passed": not missing,
            "observed": ", ".join(missing) if missing else "present",
            "threshold": ", ".join(PAPER_ARTIFACT_REQUIRED_MARKERS),
        }
    )
    return checks


def _review_artifact_structure_checks(path: Path, *, prefix: str) -> list[dict[str, Any]]:
    try:
        review = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [
            {
                "name": f"{prefix}review_artifact_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "UTF-8 text",
            }
        ]
    return [
        {
            "name": f"{prefix}review_artifact_parseable",
            "passed": True,
            "observed": "text",
            "threshold": "UTF-8 text",
        },
        {
            "name": f"{prefix}review_artifact_nonempty",
            "passed": bool(review.strip()),
            "observed": "nonempty" if review.strip() else "empty",
            "threshold": "non-empty review text",
        },
    ]


def _pdf_artifact_structure_checks(path: Path, *, prefix: str) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            header = handle.read(5)
    except OSError as exc:
        return [
            {
                "name": f"{prefix}pdf_artifact_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "%PDF header",
            }
        ]
    return [
        {
            "name": f"{prefix}pdf_artifact_parseable",
            "passed": True,
            "observed": "bytes",
            "threshold": "%PDF header",
        },
        {
            "name": f"{prefix}pdf_artifact_header",
            "passed": header.startswith(b"%PDF"),
            "observed": header.decode("latin1", errors="replace") or "empty",
            "threshold": "%PDF",
        },
    ]


def _trace_artifact_structure_checks(path: Path, expected_run_id: str, *, prefix: str) -> list[dict[str, Any]]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        return [
            {
                "name": f"{prefix}trace_artifact_parseable",
                "passed": False,
                "observed": str(exc),
                "threshold": "JSONL trace",
            }
        ]
    checks: list[dict[str, Any]] = [
        {
            "name": f"{prefix}trace_artifact_nonempty",
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
            "name": f"{prefix}trace_artifact_parseable",
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
            "name": f"{prefix}trace_artifact_fields",
            "passed": not missing_fields,
            "observed": ", ".join(missing_fields[:5]) if missing_fields else "present",
            "threshold": "trace_id, run_id, span_id, phase, name, time_unix_ms, attributes",
        }
    )
    wrong_run_ids = sorted({str(row.get("run_id", "")) for row in rows if row.get("run_id") != expected_run_id})
    checks.append(
        {
            "name": f"{prefix}trace_artifact_run_id",
            "passed": bool(expected_run_id) and not wrong_run_ids,
            "observed": ", ".join(wrong_run_ids[:5]) if wrong_run_ids else (expected_run_id or "missing"),
            "threshold": expected_run_id or "run.json run_id",
        }
    )
    bad_phases = sorted({str(row.get("phase", "")) for row in rows if row.get("phase") not in phases})
    checks.append(
        {
            "name": f"{prefix}trace_artifact_phase",
            "passed": not bad_phases,
            "observed": ", ".join(bad_phases[:5]) if bad_phases else "valid",
            "threshold": ", ".join(sorted(phases)),
        }
    )
    return checks


def _read_json_file(path: Path) -> tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _normalise_generated_report(artifact: str, text: str) -> str:
    if artifact != "html":
        return text
    start = '<script id="run-json" type="application/json">'
    end = "</script>"
    if start not in text:
        return text
    before, rest = text.split(start, 1)
    if end not in rest:
        return text
    _, after = rest.split(end, 1)
    return before + start + end + after


def _expected_generated_sidecars(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        from .audit import _run_from_payload
        from .report import claim_graph, html_report, markdown_report, run_evals

        run = _run_from_payload(payload)
        return {
            "markdown": markdown_report(run),
            "html": html_report(run),
            "graph": claim_graph(run),
            "evals": run_evals(run),
        }, ""
    except Exception as exc:
        return {}, str(exc)


def _sidecar_artifact_path(base_dir: Path, artifacts: dict[str, Any], name: str) -> Path | None:
    raw_path = artifacts.get(name)
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return resolve_run_artifact_path(base_dir, raw_path)


def _discoveries_sidecar_payload_checks(sidecar: Any, payload: dict[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {
            "name": f"{prefix}discoveries_sidecar_parseable",
            "passed": isinstance(sidecar, dict),
            "observed": type(sidecar).__name__,
            "threshold": "object",
        }
    ]
    if not isinstance(sidecar, dict):
        return checks
    for key in ("run_id", "question"):
        checks.append(
            {
                "name": f"{prefix}discoveries_sidecar_{key}_matches_run",
                "passed": sidecar.get(key) == payload.get(key),
                "observed": "equal" if sidecar.get(key) == payload.get(key) else "changed",
                "threshold": f"run.json {key}",
            }
        )
    for key in ("discoveries", "hypotheses"):
        value = sidecar.get(key, [])
        expected = payload.get(key, [])
        checks.append(
            {
                "name": f"{prefix}discoveries_sidecar_{key}_parseable",
                "passed": isinstance(value, list),
                "observed": type(value).__name__,
                "threshold": "list",
            }
        )
        if isinstance(value, list):
            checks.append(
                {
                    "name": f"{prefix}discoveries_sidecar_{key}_matches_run",
                    "passed": value == expected,
                    "observed": "equal" if value == expected else "changed",
                    "threshold": f"run.json {key}",
                }
            )
    return checks


def _discovery_graph_checks(payload: dict[str, Any], run_sources: list[Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    hypotheses_raw = payload.get("hypotheses", [])
    experiments_raw = payload.get("experiments", [])
    discoveries_raw = payload.get("discoveries", [])
    claims_raw = payload.get("claims", [])
    hypotheses = hypotheses_raw if isinstance(hypotheses_raw, list) else []
    experiments = experiments_raw if isinstance(experiments_raw, list) else []
    discoveries = discoveries_raw if isinstance(discoveries_raw, list) else []
    claims = claims_raw if isinstance(claims_raw, list) else []

    parseable = True
    for field, raw in (
        ("hypotheses", hypotheses_raw),
        ("experiments", experiments_raw),
        ("discoveries", discoveries_raw),
    ):
        if isinstance(raw, list):
            continue
        parseable = False
        checks.append(
            {
                "name": f"run_{field}_parseable",
                "passed": False,
                "observed": type(raw).__name__,
                "threshold": "list",
            }
        )
    if not parseable:
        return checks

    hypothesis_ids = _row_ids(hypotheses)
    experiment_ids = _row_ids(experiments)
    discovery_ids = _row_ids(discoveries)
    checks.append(_unique_ids_check("hypothesis_ids_unique", hypothesis_ids))
    checks.append(_unique_ids_check("experiment_ids_unique", experiment_ids))
    checks.append(_unique_ids_check("discovery_ids_unique", discovery_ids))

    source_id_set = set(_row_ids(run_sources))
    claim_id_set = set(_row_ids(claims))
    hypothesis_id_set = set(hypothesis_ids)
    experiment_ref_set = set(experiment_ids)
    experiment_ref_set.update(_row_field_values(experiments, "spec_id"))

    for index, row in enumerate(hypotheses):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"run_hypothesis_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        hypothesis_id = row.get("id")
        hypothesis_id_valid = isinstance(hypothesis_id, str) and bool(hypothesis_id)
        checks.append(
            {
                "name": f"hypothesis_id_declared:{label}",
                "passed": hypothesis_id_valid,
                "observed": type(hypothesis_id).__name__ if not isinstance(hypothesis_id, str) else (hypothesis_id or "empty"),
                "threshold": "non-empty string",
            }
        )
        hypothesis_label = hypothesis_id if hypothesis_id_valid else label
        checks.extend(
            _id_list_reference_checks(
                row,
                owner="hypothesis",
                owner_label=hypothesis_label,
                field="experiment_ids",
                declared_prefix="hypothesis_experiment",
                tracked_prefix="hypothesis_experiment",
                target_ids=experiment_ref_set,
                target_description="experiment id or spec_id present in run experiments",
            )
        )
        checks.extend(
            _id_list_reference_checks(
                row,
                owner="hypothesis",
                owner_label=hypothesis_label,
                field="source_ids",
                declared_prefix="hypothesis_source_id",
                tracked_prefix="hypothesis_source",
                target_ids=source_id_set,
                target_description="source id present in run sources",
            )
        )

    for index, row in enumerate(experiments):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"run_experiment_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        experiment_id = row.get("id")
        experiment_id_valid = isinstance(experiment_id, str) and bool(experiment_id)
        checks.append(
            {
                "name": f"experiment_id_declared:{label}",
                "passed": experiment_id_valid,
                "observed": type(experiment_id).__name__ if not isinstance(experiment_id, str) else (experiment_id or "empty"),
                "threshold": "non-empty string",
            }
        )

    for index, row in enumerate(discoveries):
        label = _source_label(row, index)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"run_discovery_parseable:{label}",
                    "passed": False,
                    "observed": type(row).__name__,
                    "threshold": "object",
                }
            )
            continue
        discovery_id = row.get("id")
        discovery_id_valid = isinstance(discovery_id, str) and bool(discovery_id)
        checks.append(
            {
                "name": f"discovery_id_declared:{label}",
                "passed": discovery_id_valid,
                "observed": type(discovery_id).__name__ if not isinstance(discovery_id, str) else (discovery_id or "empty"),
                "threshold": "non-empty string",
            }
        )
        discovery_label = discovery_id if discovery_id_valid else label
        checks.extend(
            _id_list_reference_checks(
                row,
                owner="discovery",
                owner_label=discovery_label,
                field="supporting_experiments",
                declared_prefix="discovery_experiment",
                tracked_prefix="discovery_experiment",
                target_ids=experiment_ref_set,
                target_description="experiment id or spec_id present in run experiments",
            )
        )
        checks.extend(
            _id_list_reference_checks(
                row,
                owner="discovery",
                owner_label=discovery_label,
                field="claim_ids",
                declared_prefix="discovery_claim",
                tracked_prefix="discovery_claim",
                target_ids=claim_id_set,
                target_description="claim id present in run claims",
            )
        )
        hypothesis_id = row.get("hypothesis_id", "")
        if hypothesis_id in ("", None):
            continue
        hypothesis_id_valid = isinstance(hypothesis_id, str)
        checks.append(
            {
                "name": f"discovery_hypothesis_declared:{discovery_label}",
                "passed": hypothesis_id_valid,
                "observed": type(hypothesis_id).__name__,
                "threshold": "string",
            }
        )
        if hypothesis_id_valid:
            checks.append(
                {
                    "name": f"discovery_hypothesis_tracked:{discovery_label}:{hypothesis_id}",
                    "passed": hypothesis_id in hypothesis_id_set,
                    "observed": hypothesis_id,
                    "threshold": "hypothesis id present in run hypotheses",
                }
            )
    return checks


def _id_list_reference_checks(
    row: dict[str, Any],
    *,
    owner: str,
    owner_label: str,
    field: str,
    declared_prefix: str,
    tracked_prefix: str,
    target_ids: set[str],
    target_description: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    refs = row.get(field, [])
    checks.append(
        {
            "name": f"{owner}_{field}_parseable:{owner_label}",
            "passed": isinstance(refs, list),
            "observed": type(refs).__name__,
            "threshold": "list",
        }
    )
    if not isinstance(refs, list):
        return checks
    for ref_index, ref in enumerate(refs):
        ref_valid = isinstance(ref, str) and bool(ref)
        suffix = ref if ref_valid else str(ref_index)
        checks.append(
            {
                "name": f"{declared_prefix}_declared:{owner_label}:{suffix}",
                "passed": ref_valid,
                "observed": type(ref).__name__ if not isinstance(ref, str) else (ref or "empty"),
                "threshold": "non-empty string",
            }
        )
        if ref_valid:
            checks.append(
                {
                    "name": f"{tracked_prefix}_tracked:{owner_label}:{ref}",
                    "passed": ref in target_ids,
                    "observed": ref,
                    "threshold": target_description,
                }
            )
    return checks


def _row_ids(rows: list[Any]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            ids.append(row_id)
    return ids


def _row_field_values(rows: list[Any], field: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(field)
        if isinstance(value, str) and value:
            values.append(value)
    return values


def _unique_ids_check(name: str, ids: list[str]) -> dict[str, Any]:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    return {
        "name": name,
        "passed": not duplicates,
        "observed": ", ".join(duplicates) if duplicates else "unique",
        "threshold": "unique non-empty ids",
    }


def source_digest_from_payload(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("text") if isinstance(row.get("text"), str) else ""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "kind": row.get("kind", "document"),
        "url": row.get("url", ""),
        "text_sha256": sha256_text(text),
        "text_bytes": len(text.encode("utf-8")),
        "created_at": row.get("created_at"),
        "metadata": metadata,
    }


def _source_label(row: Any, index: int) -> str:
    if isinstance(row, dict):
        source_id = row.get("id")
        if isinstance(source_id, str) and source_id:
            return source_id
    return str(index)


def _compact_observed(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 120:
        return f"{value[:117]}..."
    return value


def _manifest_path(run_json: Path, payload: dict[str, Any]) -> Path | None:
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    raw_manifest = artifacts.get("manifest")
    if raw_manifest:
        candidate = resolve_run_artifact_path(run_json.parent, raw_manifest)
        if candidate.exists():
            return candidate
    sibling = run_json.parent / MANIFEST_NAME
    return sibling if sibling.exists() else None


def _paths_equivalent(left: str | Path, right: str | Path, *, base_dir: str | Path | None = None) -> bool:
    left_path = resolve_run_artifact_path(base_dir, left)
    right_path = resolve_run_artifact_path(base_dir, right)
    if left_path == right_path:
        return True
    try:
        return left_path.expanduser().resolve() == right_path.expanduser().resolve()
    except OSError:
        return False


def resolve_run_artifact_path(base_dir: str | Path | None, raw_path: str | Path) -> Path:
    """Resolve a stored artifact path against a run directory when needed."""

    path = Path(raw_path)
    if path.is_absolute() or base_dir is None:
        return path
    base = Path(base_dir)
    candidates = [base / path]
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part != base.name:
            continue
        suffix = Path(*parts[index + 1 :])
        candidates.append(base / suffix)
    if path.name:
        candidates.append(base / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _verify_result(run_json: Path, manifest: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [check["name"] for check in checks if not check.get("passed")]
    return {
        "path": str(run_json),
        "manifest": manifest,
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "next_actions": _next_actions(failed),
    }


def _next_actions(failed: list[str]) -> list[str]:
    if not failed:
        return []
    if any(name.startswith("artifact_sha256:") for name in failed):
        return ["Regenerate the edited artifact or rerun the dossier so manifest hashes match current files."]
    if "run_ledger_sha256" in failed or "run_ledger_bytes" in failed or "run_ledger_parseable" in failed:
        return ["Inspect the run ledger before sharing the dossier; run findings changed after manifest creation."]
    if any(
        name.startswith(("source_", "manifest_source_", "run_source_")) or name in {"run_sources_parseable", "manifest_sources_parseable"}
        for name in failed
    ):
        return ["Inspect the run source ledger before sharing the dossier; source provenance changed after manifest creation."]
    if any(
        name.startswith(("evidence_", "claim_", "run_evidence_", "run_claim_"))
        or name in {"run_evidence_parseable", "run_claims_parseable"}
        for name in failed
    ):
        return ["Fix the run evidence graph before sharing the dossier; every evidence chunk and claim citation must point at declared ledger IDs."]
    if any(
        name.startswith(("hypothesis_", "experiment_", "discovery_", "run_hypothesis_", "run_experiment_", "run_discovery_"))
        or name in {"run_hypotheses_parseable", "run_experiments_parseable", "run_discoveries_parseable"}
        for name in failed
    ):
        return ["Fix the discovery graph before sharing the dossier; every hypothesis, experiment, and discovery reference must point at declared ledger IDs."]
    if any(
        name.startswith(
            (
                "paper_artifact_",
                "review_artifact_",
                "pdf_artifact_",
                "trace_artifact_",
                "markdown_sidecar_",
                "html_sidecar_",
                "graph_sidecar_",
                "evals_sidecar_",
                "experiments_sidecar_",
                "discoveries_sidecar_",
            )
        )
        for name in failed
    ):
        return ["Regenerate the generated sidecar files before sharing the dossier; reports and sidecar ledgers must match run.json."]
    if any(
        name.startswith(
            (
                "manifest_tracks_declared_artifact:",
                "manifest_artifact_declared:",
                "manifest_artifact_parseable:",
                "artifact_path_declared:",
                "artifact_bytes:",
                "artifact_bytes_declared:",
                "run_ledger_sha256_declared",
                "run_ledger_bytes_declared",
                "mode_matches_manifest",
                "provenance_matches_manifest",
                "artifact_sha256_declared:",
            )
        )
        for name in failed
    ) or "manifest_artifacts_parseable" in failed or "manifest_schema_version_supported" in failed:
        return ["Refresh the run manifest so manifest.json covers the same artifact set declared in run.json."]
    if any(name.startswith(("artifact_exists:", "declared_artifact_exists:")) for name in failed):
        return ["Regenerate missing artifacts or remove stale artifact paths from run.json."]
    if "manifest_exists" in failed:
        return ["Rerun the dossier with the current MechFerret version to create manifest.json."]
    return ["Inspect the run artifact and manifest mismatch before sharing the dossier."]
