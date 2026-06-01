from __future__ import annotations

import importlib.util
import json
import math
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from .text import compact_text

PROJECT_ROOT = Path("projects/openvla_sae")
TEMPLATE_ROOT = "templates/openvla_sae"
DEFAULT_MANIFEST = Path("data/openvla_sae_phase1.jsonl")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REQUIRED_FILES = (
    "README.md",
    "configs/phase1.yaml",
    "paper/outline.md",
    "src/cache_openvla_activations.py",
    "src/train_sae_from_cache.py",
    "src/simple_topk_sae.py",
    "scripts/install_openvla_min.sh",
    "scripts/phase1_commands.sh",
)


def _text(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return default


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


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _flag(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _path(value: Any, default: str | Path = "") -> Path:
    if isinstance(value, (str, Path)):
        text = str(value)
        return Path(text) if text else Path(default)
    return Path(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {_text(key, str(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)


def status(project_root: str | Path = PROJECT_ROOT, manifest: str | Path | None = None) -> dict[str, Any]:
    root = _path(project_root, PROJECT_ROOT)
    manifest_path = _path(manifest, DEFAULT_MANIFEST) if manifest else DEFAULT_MANIFEST
    files = {name: (root / name).exists() for name in REQUIRED_FILES}
    template_available = _template_available()
    deps = _dependency_status()
    manifest_status = validate_manifest(manifest_path) if manifest_path.exists() else {
        "path": str(manifest_path),
        "exists": False,
        "rows": 0,
        "valid_rows": 0,
        "errors": [],
        "missing_images": [],
    }
    ready_local = all(files.values())
    ready_gpu = ready_local and manifest_status["valid_rows"] > 0 and deps["torch"] and deps["transformers"]
    next_actions = []
    if not ready_local:
        if template_available:
            next_actions.append(f"Run `mechferret sae openvla init --project-root {root}` to scaffold the workflow files.")
        else:
            next_actions.append("Restore the OpenVLA SAE project files under projects/openvla_sae.")
    if not manifest_status["exists"]:
        next_actions.append(f"Create a JSONL manifest at {manifest_path} with image_path and instruction fields.")
    elif manifest_status["errors"] or manifest_status["missing_images"]:
        next_actions.append("Fix manifest rows or image paths before caching activations.")
    if not deps["torch"] or not deps["transformers"]:
        next_actions.append(f"Install GPU dependencies with {root / 'scripts' / 'install_openvla_min.sh'}.")
    if ready_gpu:
        next_actions.append("Run `mechferret sae openvla commands` on a GPU machine, then cache activations and train SAEs.")
    return {
        "project_root": str(root),
        "files": files,
        "dependencies": deps,
        "manifest": manifest_status,
        "template_available": template_available,
        "ready_local": ready_local,
        "ready_gpu": ready_gpu,
        "next_actions": next_actions,
    }


def init_project(
    out_dir: str | Path = PROJECT_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    target = _path(out_dir, PROJECT_ROOT)
    force = _flag(force)
    template = _template_root()
    if not _template_available():
        return {
            "ok": False,
            "project_root": str(target),
            "files_written": [],
            "existing_files": [],
            "next_actions": ["Template files are missing from the installed package."],
        }
    existing = [
        str(target / rel)
        for rel in _template_file_names(template)
        if (target / rel).exists()
    ]
    if existing and not force:
        return {
            "ok": False,
            "project_root": str(target),
            "files_written": [],
            "existing_files": existing[:20],
            "next_actions": ["Pass --force to overwrite existing OpenVLA SAE workflow files."],
        }
    written: list[str] = []
    _copy_template_tree(template, target, written=written)
    st = status(target)
    return {
        "ok": st["ready_local"],
        "project_root": str(target),
        "files_written": written,
        "existing_files": existing[:20],
        "next_actions": st["next_actions"],
    }


def validate_manifest(path: str | Path, *, max_rows: int = 1000) -> dict[str, Any]:
    manifest = _path(path, DEFAULT_MANIFEST)
    row_limit = _positive_int(max_rows, 1000)
    errors: list[str] = []
    missing_images: list[str] = []
    rows = valid_rows = 0
    if not manifest.is_file():
        return {
            "path": str(manifest),
            "exists": False,
            "rows": 0,
            "valid_rows": 0,
            "errors": [],
            "missing_images": [],
        }
    try:
        lines = manifest.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return {
            "path": str(manifest),
            "exists": True,
            "rows": 0,
            "valid_rows": 0,
            "errors": [f"read failed: {exc}"],
            "missing_images": [],
        }
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        rows += 1
        if rows > row_limit:
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(item, dict):
            errors.append(f"line {line_no}: expected object")
            continue
        image_path = _text(item.get("image_path")).strip()
        instruction = _text(item.get("instruction")).strip()
        missing = [key for key, value in (("image_path", image_path), ("instruction", instruction)) if not value]
        if missing:
            errors.append(f"line {line_no}: missing {', '.join(missing)}")
            continue
        image = Path(image_path)
        if not image.is_file():
            missing_images.append(str(image))
        valid_rows += 1
    return {
        "path": str(manifest),
        "exists": True,
        "rows": rows,
        "valid_rows": valid_rows,
        "errors": errors[:12],
        "missing_images": missing_images[:12],
    }


def create_manifest(
    image_dir: str | Path,
    out: str | Path = DEFAULT_MANIFEST,
    *,
    instruction: str = "perform the task shown in the image",
    action: str = "",
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = _path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")
    target = _path(out, DEFAULT_MANIFEST)
    force = _flag(force)
    if target.exists() and not force:
        raise FileExistsError(f"manifest already exists: {target} (pass --force to overwrite)")
    instruction_text = _text(instruction).strip() or "perform the task shown in the image"
    action_text = _text(action).strip()
    images = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if limit is not None:
        images = images[:_positive_int(limit, len(images))]
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in images:
        row = {"image_path": str(image), "instruction": instruction_text}
        if action_text:
            row["action"] = action_text
        rows.append(row)
    target.write_text(
        "\n".join(_json_dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    validation = validate_manifest(target)
    return {
        "path": str(target),
        "image_dir": str(root),
        "rows_written": len(rows),
        "valid_rows": validation["valid_rows"],
        "missing_images": validation["missing_images"],
        "errors": validation["errors"],
    }


def smoke_test(
    out_dir: str | Path = "runs/openvla_sae/smoke",
    *,
    d_model: int = 32,
    tokens: int = 256,
    steps: int = 20,
    k: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    deps = _dependency_status()
    out = _path(out_dir, "runs/openvla_sae/smoke")
    out.mkdir(parents=True, exist_ok=True)
    d_model = _positive_int(d_model, 32)
    tokens = _positive_int(tokens, 256)
    steps = _positive_int(steps, 20)
    k = min(_positive_int(k, 4), max(1, d_model * 4))
    seed = _positive_int(seed, 0, minimum=0)
    if not deps["torch"]:
        payload = {
            "ok": False,
            "out_dir": str(out),
            "reason": "torch is not installed",
            "install": "pip install torch pyyaml tqdm",
        }
        _write_smoke_artifacts(out, payload)
        return payload

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    src_dir = Path(PROJECT_ROOT) / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from simple_topk_sae import TopKSAE  # type: ignore

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(tokens, d_model)
    # Add a few sparse latent directions so the toy run has signal.
    latents = torch.randn(tokens, max(4, d_model // 4))
    dictionary = torch.randn(latents.shape[-1], d_model)
    mask = (torch.rand_like(latents) > 0.85).float()
    x = x * 0.1 + (latents * mask) @ dictionary
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x - mean) / std

    model = TopKSAE(d_in=d_model, d_sae=d_model * 4, k=k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_loss = final_loss = 0.0
    batch_size = min(64, tokens)
    for step in range(max(1, steps)):
        idx = torch.randint(0, x_train.shape[0], (batch_size,))
        batch = x_train[idx].to(device)
        x_hat, _z, aux = model(batch)
        loss = F.mse_loss(x_hat, batch)
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.normalize_decoder()
        final_loss = float(loss.detach().cpu())

    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    torch.save({"activation": x[: min(tokens, 64)], "row": {"instruction": "synthetic smoke"}}, cache_dir / "000000.pt")
    checkpoint = out / "smoke_sae.pt"
    metrics = {
        "ok": True,
        "device": device,
        "tokens": tokens,
        "d_model": d_model,
        "steps": steps,
        "k": k,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
    }
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "metrics": metrics}, checkpoint)
    payload = {"out_dir": str(out), **metrics}
    _write_smoke_artifacts(out, payload)
    return payload


def evaluate_artifacts(
    cache_dir: str | Path = "runs/openvla_sae/cache_l24",
    checkpoint: str | Path = "runs/openvla_sae/sae_l24_topk.pt",
    out_dir: str | Path = "runs/openvla_sae/eval",
) -> dict[str, Any]:
    cache = _path(cache_dir, "runs/openvla_sae/cache_l24")
    ckpt = _path(checkpoint, "runs/openvla_sae/sae_l24_topk.pt")
    out = _path(out_dir, "runs/openvla_sae/eval")
    out.mkdir(parents=True, exist_ok=True)
    deps = _dependency_status()
    cache_files = sorted(cache.glob("*.pt")) if cache.exists() else []
    payload: dict[str, Any] = {
        "ok": False,
        "cache_dir": str(cache),
        "checkpoint": str(ckpt),
        "out_dir": str(out),
        "cache_exists": cache.exists(),
        "cache_files": len(cache_files),
        "checkpoint_exists": ckpt.exists(),
        "dependencies": deps,
        "metrics": {},
        "next_actions": [],
    }
    payload["metrics"].update(_read_neighbor_metrics(ckpt, cache))
    if not cache.exists():
        payload["next_actions"].append("Cache activations before evaluation.")
    elif not cache_files:
        payload["next_actions"].append(f"No .pt activation files found in {cache}.")
    if not ckpt.exists():
        payload["next_actions"].append("Train an SAE checkpoint before evaluating reconstruction/action effects.")
    if deps["torch"] and (cache_files or ckpt.exists()):
        try:
            payload.update(_torch_artifact_summary(cache_files, ckpt))
        except Exception as exc:  # noqa: BLE001 - artifact inspection should stay actionable
            payload["next_actions"].append(f"Torch artifact inspection failed: {str(exc)[:160]}")
    elif cache_files or ckpt.exists():
        payload["next_actions"].append("Install torch to inspect tensor shapes and checkpoint contents.")
    payload["ok"] = bool(cache_files) and ckpt.exists() and not payload["next_actions"]
    _write_eval_artifacts(out, payload)
    return payload


def feature_report(
    cache_dir: str | Path = "runs/openvla_sae/cache_l24",
    checkpoint: str | Path = "runs/openvla_sae/sae_l24_topk.pt",
    out_dir: str | Path = "runs/openvla_sae/features",
    *,
    top_k: int = 20,
    max_files: int = 64,
) -> dict[str, Any]:
    cache = _path(cache_dir, "runs/openvla_sae/cache_l24")
    ckpt = _path(checkpoint, "runs/openvla_sae/sae_l24_topk.pt")
    out = _path(out_dir, "runs/openvla_sae/features")
    out.mkdir(parents=True, exist_ok=True)
    deps = _dependency_status()
    max_files = _positive_int(max_files, 64)
    top_k = _positive_int(top_k, 20)
    cache_files = sorted(cache.glob("*.pt"))[:max_files] if cache.exists() else []
    payload: dict[str, Any] = {
        "ok": False,
        "cache_dir": str(cache),
        "checkpoint": str(ckpt),
        "out_dir": str(out),
        "cache_files_used": len(cache_files),
        "top_k": top_k,
        "features": [],
        "next_actions": [],
        "dependencies": deps,
    }
    if not cache_files:
        payload["next_actions"].append("Cache activations before generating a feature report.")
    if not ckpt.exists():
        payload["next_actions"].append("Train an SAE checkpoint before generating a feature report.")
    if not deps["torch"]:
        payload["next_actions"].append("Install torch to encode cache activations and rank SAE features.")
    if cache_files and ckpt.exists() and deps["torch"]:
        try:
            payload.update(_torch_feature_report(cache_files, ckpt, top_k))
        except Exception as exc:  # noqa: BLE001 - feature reports should explain bad artifacts
            payload["next_actions"].append(f"Torch feature extraction failed: {str(exc)[:160]}")
    payload["ok"] = bool(payload["features"]) and not payload["next_actions"]
    _write_feature_artifacts(out, payload)
    return payload


def write_dossier(
    out_dir: str | Path = "runs/openvla_sae/dossier",
    *,
    project_root: str | Path = PROJECT_ROOT,
    manifest: str | Path | None = None,
    cache_dir: str | Path = "runs/openvla_sae/cache_l24",
    checkpoint: str | Path = "runs/openvla_sae/sae_l24_topk.pt",
    eval_dir: str | Path = "runs/openvla_sae/eval",
    features_dir: str | Path = "runs/openvla_sae/features",
) -> dict[str, Any]:
    out = _path(out_dir, "runs/openvla_sae/dossier")
    out.mkdir(parents=True, exist_ok=True)
    st = status(project_root=project_root, manifest=manifest)
    eval_root = _path(eval_dir, "runs/openvla_sae/eval")
    features_root = _path(features_dir, "runs/openvla_sae/features")
    cache = _path(cache_dir, "runs/openvla_sae/cache_l24")
    ckpt = _path(checkpoint, "runs/openvla_sae/sae_l24_topk.pt")
    project = _path(project_root, PROJECT_ROOT)
    eval_payload = _read_json(eval_root / "openvla_sae_eval.json")
    feature_payload = _read_json(features_root / "openvla_sae_features.json")
    outline = project / "paper" / "outline.md"
    payload: dict[str, Any] = {
        "ok": False,
        "out_dir": str(out),
        "status": st,
        "cache_dir": str(cache),
        "checkpoint": str(ckpt),
        "eval": eval_payload,
        "features": feature_payload,
        "outline": str(outline),
        "next_actions": [],
    }
    if not st["manifest"]["exists"]:
        payload["next_actions"].append("Create and validate a real image/instruction manifest.")
    elif st["manifest"]["valid_rows"] == 0:
        payload["next_actions"].append("Add valid manifest rows before caching OpenVLA activations.")
    if not cache.exists():
        payload["next_actions"].append("Cache OpenVLA activations for at least one target site.")
    if not ckpt.exists():
        payload["next_actions"].append("Train a Top-K SAE checkpoint from the activation cache.")
    if not eval_payload:
        payload["next_actions"].append("Run `mechferret sae openvla eval` to summarize artifacts.")
    elif eval_payload.get("ok") is False:
        payload["next_actions"].append("Resolve failed eval checks before drafting claims.")
    if not feature_payload:
        payload["next_actions"].append("Run `mechferret sae openvla features` to build a feature atlas.")
    elif feature_payload.get("ok") is False:
        payload["next_actions"].append("Resolve failed feature-atlas checks before drafting claims.")
    payload["ok"] = not payload["next_actions"]
    _write_dossier_artifacts(out, payload, outline)
    return payload


def write_plan(
    out_dir: str | Path = "runs/openvla_sae/plan",
    *,
    project_root: str | Path = PROJECT_ROOT,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    out = _path(out_dir, "runs/openvla_sae/plan")
    out.mkdir(parents=True, exist_ok=True)
    report = plan_markdown(project_root=project_root, manifest=manifest)
    md = out / "openvla_sae_plan.md"
    payload = status(project_root, manifest)
    json_path = out / "openvla_sae_status.json"
    md.write_text(report, encoding="utf-8")
    json_path.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"markdown": str(md), "json": str(json_path), "ready_gpu": payload["ready_gpu"]}


def plan_markdown(project_root: str | Path = PROJECT_ROOT, manifest: str | Path | None = None) -> str:
    st = status(project_root, manifest)
    commands = command_lines(project_root)
    checks = "\n".join(
        f"- [{'x' if ok else ' '}] `{name}`"
        for name, ok in st["files"].items()
    )
    deps = "\n".join(
        f"- [{'x' if ok else ' '}] `{name}`"
        for name, ok in st["dependencies"].items()
        if name != "cuda"
    )
    next_actions = "\n".join(f"- {action}" for action in st["next_actions"])
    return f"""# OpenVLA SAE Workflow

This is the supported path for prompts like "find SAEs for OpenVLA". It does not pretend this is a text-only head-circuit discovery task; it prepares the activation-cache, SAE-training, and causal-evaluation workflow that the project needs.

## Project Files
{checks}

## Environment
{deps}
- [{'x' if st['dependencies']['cuda'] else ' '}] `cuda`

## Manifest
- path: `{st['manifest']['path']}`
- rows: {st['manifest']['rows']}
- valid rows: {st['manifest']['valid_rows']}
- missing images shown: {len(st['manifest']['missing_images'])}
- parse/field errors shown: {len(st['manifest']['errors'])}

## Commands
```bash
{commands}
```

## Next Actions
{next_actions or "- Ready for the first GPU cache/train run."}
"""


def command_lines(project_root: str | Path = PROJECT_ROOT) -> str:
    root = _path(project_root, PROJECT_ROOT)
    script = root / "scripts" / "phase1_commands.sh"
    if script.exists():
        return f"bash {script}"
    return (
        "MANIFEST=data/openvla_sae_phase1.jsonl \\\n"
        "SITE=language_model.model.layers.24 \\\n"
        "CACHE_DIR=runs/openvla_sae/cache_l24 \\\n"
        "SAE_OUT=runs/openvla_sae/sae_l24_topk.pt \\\n"
        "bash projects/openvla_sae/scripts/phase1_commands.sh"
    )


def print_init_result(payload: dict[str, Any]) -> None:
    print(f"Init ok: {payload['ok']}")
    print(f"Project root: {payload['project_root']}")
    print(f"Files written: {len(payload['files_written'])}")
    if payload.get("existing_files"):
        print("Existing files:")
        for item in payload["existing_files"][:8]:
            print(f"  - {item}")
    if payload["next_actions"]:
        print("Next actions:")
        for action in payload["next_actions"]:
            print(f"  - {action}")


def print_status(payload: dict[str, Any]) -> None:
    print(f"OpenVLA SAE project: {payload['project_root']}")
    print(f"Project files: {sum(payload['files'].values())}/{len(payload['files'])}")
    print(f"Template available: {payload['template_available']}")
    print(f"Manifest: {payload['manifest']['path']} ({payload['manifest']['valid_rows']} valid rows)")
    print(f"Ready local: {payload['ready_local']}")
    print(f"Ready GPU:   {payload['ready_gpu']}")
    if payload["next_actions"]:
        print("Next actions:")
        for action in payload["next_actions"]:
            print(f"  - {action}")


def print_plan_result(payload: dict[str, Any]) -> None:
    print(f"Plan: {payload['markdown']}")
    print(f"Status JSON: {payload['json']}")
    print(f"Ready GPU: {payload['ready_gpu']}")


def print_manifest_result(payload: dict[str, Any]) -> None:
    print(f"Manifest: {payload['path']}")
    print(f"Rows written: {payload['rows_written']}")
    print(f"Valid rows: {payload['valid_rows']}")
    if payload.get("errors") or payload.get("missing_images"):
        print("Warnings:")
        for item in payload.get("errors", [])[:6]:
            print(f"  - {item}")
        for item in payload.get("missing_images", [])[:6]:
            print(f"  - missing image: {item}")


def print_smoke_result(payload: dict[str, Any]) -> None:
    print(f"Smoke ok: {payload['ok']}")
    print(f"Out: {payload['out_dir']}")
    if payload.get("report"):
        print(f"Report: {payload['report']}")
    if not payload["ok"]:
        print(f"Reason: {payload['reason']}")
        print(f"Install: {payload['install']}")
        return
    print(f"Device: {payload['device']}")
    print(f"Loss: {payload['initial_loss']:.4f} -> {payload['final_loss']:.4f}")
    print(f"Checkpoint: {payload['checkpoint']}")


def print_eval_result(payload: dict[str, Any]) -> None:
    print(f"Eval ok: {payload['ok']}")
    print(f"Cache: {payload['cache_dir']} ({payload['cache_files']} .pt files)")
    print(f"Checkpoint: {payload['checkpoint']} ({'found' if payload['checkpoint_exists'] else 'missing'})")
    if payload.get("report"):
        print(f"Report: {payload['report']}")
    if payload["metrics"]:
        print(f"Metrics: {payload['metrics']}")
    if payload["next_actions"]:
        print("Next actions:")
        for action in payload["next_actions"]:
            print(f"  - {action}")


def print_feature_result(payload: dict[str, Any]) -> None:
    print(f"Features ok: {payload['ok']}")
    print(f"Cache: {payload['cache_dir']} ({payload['cache_files_used']} files used)")
    print(f"Checkpoint: {payload['checkpoint']}")
    if payload.get("report"):
        print(f"Report: {payload['report']}")
    if payload["features"]:
        print(f"Top features: {len(payload['features'])}")
    if payload["next_actions"]:
        print("Next actions:")
        for action in payload["next_actions"]:
            print(f"  - {action}")


def print_dossier_result(payload: dict[str, Any]) -> None:
    print(f"Dossier ok: {payload['ok']}")
    print(f"Dossier: {payload['markdown']}")
    print(f"JSON: {payload['json']}")
    if payload["next_actions"]:
        print("Next actions:")
        for action in payload["next_actions"]:
            print(f"  - {action}")


def _write_smoke_artifacts(out: Path, payload: dict[str, Any]) -> None:
    metrics = out / "metrics.json"
    report = out / "smoke_report.md"
    payload["metrics"] = str(metrics)
    payload["report"] = str(report)
    metrics.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if payload["ok"]:
        body = f"""# OpenVLA SAE Smoke Test

Status: passed

- device: `{payload['device']}`
- tokens: {payload['tokens']}
- d_model: {payload['d_model']}
- steps: {payload['steps']}
- k: {payload['k']}
- loss: {payload['initial_loss']:.4f} -> {payload['final_loss']:.4f}
- checkpoint: `{payload['checkpoint']}`
- synthetic cache: `{payload['cache_dir']}`

This verifies the local Top-K SAE training path on synthetic activations. It does not validate OpenVLA model loading, activation hooks, or real robot-policy features.
"""
    else:
        body = f"""# OpenVLA SAE Smoke Test

Status: blocked before training

Reason: {payload['reason']}

Install:

```bash
{payload['install']}
```

After installing dependencies, rerun:

```bash
mechferret sae openvla smoke --out {payload['out_dir']}
```

This smoke test is intentionally tiny. It checks the local SAE training path before you spend time loading OpenVLA or caching real activations.
"""
    report.write_text(body, encoding="utf-8")


def _read_neighbor_metrics(checkpoint: Path, cache: Path) -> dict[str, Any]:
    candidates = [
        checkpoint.with_suffix(".json"),
        checkpoint.parent / "metrics.json",
        cache.parent / "metrics.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        return {
            _text(k, str(k)): (_number(v) if isinstance(v, (int, float)) else v)
            for k, v in data.items()
            if isinstance(v, (int, float, str, bool))
        }
    return {}


def _torch_artifact_summary(cache_files: list[Path], checkpoint: Path) -> dict[str, Any]:
    import torch  # type: ignore

    out: dict[str, Any] = {"tensor_summary": {}, "checkpoint_summary": {}}
    if cache_files:
        sample = torch.load(cache_files[0], map_location="cpu")
        tensor = sample
        if isinstance(sample, dict):
            tensor = next((value for value in sample.values() if torch.is_tensor(value)), None)
        if tensor is not None:
            out["tensor_summary"] = {
                "sample_file": str(cache_files[0]),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            out["checkpoint_summary"] = {
                "keys": sorted(str(key) for key in ckpt.keys())[:20],
                "d_in": ckpt.get("d_in"),
                "d_sae": ckpt.get("d_sae"),
                "k": ckpt.get("k"),
            }
    return out


def _write_eval_artifacts(out: Path, payload: dict[str, Any]) -> None:
    metrics = out / "openvla_sae_eval.json"
    report = out / "openvla_sae_eval.md"
    payload["json"] = str(metrics)
    payload["report"] = str(report)
    metrics.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    actions = "\n".join(f"- {action}" for action in payload["next_actions"]) or "- Ready for deeper feature/action evaluation."
    tensor = payload.get("tensor_summary", {})
    checkpoint = payload.get("checkpoint_summary", {})
    body = f"""# OpenVLA SAE Artifact Evaluation

Status: {'passed' if payload['ok'] else 'needs work'}

## Inputs
- cache: `{payload['cache_dir']}` ({payload['cache_files']} `.pt` files)
- checkpoint: `{payload['checkpoint']}` ({'found' if payload['checkpoint_exists'] else 'missing'})

## Metrics
```json
{_json_dumps(payload['metrics'], indent=2, sort_keys=True)}
```

## Tensor Summary
```json
{_json_dumps(tensor, indent=2, sort_keys=True)}
```

## Checkpoint Summary
```json
{_json_dumps(checkpoint, indent=2, sort_keys=True)}
```

## Next Actions
{actions}
"""
    report.write_text(body, encoding="utf-8")


def _torch_feature_report(cache_files: list[Path], checkpoint: Path, top_k: int) -> dict[str, Any]:
    import torch  # type: ignore

    src_dir = Path(PROJECT_ROOT) / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from simple_topk_sae import TopKSAE  # type: ignore

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = TopKSAE(d_in=int(ckpt["d_in"]), d_sae=int(ckpt["d_sae"]), k=int(ckpt["k"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    feature_max: dict[int, float] = {}
    feature_mean_sum: dict[int, float] = {}
    feature_count: dict[int, int] = {}
    top_examples: dict[int, list[dict[str, Any]]] = {}
    tokens_seen = 0
    with torch.no_grad():
        for path in cache_files:
            sample = torch.load(path, map_location="cpu")
            tensor = sample
            row = {}
            if isinstance(sample, dict):
                row = sample.get("row", {}) if isinstance(sample.get("row", {}), dict) else {}
                tensor = next((value for value in sample.values() if torch.is_tensor(value)), None)
            if tensor is None:
                continue
            x = tensor.reshape(-1, tensor.shape[-1]).float()
            if mean is not None and std is not None:
                x = (x - mean) / std.clamp_min(1e-6)
            z = model.encode(x)
            tokens_seen += int(z.shape[0])
            active = torch.nonzero(z > 0, as_tuple=False)
            for token_idx, feat_idx in active.tolist():
                value = float(z[token_idx, feat_idx])
                feature_max[feat_idx] = max(feature_max.get(feat_idx, 0.0), value)
                feature_mean_sum[feat_idx] = feature_mean_sum.get(feat_idx, 0.0) + value
                feature_count[feat_idx] = feature_count.get(feat_idx, 0) + 1
                bucket = top_examples.setdefault(feat_idx, [])
                bucket.append({
                    "value": value,
                    "file": str(path),
                    "token_index": token_idx,
                    "instruction": str(row.get("instruction", ""))[:160],
                    "image_path": str(row.get("image_path", ""))[:240],
                })
                bucket.sort(key=lambda item: item["value"], reverse=True)
                del bucket[3:]
    ranked = sorted(feature_max, key=lambda feat: feature_max[feat], reverse=True)[:top_k]
    features = [
        {
            "feature": feat,
            "max_activation": round(feature_max[feat], 6),
            "mean_active_activation": round(feature_mean_sum[feat] / max(feature_count[feat], 1), 6),
            "active_count": feature_count[feat],
            "top_examples": top_examples.get(feat, []),
        }
        for feat in ranked
    ]
    return {"features": features, "tokens_seen": tokens_seen}


def _write_feature_artifacts(out: Path, payload: dict[str, Any]) -> None:
    json_path = out / "openvla_sae_features.json"
    md_path = out / "openvla_sae_features.md"
    payload["json"] = str(json_path)
    payload["report"] = str(md_path)
    json_path.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for item in payload["features"][:50]:
        examples = "; ".join(
            ex.get("instruction") or ex.get("file", "")
            for ex in item.get("top_examples", [])[:2]
        )
        rows.append(
            f"| {item['feature']} | {item['max_activation']:.4f} | {item['mean_active_activation']:.4f} | "
            f"{item['active_count']} | {examples} |"
        )
    table = "\n".join(rows) if rows else "| - | - | - | - | - |"
    actions = "\n".join(f"- {action}" for action in payload["next_actions"]) or "- Inspect top examples and label coherent features."
    body = f"""# OpenVLA SAE Feature Report

Status: {'passed' if payload['ok'] else 'needs work'}

- cache: `{payload['cache_dir']}`
- checkpoint: `{payload['checkpoint']}`
- cache files used: {payload['cache_files_used']}
- tokens seen: {payload.get('tokens_seen', 0)}

| Feature | Max activation | Mean active activation | Active count | Top examples |
| --- | ---: | ---: | ---: | --- |
{table}

## Next Actions
{actions}
"""
    md_path.write_text(body, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _template_root():
    return resources.files("mechferret").joinpath(TEMPLATE_ROOT)


def _template_available() -> bool:
    try:
        root = _template_root()
        return all(root.joinpath(*Path(name).parts).is_file() for name in REQUIRED_FILES)
    except (FileNotFoundError, ModuleNotFoundError):
        return False


def _template_file_names(root) -> list[Path]:
    names: list[Path] = []

    def walk(node, prefix: Path = Path("")) -> None:
        for child in node.iterdir():
            if child.name == "__pycache__" or child.name.endswith(".pyc"):
                continue
            rel = prefix / child.name
            if child.is_dir():
                walk(child, rel)
            elif child.is_file():
                names.append(rel)

    walk(root)
    return names


def _copy_template_tree(source, target: Path, *, written: list[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "__pycache__" or child.name.endswith(".pyc"):
            continue
        dest = target / child.name
        if child.is_dir():
            _copy_template_tree(child, dest, written=written)
            continue
        if not child.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(child.read_bytes())
        if dest.suffix == ".sh":
            dest.chmod(dest.stat().st_mode | 0o111)
        written.append(str(dest))


def _artifact_report_path(payload: dict[str, Any]) -> str:
    report = payload.get("report")
    if isinstance(report, str) and report:
        return report
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        report = artifacts.get("report")
        if isinstance(report, str) and report:
            return report
    return "missing"


def _feature_snapshot_row(item: dict[str, Any]) -> str:
    label = item.get("feature", item.get("id", "?"))
    parts = []
    for key in ("max_activation", "active_count", "score"):
        value = item.get(key)
        if value is not None:
            parts.append(f"{key}={_json_safe(value)}")
    examples = item.get("examples")
    if isinstance(examples, list) and examples:
        parts.append(f"example={compact_text(str(examples[0]), 80)}")
    detail = "; ".join(parts) if parts else "no summary metrics"
    return f"- feature {label}: {detail}"


def _write_dossier_artifacts(out: Path, payload: dict[str, Any], outline: Path) -> None:
    json_path = out / "openvla_sae_dossier.json"
    md_path = out / "openvla_sae_dossier.md"
    payload["json"] = str(json_path)
    payload["markdown"] = str(md_path)
    json_path.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    outline_text = outline.read_text(encoding="utf-8", errors="ignore") if outline.exists() else "(missing outline)"
    status_payload = _mapping(payload.get("status"))
    manifest_payload = _mapping(status_payload.get("manifest"))
    eval_payload = _mapping(payload.get("eval"))
    feature_payload = _mapping(payload.get("features"))
    actions = "\n".join(f"- {_text(action)}" for action in _items(payload.get("next_actions"))) or "- Ready for paper drafting and causal feature tests."
    features = _items(feature_payload.get("features", []))
    feature_rows = "\n".join(
        _feature_snapshot_row(item)
        for item in features[:12]
        if isinstance(item, dict)
    ) or "- No ranked features yet."
    body = f"""# OpenVLA SAE Research Dossier

Status: {'ready' if payload['ok'] else 'in progress'}

## Current State
- manifest: `{_text(manifest_payload.get('path'))}` ({int(_number(manifest_payload.get('valid_rows')))} valid rows)
- cache: `{payload['cache_dir']}`
- checkpoint: `{payload['checkpoint']}`
- eval report: `{_artifact_report_path(eval_payload)}`
- feature report: `{_artifact_report_path(feature_payload)}`

## Artifact Metrics
```json
{_json_dumps(_mapping(eval_payload.get('metrics')), indent=2, sort_keys=True)}
```

## Feature Snapshot
{feature_rows}

## Paper Outline
{outline_text}

## Next Actions
{actions}
"""
    md_path.write_text(body, encoding="utf-8")


def _dependency_status() -> dict[str, bool]:
    deps = {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "PIL": importlib.util.find_spec("PIL") is not None,
        "yaml": importlib.util.find_spec("yaml") is not None,
        "tqdm": importlib.util.find_spec("tqdm") is not None,
    }
    deps["cuda"] = False
    if deps["torch"]:
        try:
            import torch  # type: ignore

            deps["cuda"] = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - status should never crash
            deps["cuda"] = False
    return deps
