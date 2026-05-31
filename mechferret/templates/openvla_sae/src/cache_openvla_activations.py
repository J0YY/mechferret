"""Cache OpenVLA activations for SAE training.

This is a first-pass hook script. It uses HF remote code, so run only in a trusted env.
Input manifest JSONL rows: {"image_path": "...", "instruction": "pick up ...", "action": optional}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--site", required=True, help="Dotted module path to hook")
    ap.add_argument("--max-examples", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--dry-run", action="store_true", help="Validate manifest/dependencies without loading the model.")
    return ap


def dependency_status() -> dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "PIL": importlib.util.find_spec("PIL") is not None,
        "tqdm": importlib.util.find_spec("tqdm") is not None,
    }


def load_manifest(path: str | Path, max_examples: int) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        missing = [key for key in ("image_path", "instruction") if not row.get(key)]
        if missing:
            errors.append(f"line {line_no}: missing {', '.join(missing)}")
            continue
        if not Path(row["image_path"]).exists():
            errors.append(f"line {line_no}: image not found: {row['image_path']}")
            continue
        rows.append(row)
        if len(rows) >= max_examples:
            break
    return rows, errors


def dry_run_report(args: argparse.Namespace) -> dict:
    rows, errors = load_manifest(args.manifest, args.max_examples)
    deps = dependency_status()
    return {
        "ok": bool(rows) and not errors and all(deps.values()),
        "model": args.model,
        "site": args.site,
        "manifest": args.manifest,
        "out_dir": args.out_dir,
        "max_examples": args.max_examples,
        "valid_rows": len(rows),
        "errors": errors[:20],
        "dependencies": deps,
    }


def get_submodule(root, dotted: str):
    obj = root
    for name in dotted.split('.'):
        obj = getattr(obj, name)
    return obj


def main():
    args = build_parser().parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_report(args), indent=2, sort_keys=True))
        return

    try:
        from PIL import Image  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency for OpenVLA activation caching: {exc.name}. "
            "Install with projects/openvla_sae/scripts/install_openvla_min.sh, "
            "or run this command with --dry-run first."
        ) from None

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()

    acts = []
    def hook(_module, _inp, out):
        y = out[0] if isinstance(out, tuple) else out
        acts.append(y.detach().float().cpu())

    handle = get_submodule(model, args.site).register_forward_hook(hook)

    rows, errors = load_manifest(args.manifest, args.max_examples)
    if errors:
        raise SystemExit("Manifest is not ready:\n" + "\n".join(errors[:20]))
    prompt_template = "In: What action should the robot take to {instruction}?\nOut:"

    with torch.no_grad():
        for i, row in enumerate(tqdm(rows)):
            acts.clear()
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = prompt_template.format(instruction=row["instruction"])
            inputs = processor(prompt, image).to(device, dtype=dtype)
            # Force a short generation/prediction path. If predict_action exists, prefer it.
            if hasattr(model, "predict_action"):
                _ = model.predict_action(**inputs, do_sample=False)
            else:
                _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            if not acts:
                raise RuntimeError(f"Hook site produced no activations: {args.site}")
            torch.save({"activation": acts[0], "row": row}, out_dir / f"{i:06d}.pt")

    handle.remove()
    print(f"cached {len(rows)} examples to {out_dir}")


if __name__ == "__main__":
    main()
