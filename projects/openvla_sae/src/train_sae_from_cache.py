"""Train a minimal Top-K SAE from cached activation tensors.

Expected cache format: one or more `.pt` files containing tensors shaped [..., d_model].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import secrets
from typing import Any

DEFAULT_CONFIG = {
    "sae": {
        "architecture": "topk",
        "expansion_factor": 16,
        "k": 64,
        "lr": 0.0003,
        "batch_size": 4096,
        "steps": 20000,
        "normalize_activations": True,
    }
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON/YAML config without requiring PyYAML for the simple phase1 file."""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    last_key_by_indent: dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            key = last_key_by_indent.get(indent)
            if key:
                parent.setdefault(key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        last_key_by_indent[indent + 2] = key
        if value:
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for section, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section].update(value)
        else:
            merged[section] = value
    sae = merged.setdefault("sae", {})
    mapping = {
        "steps": "steps",
        "batch_size": "batch_size",
        "k": "k",
        "lr": "lr",
        "expansion_factor": "expansion_factor",
        "seed": "seed",
    }
    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            sae[cfg_name] = value
    return merged


def resolve_seed(value: Any = None) -> tuple[int, str]:
    if value is None or value == "":
        return secrets.SystemRandom().randrange(1, 2**31 - 1), "run_generated"
    if type(value) is bool:
        raise ValueError("sae.seed must be an integer, not a boolean")
    seed = int(value)
    if seed < 0:
        raise ValueError("sae.seed must be non-negative")
    return seed, "explicit"


def load_cache(cache_dir: Path, max_tokens: int | None = None):
    import torch  # type: ignore

    tensors = []
    for p in sorted(cache_dir.glob("*.pt")):
        x = torch.load(p, map_location="cpu")
        if isinstance(x, dict):
            # Use first tensor value by default.
            x = next(v for v in x.values() if torch.is_tensor(v))
        x = x.reshape(-1, x.shape[-1]).float()
        tensors.append(x)
        if max_tokens and sum(t.shape[0] for t in tensors) >= max_tokens:
            break
    if not tensors:
        raise FileNotFoundError(f"No .pt activation files in {cache_dir}")
    x = torch.cat(tensors, dim=0)
    if max_tokens:
        x = x[:max_tokens]
    return x


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/openvla_sae/configs/phase1.yaml")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--steps", type=int, help="Override sae.steps for quick runs.")
    ap.add_argument("--batch-size", type=int, help="Override sae.batch_size.")
    ap.add_argument("--k", type=int, help="Override sae.k.")
    ap.add_argument("--lr", type=float, help="Override sae.lr.")
    ap.add_argument("--expansion-factor", type=int, help="Override sae.expansion_factor.")
    ap.add_argument("--seed", type=int, help="Override sae.seed.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--no-pin-memory", action="store_true", help="Disable pinned host memory even on CUDA.")
    return ap


def main():
    args = build_parser().parse_args()

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    from simple_topk_sae import TopKSAE

    try:
        from tqdm import trange  # type: ignore
    except ImportError:
        trange = range

    cfg = apply_overrides(load_config(args.config), args)
    sae_cfg = cfg["sae"]
    seed, seed_source = resolve_seed(sae_cfg.get("seed"))
    sae_cfg["seed"] = seed
    sae_cfg["seed_source"] = seed_source
    torch.manual_seed(seed); random.seed(seed)

    x = load_cache(Path(args.cache_dir), args.max_tokens)
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    if sae_cfg.get("normalize_activations", True):
        x_train = (x - mean) / std
    else:
        x_train = x

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    else:
        device = args.device
    d_in = x_train.shape[-1]
    d_sae = int(d_in * sae_cfg["expansion_factor"])
    model = TopKSAE(d_in=d_in, d_sae=d_sae, k=int(sae_cfg["k"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sae_cfg["lr"]))

    bs = int(sae_cfg["batch_size"])
    steps = int(sae_cfg["steps"])
    if device == "cuda" and not args.no_pin_memory:
        x_train = x_train.pin_memory()

    final_loss = None
    for step in trange(steps):
        idx = torch.randint(0, x_train.shape[0], (bs,))
        batch = x_train[idx].to(device, non_blocking=(device == "cuda" and not args.no_pin_memory))
        x_hat, z, aux = model(batch)
        loss = F.mse_loss(x_hat, batch)
        final_loss = float(loss.detach().cpu())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.normalize_decoder()
        if step % 500 == 0:
            print({"step": step, "loss": float(loss), "l0": float(aux["l0"])})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "d_in": d_in,
        "d_sae": d_sae,
        "k": model.k,
        "mean": mean,
        "std": std,
        "config": cfg,
        "metrics": {
            "steps": steps,
            "batch_size": bs,
            "device": device,
            "final_loss": final_loss,
            "seed": seed,
            "seed_source": seed_source,
        },
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
