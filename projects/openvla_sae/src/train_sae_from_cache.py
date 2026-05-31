"""Train a minimal Top-K SAE from cached activation tensors.

Expected cache format: one or more `.pt` files containing tensors shaped [..., d_model].
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import yaml
import torch
import torch.nn.functional as F
from tqdm import trange

from simple_topk_sae import TopKSAE


def load_cache(cache_dir: Path, max_tokens: int | None = None) -> torch.Tensor:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/openvla_sae/configs/phase1.yaml")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sae_cfg = cfg["sae"]
    seed = int(sae_cfg.get("seed", 0))
    torch.manual_seed(seed); random.seed(seed)

    x = load_cache(Path(args.cache_dir), args.max_tokens)
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    if sae_cfg.get("normalize_activations", True):
        x_train = (x - mean) / std
    else:
        x_train = x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in = x_train.shape[-1]
    d_sae = int(d_in * sae_cfg["expansion_factor"])
    model = TopKSAE(d_in=d_in, d_sae=d_sae, k=int(sae_cfg["k"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sae_cfg["lr"]))

    bs = int(sae_cfg["batch_size"])
    steps = int(sae_cfg["steps"])
    x_train = x_train.pin_memory()

    for step in trange(steps):
        idx = torch.randint(0, x_train.shape[0], (bs,))
        batch = x_train[idx].to(device, non_blocking=True)
        x_hat, z, aux = model(batch)
        loss = F.mse_loss(x_hat, batch)
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
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
