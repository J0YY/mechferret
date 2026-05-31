"""Cache OpenVLA activations for SAE training.

This is a first-pass hook script. It uses HF remote code, so run only in a trusted env.
Input manifest JSONL rows: {"image_path": "...", "instruction": "pick up ...", "action": optional}
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm


def get_submodule(root, dotted: str):
    obj = root
    for name in dotted.split('.'):
        obj = getattr(obj, name)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--site", required=True, help="Dotted module path to hook")
    ap.add_argument("--max-examples", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

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

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    rows = rows[: args.max_examples]
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
