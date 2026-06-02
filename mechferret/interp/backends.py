"""Backend selection and the optional real TransformerLens backend.

Every probe is written against a small duck-typed backend interface, so the
*identical* probe + engine code runs whether the numbers come from the offline
:class:`~mechferret.interp.synthetic.SyntheticBackend` or a real model measured
with :class:`TransformerLensBackend`.

Resolution order for an explicit ``backend="auto"`` request:

1. ``transformer_lens`` if importable (real measurement), unless disabled.
2. ``synthetic`` otherwise (always available locally).

If TransformerLens is importable but loading the requested model fails, auto
raises that error instead of silently substituting synthetic measurements.
Set ``MECHFERRET_FORCE_SYNTHETIC=1`` or pass ``--backend synthetic`` when the
offline smoke backend is intentional.
"""

from __future__ import annotations

import importlib.util
import os

from .synthetic import SyntheticBackend
from .tasks import get_task


def transformer_lens_available() -> bool:
    if os.getenv("MECHFERRET_FORCE_SYNTHETIC"):
        return False
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformer_lens") is not None
    )


def resolve_backend(model: str | None, backend: str | None = None):
    """Return a backend instance honouring an explicit request or auto-detection."""

    model_name = _model_name(model)
    choice = (backend or "").strip().lower()
    if not choice:
        raise ValueError(
            "backend is required; pass backend='synthetic' for smoke data or "
            "backend='transformer_lens'/'real' for real measurements."
        )
    if choice == "synthetic":
        return SyntheticBackend(model_name)
    if choice in {"transformer_lens", "tl", "real"}:
        if not transformer_lens_available():
            raise RuntimeError(
                "transformer_lens backend requested but torch/transformer_lens are not installed. "
                "Install with `pip install -e '.[interp]'` or run with --backend synthetic."
            )
        return TransformerLensBackend(model_name)
    if choice != "auto":
        raise ValueError(
            "unknown backend; pass backend='synthetic', 'transformer_lens', 'tl', 'real', "
            "or explicit backend='auto'."
        )
    # Explicit auto / modal-local: prefer real measurement when the deps are present.
    if transformer_lens_available():
        try:
            return TransformerLensBackend(model_name)
        except Exception as exc:  # pragma: no cover - depends on optional deps/model load
            raise RuntimeError(
                f"transformer_lens backend is available, but loading model {model_name!r} failed. "
                "Pass --backend synthetic or set MECHFERRET_FORCE_SYNTHETIC=1 only if synthetic smoke data is intentional."
            ) from exc
    return SyntheticBackend(model_name)


def _model_name(model: str | None) -> str:
    name = (model or "").strip() if isinstance(model, str) else ""
    if not name:
        raise ValueError("model is required; pass --model or use a skill that declares one.")
    return name


class TransformerLensBackend:  # pragma: no cover - exercised only with torch installed
    """Measures a real HookedTransformer. Same method surface as SyntheticBackend."""

    name = "transformer_lens"
    available = True

    def __init__(self, model: str) -> None:
        import torch  # noqa: F401
        from transformer_lens import HookedTransformer

        self.model_name = _model_name(model).lower()
        self.model = HookedTransformer.from_pretrained(self.model_name)
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        self.d_model = self.model.cfg.d_model

    # --- helpers ----------------------------------------------------------------

    def _answer_tokens(self, task):
        return [
            (self.model.to_single_token(" " + correct), self.model.to_single_token(" " + wrong))
            for correct, wrong in task.answers
        ]

    def _logit_diff(self, logits, answer_tokens) -> float:
        import torch

        last = logits[:, -1, :]
        diffs = []
        for row, (correct, wrong) in zip(last, answer_tokens):
            diffs.append((row[correct] - row[wrong]).item())
        return float(torch.tensor(diffs).mean())

    def _prompts(self, task, seed: int):
        prompts = list(task.clean_prompts)
        shift = seed % max(1, len(prompts))
        return prompts[shift:] + prompts[:shift]

    # --- probe-facing measurements ----------------------------------------------

    def clean_metric(self, task_name: str, seed: int) -> float:
        task = get_task(task_name)
        tokens = self.model.to_tokens(self._prompts(task, seed))
        logits = self.model(tokens)
        return round(self._logit_diff(logits, self._answer_tokens(task)), 4)

    def head_ablation_effect(self, task_name: str, layer: int, head: int, seed: int) -> float:
        import torch

        task = get_task(task_name)
        answer_tokens = self._answer_tokens(task)
        tokens = self.model.to_tokens(self._prompts(task, seed))
        clean_logits, cache = self.model.run_with_cache(tokens)
        clean = self._logit_diff(clean_logits, answer_tokens)
        mean_z = cache[f"blocks.{layer}.attn.hook_z"].mean(dim=0, keepdim=True)

        def hook(z, hook):  # noqa: ANN001
            z[:, :, head, :] = mean_z[:, :, head, :]
            return z

        ablated_logits = self.model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)]
        )
        ablated = self._logit_diff(ablated_logits, answer_tokens)
        return round(clean - ablated, 4)

    def patch_recovery(self, task_name: str, layer: int, head: int, seed: int) -> float:
        task = get_task(task_name)
        answer_tokens = self._answer_tokens(task)
        clean_tokens = self.model.to_tokens(self._prompts(task, seed))
        corrupt_tokens = self.model.to_tokens(list(task.corrupt_prompts))
        _, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupt_logits = self.model(corrupt_tokens)
        corrupt = self._logit_diff(corrupt_logits, answer_tokens)
        clean = self.clean_metric(task_name, seed)

        def hook(z, hook):  # noqa: ANN001
            z[:, :, head, :] = clean_cache[f"blocks.{layer}.attn.hook_z"][:, :, head, :]
            return z

        patched_logits = self.model.run_with_hooks(
            corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)]
        )
        patched = self._logit_diff(patched_logits, answer_tokens)
        denom = (clean - corrupt) or 1e-6
        return round((patched - corrupt) / denom, 4)

    def attention_score(self, task_name: str, layer: int, head: int, seed: int) -> dict[str, float]:
        task = get_task(task_name)
        tokens = self.model.to_tokens(self._prompts(task, seed))
        _, cache = self.model.run_with_cache(tokens)
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]  # [batch, q, k]
        prev = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean().item()
        cur = pattern.diagonal(offset=0, dim1=-2, dim2=-1).mean().item()
        return {
            "induction": round(prev, 4),
            "previous_token": round(prev, 4),
            "duplicate_token": round(prev, 4),
            "current_token": round(cur, 4),
        }

    def direct_logit_attribution(self, task_name: str, layer: int, head: int, seed: int) -> float:
        # Approximate DLA with the head's ablation effect (sign-consistent proxy).
        return self.head_ablation_effect(task_name, layer, head, seed)

    def logit_lens(self, task_name: str, seed: int) -> list[dict[str, float]]:
        import torch

        task = get_task(task_name)
        answer_tokens = self._answer_tokens(task)
        tokens = self.model.to_tokens(self._prompts(task, seed))
        _, cache = self.model.run_with_cache(tokens)
        rows: list[dict[str, float]] = []
        for layer in range(self.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            resid = self.model.ln_final(resid)
            logits = self.model.unembed(resid)
            ld = self._logit_diff(logits, answer_tokens)
            last = logits[:, -1, :].softmax(dim=-1)
            correct_prob = float(
                torch.tensor([row[c].item() for row, (c, _) in zip(last, answer_tokens)]).mean()
            )
            rows.append({"layer": layer, "correct_prob": round(correct_prob, 4), "logit_diff": round(ld, 4)})
        return rows

    def control_head(self, task_name: str, seed: int) -> tuple[int, int]:
        return ((seed + 1) % self.n_layers, (seed + 3) % self.n_heads)

    def top_heads(self, task_name: str):  # not used for TL; kept for parity
        return []
