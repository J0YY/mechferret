#!/usr/bin/env bash
set -euo pipefail

# Edit these paths for your GPU machine.
CONFIG=${CONFIG:-projects/openvla_sae/configs/phase1.yaml}
MODEL=${MODEL:-}
MANIFEST=${MANIFEST:-data/openvla_sae_phase1.jsonl}
SITE=${SITE:-language_model.model.layers.24}
CACHE_DIR=${CACHE_DIR:-runs/openvla_sae/cache_l24}
SAE_OUT=${SAE_OUT:-runs/openvla_sae/sae_l24_topk.pt}
CACHE_MAX_EXAMPLES=${CACHE_MAX_EXAMPLES:-2048}
SAE_STEPS=${SAE_STEPS:-}
SAE_BATCH_SIZE=${SAE_BATCH_SIZE:-}
SAE_K=${SAE_K:-}
SAE_MAX_TOKENS=${SAE_MAX_TOKENS:-2000000}

cache_args=(
  --config "$CONFIG"
  --manifest "$MANIFEST"
  --out-dir "$CACHE_DIR"
  --site "$SITE"
  --max-examples "$CACHE_MAX_EXAMPLES"
)
if [[ -n "$MODEL" ]]; then cache_args+=(--model "$MODEL"); fi

python projects/openvla_sae/src/cache_openvla_activations.py "${cache_args[@]}"

train_args=(
  --cache-dir "$CACHE_DIR"
  --out "$SAE_OUT"
  --max-tokens "$SAE_MAX_TOKENS"
)
if [[ -n "$SAE_STEPS" ]]; then train_args+=(--steps "$SAE_STEPS"); fi
if [[ -n "$SAE_BATCH_SIZE" ]]; then train_args+=(--batch-size "$SAE_BATCH_SIZE"); fi
if [[ -n "$SAE_K" ]]; then train_args+=(--k "$SAE_K"); fi

python projects/openvla_sae/src/train_sae_from_cache.py "${train_args[@]}"
