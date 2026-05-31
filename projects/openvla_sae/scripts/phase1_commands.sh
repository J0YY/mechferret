#!/usr/bin/env bash
set -euo pipefail

# Edit these paths for your GPU machine.
MANIFEST=${MANIFEST:-data/openvla_sae_phase1.jsonl}
SITE=${SITE:-language_model.model.layers.24}
CACHE_DIR=${CACHE_DIR:-runs/openvla_sae/cache_l24}
SAE_OUT=${SAE_OUT:-runs/openvla_sae/sae_l24_topk.pt}

python projects/openvla_sae/src/cache_openvla_activations.py \
  --manifest "$MANIFEST" \
  --out-dir "$CACHE_DIR" \
  --site "$SITE" \
  --max-examples 2048

python projects/openvla_sae/src/train_sae_from_cache.py \
  --cache-dir "$CACHE_DIR" \
  --out "$SAE_OUT"
