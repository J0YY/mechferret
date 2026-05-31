#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
pip install torch torchvision transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 accelerate einops datasets pillow pyyaml tqdm matplotlib scikit-learn
