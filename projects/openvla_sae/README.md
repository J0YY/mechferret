# SAE Interpretability for OpenVLA

Working project for a paper on sparse autoencoders (SAEs) for OpenVLA / vision-language-action models.

Core claim to test: SAEs trained on OpenVLA activations recover sparse, interpretable, and causally action-relevant features for robot policies.

## Structure
- `paper/`: paper outline and draft skeleton
- `src/`: experiment code
- `configs/`: SAE/model configs
- `scripts/`: runnable commands
- `results/`: metrics and cached summaries
- `figures/`: generated plots

## Minimal workflow
1. Install OpenVLA deps on a GPU machine.
2. Cache activations from `openvla/openvla-7b`.
3. Train Top-K SAEs on selected activation sites.
4. Evaluate reconstruction/action KL.
5. Find top activating examples.
6. Run causal ablation/steering tests.

See `paper/outline.md` and `configs/phase1.yaml`.
