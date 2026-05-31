# Paper outline: Sparse Autoencoders for Interpreting OpenVLA

## Working title
Sparse Autoencoders Reveal Action-Relevant Features in Vision-Language-Action Models

## Hypothesis under test
Sparse autoencoders trained on OpenVLA activations may expose object, spatial, task-phase, gripper, and action-direction features; causal tests determine whether any such features mediate action-token predictions.

## Contributions
1. SAE training and evaluation pipeline for OpenVLA activations.
2. Feature atlas for an open 7B vision-language-action robot policy.
3. Causal validation via feature ablation, activation steering, and clean/corrupted patching.
4. Failure-mode analysis: SAE features as interpretable predictors of wrong-object and gripper-timing errors.

## Main experiments

### E1: SAE quality across layers/sites
Activation sites:
- multimodal projector output
- LLM residual stream layers 8, 16, 24, 32
- MLP activations at layers 16, 24, 32
- final hidden state at action-token positions

Metrics:
- reconstruction MSE
- fraction variance explained
- L0 sparsity
- action-token KL under SAE reconstruction
- offline action accuracy drop

### E2: Feature interpretability
For each SAE feature:
- top activating image/instruction/action examples
- activation distribution by task, object, action dimension
- optional visual-token heatmaps
- human or automated label agreement

### E3: Causal feature tests
For candidate features:
- ablate feature during forward pass
- steer feature up/down
- patch feature activations from clean to corrupted examples

Primary outcomes:
- change in action-token logits
- change in decoded 7-DoF action
- task-specificity against matched random-feature controls

### E4: Failure prediction
Use SAE features to predict:
- wrong object
- premature/late gripper closure
- OOD visual scene
- low-confidence/saturated action

Baselines:
- raw residual stream probe
- PCA features
- dense autoencoder features
- random SAE features

## Claims allowed only if supported
- "Interpretable feature": coherent top examples + label/probe support.
- "Action-relevant feature": significant action-logit/action-coordinate effect vs matched random features.
- "Causal mediator": ablation + steering or patching pass, replicated over seeds/tasks.

## Related work to cite
- OpenVLA: https://arxiv.org/abs/2406.09246
- Prisma vision MI/SAE toolkit: https://arxiv.org/abs/2504.19475
- BLIP causal tracing for multimodal MI: https://arxiv.org/abs/2308.14179
