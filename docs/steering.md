# Steering mechanics and usage

This document explains how SAE-based steering is implemented in this repo and how to use/tune it effectively.

## Overview
- We train one JumpReLU SAE per transformer layer on pooled hidden states.
- For selected layers and a top-k set of features per side (Hindi/English), we encode → edit features → decode → inject a norm-clamped residual into the model’s hidden states during generation.
- Multi-layer steering blends deltas from several layers using learned/derived layer weights.

Core implementation: `MultiLayerSteerer` in `he_steer_pipeline.py`.

## Modes
- steer: amplify target-language features and suppress the other side simultaneously.
- shadow_hindi: suppress only Hindi-like features (amplification disabled); goal is EN.
- shadow_english: suppress only English-like features; goal is HI.

Target is automatically inferred from mode for metrics, and used to choose amplify vs suppress sets.

## Feature selection (what gets amplified/suppressed)
- stats (default): rank by |Cohen’s d| + 0.3|Δfreq| + 0.1|magnitude|, split by sign of d.
- probe: train a small linear probe on SAE codes to rank discriminative features.
- causal: start from stats shortlist and re-rank by measured Δ success when steering each single feature on a tiny val set (best for shadow modes).
- label (new): stats shortlist, then re-rank using precomputed Gemini labels if available (no API calls in loop).

Optional constraints: distribution-aware pruning by activation frequency thresholds.

## Hook mechanics (how the delta is injected)
- For each selected layer L, we register a forward hook:
  1) Encode hidden states with the SAE to get feature codes z.
  2) Apply per-feature scaling:
     - Amplify: z[:, i] *= (1 + strength)
     - Suppress: z[:, i] *= max(0, 1 − sup_factor · strength)
  3) Decode base and modified representations to obtain delta = decode(z_mod) − decode(z_base).
  4) Norm clamp per position: scale delta so ||delta|| ≤ clamp_ratio · ||x|| (element-wise per position).
  5) Weighted sum across layers: add w_L · delta back into the model residual stream.

Two intervention scopes:
- last: affect only the last token’s hidden state (fast; keeps KV cache).
- sequence: affect all positions with a position weight ramp (pos_weight_start → pos_weight_end); disables KV cache for correctness, slower but stronger.

## Parameters to tune
- eval_strength (steer-alpha): how aggressively to scale feature codes.
- sup_factor: scales suppression relative to amplification (1.0 by default).
- clamp_ratio (norm-clamp-ratio): safety rail; lower reduces artifacting but weakens effect.
- eval_top_k_features (top-k-per-layer): number of features per side to include.
- intervention_scope: last | sequence.
- pos_weight_start/end: sequence-wide position ramp (1.0 → 0.6 by default).
- layer_weights: normalized per-layer contributions (derived from validation success rates by default).

## Label-aware nudges (optional)
- Set `--feature-ranking label` to re-rank the stats shortlist using labels from `feature_labels.json`.
- Set `--use-feature-labels` to enable slight per-feature multipliers in the hook:
  - Hinglish labels get modestly stronger suppression (hinglish_sup_boost) and weaker amplification (hinglish_amp_boost) to reduce code-mix leakage in shadow modes.

No API calls in hot loops—labels are precomputed offline via `tools/gemini_feature_labeler.py`.

## Safety and telemetry
- Norm clamping runs per position; `min_clamp_scale` avoids numerical issues.
- Deterministic generation (default) for reproducible evaluations.
- Optional telemetry can log per-layer delta L2 per prompt for diagnostics.

## Pooling details
- We pool hidden states per layer using an attention-mask aware combiner: 70% masked mean over valid tokens + 30% last valid token.
- If no mask is available (e.g., some generation paths), we fall back to unmasked mean + last position.

## How to run (examples)

Quick debug (CPU or small GPU):
```pwsh
python he_steer_pipeline.py --debug --mode shadow_hindi --feature-ranking stats --intervention-scope last
```

Label-aware ranking (with precomputed labels):
```pwsh
python he_steer_pipeline.py `
  --layers 19,20 `
  --mode shadow_hindi `
  --feature-ranking label `
  --use-feature-labels `
  --feature-labels-path he_pipeline_results/feature_labels.json `
  --top-k-per-layer 32 `
  --steer-alpha 2.2 `
  --norm-clamp-ratio 0.35 `
  --intervention-scope sequence
```

Causal shortlist for shadow mode (more compute):
```pwsh
python he_steer_pipeline.py `
  --layers 19,20 `
  --mode shadow_hindi `
  --feature-ranking causal `
  --top-k-per-layer 32 `
  --steer-alpha 2.0 `
  --norm-clamp-ratio 0.35
```

## Tips
- Prefer last-token scope for quick scans; switch to sequence for stronger effects when evaluating final settings.
- If outputs become brittle, reduce strength or increase clamp; also try lowering top-k.
- If code-mixing persists in shadow modes, enable `--use-feature-labels` and consider raising `hinglish_sup_boost`.

## Known pitfalls
- Sequence-wide interventions disable the KV cache and slow generation.
- Overly large max sequence length or dataset sizes can OOM—use the dynamic batch finder and `--debug` when iterating.
