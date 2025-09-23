# Neural Language Steering with Sparse Autoencoders (SAEs) in Llama‑3.1‑8B

## Abstract

We study Hindi–English language steering in a decoder-only LLM (Llama‑3.1‑8B
Instruct) using sparse autoencoders (SAEs) to identify and manipulate language-
selective features. We correct a common methodology error: picking layers by
representation strength (e.g., probe accuracy) instead of intervention
effectiveness. We train JumpReLU SAEs per layer (0‑indexed) in candidate bands
(18–22 and optionally 29–31), discover sparse features separating Hindi vs
English, and steer by amplifying/suppressing these features via decode-to-
residual deltas with norm clamp. We evaluate with a strict flip metric:
Hindi-like baselines must become English-like under shadow_hindi. Results show
that mid layers (19–21) typically outperform late layers for steering, despite
the latter having stronger language encoding. We additionally integrate a
causal feature ranking step (measured flip deltas) and practical controls for
determinism and safety. We release code to reproduce the pipeline and figures.

Keywords: neural steering, sparse autoencoders, interpretability, multilingual,
autoreg generation

## 1. Introduction

A frequent mistake in neural steering is to choose layers by where language is
most encoded (late layers), not where intervention changes behavior. We target
language shadowing (forcing English output from Hindi prompts) and show that
intermediate layers are generally better intervention points.

Contributions:
- Corrected evaluation: measure actual flips, not probe accuracy.
- SAE-only steering: JumpReLU SAEs, feature scaling, residual deltas (Goodfire-
  style), with norm clamp.
- Strict metric: counts Hindi→English only when baseline is Hindi-like and text
  changes.
- Open pipeline: robust data loading, dynamic batch sizing, sequence-wide
  intervention option, distribution-aware filtering, optional causal ranking,
  and scripts to reproduce figures.

## 2. Related Work

- Transformers & layer specialization: Tenney et al. (2019); Vig & Belinkov
  (2019); Rogers et al. (2020).
- Editing/intervention: Meng et al. (2022, 2023); Anthropic activation patching
  and concept interventions.
- Sparse autoencoders & interpretability: Olshausen & Field (1996); Anthropic
  (2024) “Scaling Monosemanticity”; Cunningham et al. (2023).

## 3. Methods

### 3.1 Data

- English: Wikipedia (20231101.en) + Samanantar src (streaming).
- Hindi: Wikipedia (20231101.hi) + Samanantar tgt (streaming).
- Quality filters ensure language balance; add romanized/Hinglish augmentation
  to avoid trivial script-only cues. Samples per language: 8–12k.

### 3.2 SAEs

- JumpReLU SAE per layer \(L\): encode pooled hidden states \(h\in\mathbb{R}^D\)
  to sparse \(z\in\mathbb{R}^{D\cdot f}\), decode back with tied weights.
- Training is online/streaming: pool on the fly, no giant tensors saved.
- Targets: expansion factor 16, \(L_0\approx 64\), 120 epochs on A100‑80GB.

### 3.3 Feature discovery and ranking

- Streaming stats (default). Compute per-feature mean/variance/frequency over
  HI vs EN SAE codes and rank by
  \(|d| + 0.3\,|f_{hi}-f_{en}| + 0.1(|\mu_{hi}|+|\mu_{en}|)\), where \(d\) is
  Cohen’s \(d\) (Welch-style). Split top‑K by sign(\(d\)).
- Linear probe ("probe"). Fit a tiny SGD logistic classifier on SAE codes and
  rank by weight magnitude (sign gives class). Useful when effect sizes alone
  are noisy.
- Causal ("causal"). Start from a stats shortlist, then re‑rank features by
  their measured marginal impact on flip rate on a small validation set (one
  feature steered at a time under the current mode). This better captures
  control effectiveness than pure correlation.
- Optional stability selection. Bootstrap resample a validation slice and keep
  features that persist across samples.

Distribution-aware shortlist (optional). After selecting by stats (or before
causal re‑ranking), drop features whose activation frequency is too rare or too
common in the intended class using configurable thresholds; this trims brittle
or overly generic signals.

### 3.4 Steering (Goodfire-style residual deltas)

For each hooked layer:
1) Take block output \(h\) (residual stream).
2) Encode \(z\), modify features: amplify target lang, suppress other lang.
3) Decode \(x'=\mathrm{dec}(z')\), form \(\Delta = x' - \mathrm{dec}(z)\).
4) Norm clamp per position, \(\|\Delta\|\le \rho\|h\|\), then add \(h\leftarrow
   h + w_L\Delta\).
5) Apply either on last token or sequence-wide with positional weights.

Optional label bias. Post‑hoc feature labels (heuristic or LLM‑derived) can
scale suppression/amplification (e.g., boost suppression for Hinglish‑tagged
features) to reduce collateral edits.

### 3.5 Evaluation (strict)

- Mode: shadow_hindi (suppress Hindi features only).
- Success counted iff:
  - Baseline is Hindi-like (Devanagari or romanized-Hindi markers),
  - Steered is English-like (low Devanagari, low markers),
  - and steered text differs from baseline.
- Report per-layer flip rate and selected multi-layer blend.

Quality‑aware success (optional). An LLM‑as‑judge (Gemini) scores language
compliance, meaning preservation, and fluency for generated pairs; a combined
quality‑aware success metric is written back into results for analysis.

### 3.6 Determinism, safety, and efficiency

- Determinism: generation can be fixed or sampled; by default we use
  deterministic decoding for comparable flips, with a flag to allow variability
  during exploratory sweeps.
- Safety rails: per-position norm clamping plus a telemetry guard that blocks
  pathological tiny scales (zeroing deltas) to avoid instability.
- Efficiency: sequence‑wide edits are vectorized with positional weighting;
  this disables KV cache in hooked layers, so scans can use last‑token mode for
  speed. FlashAttention 2 is auto‑enabled if installed. GPU presets are
  available for A100/H100.

### 3.7 Reporting and artifacts (optional)

- Results JSON is enriched with per-feature `causal_scores` (Δ flip‑rate vs.
  baseline), `data_balance`, and `lineage`.
- The reporting script (`tools/make_report.py`) generates:
  - Causal delta plots (top ± features per layer/side) when `causal_scores` are
    present.
  - LLM‑as‑judge plots (Gemini summary and per‑record distributions) when
    `llm_judge_gemini.json` exists.
  - Core discovery/selection/telemetry figures as before.

## 4. Experiments

- Model: meta‑llama/Llama‑3.1‑8B‑Instruct.
- Layers: 0‑indexed mid band 18–22 (primary), optionally late 29–31.
- Sequences: 512 tokens; dynamic batch sizing; FlashAttention if available.
- Steering: sequence-wide for final runs; last-token for quick scans.
- Strength: 2.0–2.5; clamp 0.35–0.45; K=32–64 features per layer.
- GPU presets: optional flags tune extraction and batch sizes for A100/H100.
- Determinism: default on for evaluation; tuning can optionally relax this.
- Hardware: Experiments run on A100‑80GB (primary). H100‑80GB offers extra
  headroom (optional). Future H100‑oriented optimizations (not required here)
  include CUDA Graphs and deeper fused kernels (e.g., fused QKV) alongside
  FlashAttention 2.

## 5. Results (typical)

- Mid layers (19–21) achieve higher flip rates than late layers, even though
  late layers show stronger language encoding with probes.
- 1–2 layers are often sufficient; adding a third yields diminishing returns.
- Sequence-wide > last-token for steering effectiveness.

Figures:
- F1: Flip rate by layer (strict metric).
- F2: Mid vs late comparison with CIs.
- F3: Single vs multi-layer ablation.
- F4: Sequence-wide vs last-token effect.
- F5: Strength/clamp sweep curves.
- F6: Feature-type counts (heuristic labels).
- F7 (optional): Causal per-feature deltas for top candidates.
- F8 (optional): LLM‑judge (Gemini) summary and distribution plots.

## 6. Discussion

- Representation strength ≠ controllability: late layers encode language
  strongly because the decision is committed; mid layers are where interventions
  still change outcomes.
- SAE features give interpretable handles (script/morphology/syntax). Shadowing
  relies more on suppression of opposite-language features than brute
  amplification of target-language features.

## 7. Limitations

- Language pair studied: HI–EN. Results may differ in other language families.
- Architecture-specific: layer numbers apply to Llama‑3.1‑8B.
- Runtime: sequence-wide hooks disable KV cache; slower but necessary for early
  commitment.
 - Causal ranking budget: only a small candidate set is re‑ranked to keep
   compute tractable; deltas are cached for analysis.

## 8. Conclusion

SAE-only residual steering with strict evaluation shows that mid layers are
generally the right intervention points for language shadowing. This corrects
the “encoding-strength” fallacy and provides a reproducible recipe for
multilingual control.

## 9. Reproducibility

- Code: SAE-only pipeline (`he_steer_pipeline.py`), figures (`make_figures.py`),
  report tooling (`tools/make_report.py`), optional LLM judge
  (`tools/gemini_judge.py`).
- Quick sanity run:
  - `python he_steer_pipeline.py --debug`
- Full example (mid layers, sequence-wide, modest eval):
  - `python he_steer_pipeline.py --layers 19,20 --train-epochs 120 --samples-per-language 12000 --max-seq-length 512 --mode shadow_hindi --steer-alpha 2.2 --norm-clamp-ratio 0.35 --intervention-scope sequence --top-k-per-layer 32 --eval-prompts 64`
- Re‑score strict flips:
  - `python rescore_results.py he_pipeline_results/results_sae_only.json`
- Make figures / report:
  - `python make_figures.py --inputs "**/results_sae_only.json" --out figs`
  - `python tools/make_report.py --results_dir he_pipeline_results`
- Results JSON includes: `config`, `effectiveness`, `selected_layers`,
  `layer_weights`, `results`, plus optional `causal_scores` (per‑feature flip
  deltas), `data_balance` (script buckets), and `lineage` (run metadata).
  When present, `tools/make_report.py` also renders LLM‑judge plots from
  `llm_judge_gemini.json`.

### Companion walkthrough

For a systems-first, end-to-end rationale of every component (data → SAEs → feature selection → steering → evaluation → Gemini tooling) with practical tips and citations, see `SYSTEMS_FIRST_PRINCIPLES_WALKTHROUGH.md` in the repo root.

### Roadmap

For prioritized next steps to make the stack best-in-class (representation/steering upgrades, data coverage, evaluation/labeling, efficiency, and guardrails), see `ROADMAP.md` in the repo root.

### Recent additions (practical)

- External Hinglish prompts (val/test only): optional loader for small Hinglish
  samples via local files or HF datasets; used only to enrich evaluation
  prompts without touching training splits.
- Quality-aware scoring: LLM-as-judge (Gemini) scores (meaning/fluency/
  compliance) merged back into `results_sae_only.json` as a combined
  `quality_aware_success` to supplement strict flips.
- Tuning with suppression factor: quick grid and a small random local search
  (flip+perplexity, optional BLEU) over strength, clamp, K, and a separate
  suppression factor to avoid over-editing.
- LLM feature labeling: a post-hoc tool `tools/label_features_llm.py` builds
  `feature_labels.json` that can bias suppression/amplification (e.g.,
  Hinglish‑specific feature handling) during steering.
- Causal feature ranking (optional): `--feature-ranking causal` re‑orders
  candidate features by measured flip‑rate deltas from single‑feature steering
  on a small validation set, improving robustness of control signals.
 - Determinism & safety rails: deterministic decoding for comparability,
   optional nondeterministic sweeps; telemetry guard prevents unstable edits.
 - Distribution-aware filtering: drop brittle/ubiquitous features by frequency
   thresholds before causal re‑ranking.
 - Data balance & lineage: script-bucket counts and run metadata logged to the
   results JSON for downstream analysis.
 - GPU presets: convenience flags for A100/H100 to set extraction and batch
   sizes; FlashAttention auto‑enabled when available.

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
- Tenney, I., et al. (2019). What does BERT look at? BlackboxNLP.
- Vig, J., & Belinkov, Y. (2019). Attention analysis. BlackboxNLP.
- Rogers, A., et al. (2020). A primer in NN interpretability. CACM.
- Meng, K., et al. (2022). ROME. NeurIPS.
- Meng, K., et al. (2023). MEMIT. ICLR.
- Olshausen, B., & Field, D. (1996). Sparse coding. Nature.
- Anthropic (2024). Scaling Monosemanticity (SAE interpretability).
- Cunningham, H., et al. (2023). Sparse autoencoders find interpretable directions. ICLR.
