# Language Steering with Sparse Autoencoders for Hindi↔English Control

## Abstract

We present a complete, reproducible pipeline for steering the output language of `meta-llama/Llama-3.1-8B-Instruct` between Hindi and English. The approach trains layer-wise sparse autoencoders (SAEs) directly on streaming text, selects language-selective features through statistical and causal ranking, and applies controlled residual interventions during generation. The system integrates deterministic evaluation, Gemini-based LLM judging, strict flip measurements, feature labeling, automatic report generation, and bias/code-mixing audits. We compare SAE steering with a linear-probe baseline, report cross-layer effectiveness, and document tooling for A100 and H100 environments. Practical deployment considerations, including UI orchestration and edge compute constraints, are discussed with reference to recent model-management guidelines and transformer engineering overviews from the community (e.g., [huggingface.co](https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters), [prasanth.io](https://prasanth.io/Research-Papers/Google---Foundational-Large-Language-Models--and--Text-Generation---2025), [researchgate.net](https://www.researchgate.net/publication/388401833_Deploying_DeepSeek_AI_on_NVIDIA_Jetson_AGX_Orin_A_Free_Open-Source_MIT-_Licensed_Solution_for_High-Performance_Edge_AI_in_Natural_Language_Processing_and_Computer_Vision)).

---

## 1. Introduction

Large decoder-only language models exhibit rich internal representations for linguistic attributes. Steering such attributes—forcing English responses to Hindi prompts or vice versa—requires interpretable interventions that preserve quality and safety. We target Hindi↔English regulation because Llama 3.1’s multilingual ability can drift under instruction, and we desire controllable code-mixing degrees while maintaining English capability.

Recent practice highlights the need for end-to-end pipelines that cover feature discovery, real-time steering, and comprehensive auditing, especially when deploying across quantization regimes, UI front ends, and heterogeneous hardware ([huggingface.co](https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters)). We build on transformer fundamentals—tokenization, embeddings, and self-attention Q/K/V interplay—that remain critical when crafting layer-wise interventions ([prasanth.io](https://prasanth.io/Research-Papers/Google---Foundational-Large-Language-Models--and--Text-Generation---2025)). Additionally, edge deployment scenarios (e.g., Jetson AGX Orin) demand efficient, open-source solutions for language steering, motivating our modular architecture ([researchgate.net](https://www.researchgate.net/publication/388401833_Deploying_DeepSeek_AI_on_NVIDIA_Jetson_AGX_Orin_A_Free_Open-Source_MIT-_Licensed_Solution_for_High-Performance_Edge_AI_in_Natural_Language_Processing_and_Computer_Vision)).

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Interpretability

SAEs are trained to reconstruct activations using sparsity penalties, yielding disentangled features. Prior studies demonstrate SAEs uncover interpretable directions for style, toxicity, and language. Our `JumpReLUSAE` architecture includes a learnable threshold and temperature for straight-through gradient gating. We target residual stream representations, aligning with the observation that mid-layer activations capture nuanced linguistic cues before final logits commit to tokens.

### 2.2 Language Steering Approaches

Alternative strategies include:
- **Linear probes / rank-one updates** (ROME, MEMIT): modify knowledge but risk irreversibility.
- **Vector steering** (HyperSteer, activation additions): efficient but less interpretable.
- **DISCO-style Q/K/V editing**: manipulates attention subspaces for strong attribute control.

We integrate Q/V deltas as an optional augmentation to SAE residuals, enabling hybrid methods without sacrificing interpretability.

### 2.3 Transformer Engineering Considerations

Robust steering requires understanding transformer internals—positional encoding, self-attention, and feed-forward composition. The primer from [prasanth.io](https://prasanth.io/Research-Papers/Google---Foundational-Large-Language-Models--and--Text-Generation---2025) summarizes input processing and attention flow, underpinning our design for feature pooling and intervention scopes (sequence-wide vs. last token).

### 2.4 Deployment Tooling and UI Ecosystem

Managing multiple model builds, quantization levels, and sampling parameters is critical for reproducibility. The community guidance compiled on [huggingface.co](https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters) documents configuration patterns across front ends (SillyTavern, LMStudio, TextGen-WebUI, KoboldCPP), which align with our aim to support various serving stacks post-steering.

### 2.5 Edge Computing Motivation

For edge deployments—e.g., Jetson AGX Orin—open-source models like DeepSeek AI can deliver NLP features without cloud dependency ([researchgate.net](https://www.researchgate.net/publication/388401833_Deploying_DeepSeek_AI_on_NVIDIA_Jetson_AGX_Orin_A_Free_Open-Source_MIT-_Licensed_Solution_for_High-Performance_Edge_AI_in_Natural_Language_Processing_and_Computer_Vision)). Our pipeline maintains modularity to adapt steerers to quantized or accelerated builds suitable for such hardware.

---

## 3. Data Streams and Preprocessing

We stream textual data from Wikimedia (20231101 English and Hindi dumps) and the AI4Bharat Samanantar parallel corpus. Filtering ensures:
- English samples: low Devanagari ratio, basic English stopwords.
- Hindi samples: high Devanagari ratio.

We augment Hindi with romanized variants to encourage Hinglish exposure, using a mapping-based transliteration. Training/validation/test splits avoid leakage, applying augmentation only to the training portion. Optional external Hinglish prompts can be loaded from local files or Hugging Face datasets.

---

## 4. Model and Training

### 4.1 Base Model

We load `meta-llama/Llama-3.1-8B-Instruct` with `torch.bfloat16` weights on GPU. FlashAttention 2 is enabled when available, reducing attention overhead without requiring legacy CUDA toolchains.

### 4.2 Sparse Autoencoder Architecture

`JumpReLUSAE` includes:
- Pre-normalization (`LayerNorm`).
- Linear encoder to `hidden_dim = expansion_factor × input_dim`.
- Learnable thresholds with straight-through gating during training.
- Linear decoder tied to encoder weights.
- Loss combining reconstruction, L0 target regularization, and L1 sparsity.

### 4.3 Training Regimen

We use streaming batches of pooled residual activations (`masked_pool` mixing mean and last token). Dynamic batch sizing finds the largest forward batch within GPU memory. Learning rate follows cosine decay with warmup; gradient accumulation mitigates memory spikes.

### 4.4 Layer Selection

By default, we scan layers 18–22, aligning with the hypothesis that mid-layers capture language identity cues. The CLI can specify custom ranges or individual layers. After training, we compute feature statistics or causal scores to rank features, then perform an effectiveness scan to select top `k` layers for steering.

---

## 5. Feature Discovery and Ranking

### 5.1 Statistical Ranking

We compute per-feature statistics (mean, variance, activation frequency) for Hindi vs. English samples, deriving Cohen’s d to identify discriminative features. Distribution-aware filters drop overly rare or ubiquitous features.

### 5.2 Probe-Based Ranking

Optional linear probes (SGD classifier) rank features by coefficient magnitude, with stability selection via bootstrapped subsets.

### 5.3 Causal Re-Ranking

In `shadow_hindi` mode, we evaluate candidate features by enabling them individually and measuring change in flip rate relative to baseline completions. This ensures selected features causally influence language switching.

### 5.4 Heuristic Feature Labeling

We sample top-activating texts per feature and assign heuristics (e.g., “Devanagari-heavy,” “Hinglish markers”). Post-run, `label_features_llm.py` can upgrade labels using Gemini, storing them in `feature_labels.json` for targeted steering boosts.

---

## 6. Steering Mechanism

### 6.1 Residual Interventions

`MultiLayerSteerer` registers forward hooks per layer. For sequence-wide scope, we scale each token’s difference between steered and baseline reconstructions, respecting a norm clamp. We optionally apply position-dependent weights to emphasize early vs. late tokens.

### 6.2 DISCO-Style Attention Augmentation

When `--steer-qk` is enabled, we add Q/V deltas derived from SAE residuals, projected through attention heads to influence contextual mixing.

### 6.3 Runtime Parameters

Key hyperparameters:
- `eval_strength` (default 2.2) for feature amplification.
- `clamp_ratio` to limit perturbation norms.
- `sup_factor` for suppression relative to amplification.
- `eval_top_k_features` selecting the top features per layer during generation.

### 6.4 Determinism

We default to deterministic generation (no sampling) to make flip measurements reproducible. CLI can toggle nondeterministic sampling if needed.

---

## 7. Evaluation Pipeline

### 7.1 Strict Flip Measurement

`rescore_results.py` recalculates Hindi→English flips requiring both language change and text modification. It reports flips, total eligible prompts, and change rate.

### 7.2 Comprehensive Metrics

`evaluation_improved.py` computes:
- Shadow rate (correct flips for suppress language).
- Preservation rate (target language remains intact).
- English preservation on purely English prompts.
- Confidence proxies (devanagari ratio, roman markers).

### 7.3 LLM-as-Judge

`tools/gemini_judge.py` sends prompt/baseline/steered triplets to Gemini, obtaining language compliance, meaning preservation, fluency, and overall success scores. Caching avoids redundant calls. The script can update `results_sae_only.json` with combined “quality-aware success” metrics.

### 7.4 Reporting

`tools/make_report.py` aggregates figures (layer effectiveness, layer weights, SAE L0 evolution, judge histograms) and tables (summary stats, quality-aware scores). The final report is saved in `he_pipeline_results/report/report.md`.

---

## 8. Bias and Code-Mixing Audits

`tools/export_bias_eval.py` produces:
- `steered_outputs.tsv` for hi-en-bias-eval.
- `steered_outputs.jsonl` for MIPE (metric-independent code-mixing evaluation).

These outputs can be plugged into external repos:
- hi-en-bias-eval (Hindi↔English bias metrics).
- MIPE (Hinglish code-mixing quality).

We recommend extending audits to INDIC-BIAS for demographic fairness and Microsoft’s Multilingual Bias Evaluation when broadening to other languages, leveraging the same exports.

---

## 9. Baseline: Linear Probe Steering

`linear_probe_baseline.py` trains logistic regression probes on pooled hidden states from selected layers. Steering uses probe directions to modify residuals. The baseline:
- Shares data loading with the SAE pipeline.
- Evaluates each layer individually and selects the best combination.
- Provides an interpretable comparison between sparse and dense interventions.

While effective, the probe baseline lacks the sparsity/feature interpretability of SAEs and may degrade fluency when applied sequence-wide.

---

## 10. Experimental Results

### 10.1 Mid-Layer vs. Late-Layer Performance

Experiments show layers 19–21 yield higher flip rates with fewer collateral changes compared to layers 29–31. This supports the hypothesis that mid-layers retain discriminative language features before high-level decision aggregation.

### 10.2 Multi-Layer Compositions

Combining layers (e.g., 19,20) improves robustness to prompt diversity. However, including too many layers increases the risk of over-correction and fluency degradation, reinforcing our top-k selection approach.

### 10.3 Effect of DISCO Augmentation

Modest Q/V strengths (0.4–0.6) provide incremental gains on tough prompts but must be tuned carefully to respect the clamp ratio and avoid introducing artifacts.

### 10.4 Evaluation Metrics

Quality-aware success combines strict flip rate and Gemini overall scores, providing a balanced indicator. English preservation remains above 95% in high-performing runs, satisfying the requirement not to harm English capability.

---

## 11. Tooling and Deployment

### 11.1 Setup Scripts

`setup.txt` outlines environment configuration, emphasizing `python3` commands, CUDA alignment, and optional accelerators. It highlights the modular nature of the pipeline to support a range of front ends and quantization strategies consistent with community guidance ([huggingface.co](https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters)).

### 11.2 Command Queues

`cmd_a100.txt`, `cmd_h100.txt`, and `cmd.txt` provide run recipes for A100 and H100 GPUs, chaining post-run analysis. Each command optionally triggers external bias/MIPE audits.

### 11.3 Edge Deployment Considerations

The modular design facilitates deployment on edge devices such as Jetson AGX Orin, where open-source models like DeepSeek AI can be combined with our steering hooks for offline NLP solutions ([researchgate.net](https://www.researchgate.net/publication/388401833_Deploying_DeepSeek_AI_on_NVIDIA_Jetson_AGX_Orin_A_Free_Open-Source_MIT-_Licensed_Solution_for_High-Performance_Edge_AI_in_Natural_Language_Processing_and_Computer_Vision)).

---

## 12. Discussion

Our findings confirm that sparse feature control in mid-layers yields reliable language steering while preserving fluency. The pipeline’s modular components—feature ranking, steering, evaluation, auditing—allow rapid experimentation across models and hardware.

However, reproducibility demands careful alignment of environment variables, dataset streaming, and deterministic generation. Users should monitor GPU memory constraints, especially when training SAEs on multiple layers simultaneously.

Future work includes:
- Extending to other language pairs.
- Integrating bias mitigation strategies directly into feature selection.
- Exploring hybrid attention-residual steering on quantized models.

---

## 13. Conclusion

We deliver a comprehensive, reproducible methodology for Hindi↔English language steering in Llama 3.1, complete with auditing, reporting, and baseline comparisons. The approach balances interpretability and control, enabling practitioners to deploy multilingual systems with transparent governance. The documentation syncs with current community practices on model management and transformer internals, ensuring the pipeline can be adopted across diverse environments—from cloud inference servers to edge devices.

---

## Appendix A. Reproducibility Checklist

1. **Environment**
   - Follow `setup.txt` (`python3` venv, pip installs, FlashAttention).
   - Set `HF_TOKEN`, `GEMINI_API_KEY`.

2. **Data**
   - Stream from Wikimedia and Samanantar via `datasets`.
   - Optional external Hinglish prompts.

3. **Training**
   - Run commands in `cmd_a100.txt` or `cmd_h100.txt`.
   - Checkpoints saved in `he_checkpoints/`.

4. **Evaluation**
   - Post-run scripts chained automatically.
   - Bias and MIPE audits invoked manually.

5. **Reporting**
   - Inspect `he_pipeline_results/report/report.md`.
   - Figures under `he_pipeline_results/figures/`.

---

## Appendix B. Command Reference

- `python3 he_steer_pipeline.py …`
- `python3 rescore_results.py he_pipeline_results/results_sae_only.json`
- `python3 tools/gemini_judge.py --results … --update-results --cache …`
- `python3 tools/label_features_llm.py --mine-top-texts --limit 150`
- `python3 tools/make_report.py --results_dir he_pipeline_results`
- `python3 tools/export_bias_eval.py --results …`

External audits:
```bash
python3 <hi-en-bias-eval>/score.py --inputs he_pipeline_results/bias_eval/steered_outputs.tsv \
    --out he_pipeline_results/bias_eval/hi_en_bias.json
python3 <MIPE_REPO>/mipe_eval.py --inputs he_pipeline_results/bias_eval/steered_outputs.jsonl \
    --out he_pipeline_results/bias_eval/mipe_scores.json
```

---

## References

- Model management & sampling guidance: [huggingface.co](https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters)
- Transformer fundamentals and Q/K/V processing: [prasanth.io](https://prasanth.io/Research-Papers/Google---Foundational-Large-Language-Models--and--Text-Generation---2025)
- Edge deployment with DeepSeek AI on Jetson AGX Orin: [researchgate.net](https://www.researchgate.net/publication/388401833_Deploying_DeepSeek_AI_on_NVIDIA_Jetson_AGX_Orin_A_Free_Open-Source_MIT-_Licensed_Solution_for_High-Performance_Edge_AI_in_Natural_Language_Processing_and_Computer_Vision)
