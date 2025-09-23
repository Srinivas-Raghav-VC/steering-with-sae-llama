# AI agent guide for this repo

Purpose: This project trains Sparse Autoencoders (SAEs) on Llama‑3.1‑8B hidden states to steer language (Hindi ↔ English) by manipulating SAE features, then evaluates and visualizes results.

Big picture architecture
- Entry point: `he_steer_pipeline.py` orchestrates everything (data → SAEs → feature stats → steering hooks → evaluation → results JSON).
- SAEs: `JumpReLUSAE` (tied weights, learned thresholds) trained per layer on pooled representations; reconstruction + soft-L0 + L1 losses; online/streaming extraction.
- Steering: For selected layers, encode→edit features→decode delta→norm‑clamped residual injection (Goodfire‑style). Sequence‑wide or last‑token scopes.
- Baselines/analysis: `linear_probe_baseline.py` (probe‑direction steering), `rescore_results.py`, `evaluation_improved.py`, figures via `make_figures.py`, report via `tools/make_report.py`.
 - Optional LLM‑as‑judge: `tools/gemini_judge.py` scores (language compliance, meaning preservation, fluency, overall) post‑hoc over `results_sae_only.json`.

Required environment and deps
- Model: `meta-llama/Llama-3.1-8B-Instruct` (gated). Set `HF_TOKEN` in env to login; A100/large GPU strongly recommended; bfloat16 inference.
- Data: Hugging Face streaming datasets (Wikipedia EN/HI + Samanantar). No large local caches.
- Accel: Optional FlashAttention 2 (`flash-attn>=2`); code auto‑enables if present. See `requirements.txt` for full list.

How to run (PowerShell examples)
- Quick sanity (small, faster):
```pwsh
python he_steer_pipeline.py --debug
```
- Full mid‑layer run (example from `cmd.txt`):
```pwsh
python he_steer_pipeline.py --layers 19,20 --train-epochs 120 --samples-per-language 12000 --max-seq-length 512 --mode shadow_hindi --steer-alpha 2.2 --norm-clamp-ratio 0.35 --intervention-scope sequence --top-k-per-layer 32 --eval-prompts 64
```
- Re‑score strict flips:
```pwsh
python rescore_results.py he_pipeline_results/results_sae_only.json
```
- Make figures / report:
```pwsh
python make_figures.py --inputs "**/results_sae_only.json" --out figs
python tools/make_report.py --results_dir he_pipeline_results
```
 - LLM‑as‑judge (Gemini) on results:
```pwsh
$env:GEMINI_API_KEY = "<your_key>"
python tools/gemini_judge.py --results he_pipeline_results/results_sae_only.json --out he_pipeline_results/llm_judge_gemini.json --model gemini-1.5-flash --limit 64
```

Core patterns to follow
- Representation pooling: see `masked_pool` and `pooled_from_layer_batch` (hook `model.model.layers[L]`). Keep model in bfloat16; convert pooled tensors to fp32 for SAE.
- Dynamic batch sizing: use `find_max_batch_size` before extraction to avoid OOM; catches CUDA OOM and backs off.
- Feature discovery: `feature_stats_streaming` computes per‑feature means/vars/freqs and Cohen’s d for HI vs EN; top features via `pick_top_features` (effect size + selectivity + magnitude).
- Hooking for steering: `MultiLayerSteerer._hook_sae` registers forward hooks per selected layer; sequence‑wide scope disables KV cache (`use_cache=False`)—expect slower gen.
- Results format: default path `he_pipeline_results/results_sae_only.json` with `config`, `effectiveness` per layer, `selected_layers`, `layer_weights`, and `results` records. Keys for layers may appear as strings—normalize when reading.

Conventions and integration points
- Config: `Config` dataclass centralizes knobs (layers, strengths, clamp, scope, data sizes). Respect CLI overrides; prefer adding new options via `argparse` → `Config`.
- Language detection: `detect_language_simple` (Devanagari ratio + optional `langid`). If integrating an LLM‑as‑judge, add a post‑hoc scorer; don’t place API calls in training/extraction loops.
 - LLM‑as‑judge integration point: consume `he_pipeline_results/results_sae_only.json`, write a sibling JSON (e.g., `llm_judge_gemini.json`) with per‑record scores and a `summary`. Keep it optional and off the training path.
- Data: Modify `_is_quality_english/_is_quality_hindi` and `romanize_hindi_basic` for other languages; keep augmentation confined to training split.
- Safety/perf: Keep norm clamping (`clamp_ratio`) in hooks; ensure deltas are scaled per‑position; avoid changing model dtype/device.

Common pitfalls
- No `HF_TOKEN` → gated model won’t load; script prints a warning. Provide token via env before runs.
- Sequence‑wide interventions disable cache, increasing latency; prefer `--intervention-scope last` for quick scans.
- Large `--max-seq-length` or `--samples-per-language` can OOM; rely on dynamic batch finder and reduce sizes in `--debug` mode.
 - LLM judge cost/latency: Use `--limit` to sub‑sample; cache results to avoid repeated API calls.

If you add features
- New metrics: write into the final results JSON under stable keys; `make_figures.py` and `tools/make_report.py` expect `effectiveness`, `selected_layers`, and `results` arrays.
- New evaluation: extend `evaluation_improved.py` or add a separate scorer; don’t change the strict rescoring contract unless you update consumers.

Open questions to confirm
- Preferred evaluator: keep heuristic scripts or integrate LLM‑as‑judge (Gemini/GPT) for semantic/fluency scoring?
- Any additional target language pairs beyond HI↔EN to wire into `load_text_pairs`?
