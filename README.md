# Hindi-English Language Steering Pipeline

A  pipeline for multi-layer language steering on Llama-3.1-8B, enabling fine-grained control over Hindi and English text generation through sparse autoencoder (SAE) feature manipulation.

###  Tools Used :
- WinSCP for Easier File Transfer
- OpenSSH for Windows Terminal

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required: Hugging Face token for model access
export HF_TOKEN="your_huggingface_token_here"

# Optional: Gemini API key for feature interpretation
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: CUDA device selection
export CUDA_VISIBLE_DEVICES=0
```

### 3. Run the Pipeline

```bash
# Debug run (quick test with 2 layers)
python3 he_steer_pipeline.py --debug

# Full research-oriented runs are provided in cmd_a100.txt / cmd_h100.txt.
# Common examples:

# Scan a mid-band range and pick top features causally
python3 he_steer_pipeline.py --test-layer-ranges 18:23 --top-k-per-layer 32 --feature-ranking causal

# Fixed layers (19,20) with causal re-ranking, sequence scope
python3 he_steer_pipeline.py --layers 19,20 --feature-ranking causal --intervention-scope sequence --steer-alpha 2.2 --norm-clamp-ratio 0.35 --eval-prompts 64
```

##  System Requirements

- **GPU**: A100 (80GB)
- **OS**(Optional): Ubuntu CUDA 12.xx
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ free space
- **Python**: 3.9+

Really cool Inference from Prime Intellect Guys!


## Detector calibration (optional)

You can evaluate and calibrate the shared language detector used across the pipeline.

Usage (PowerShell):

```pwsh
python tools/calibrate_detector.py --input data/labeled_hi_en.jsonl --limit 5000 --dataset-name hi_en_eval
```

Options:
- --input: Path to labeled data (.jsonl or .csv). Text field is auto-detected from text/sentence/input; label from label/lang/gold.
- --text-field / --label-field: Override column names explicitly if needed.
- --limit: Evaluate a subset for speed (0 = all).
- --dataset-name: Freeform name for the dataset, recorded in the output.
- --no-gemini: Force-disable Gemini fallback during evaluation.
- --compare-heuristic: Also run a second pass with Gemini disabled to compare heuristic-only performance.

Outputs:
- he_pipeline_results/detector_calibration.json: accuracy, macro-F1, per-label metrics, and metadata (plus optional heuristic-only comparison).
- he_pipeline_results/figures/detector_confusion.png: row-normalized confusion heatmap.

Environment gating:
- Set $env:DISABLE_GEMINI_DETECTOR = "1" to globally disable Gemini fallback in the detector utilities.
- When GEMINI_API_KEY is not set, Gemini is automatically disabled.

Cache note:
- The detector utilities may cache language predictions to he_pipeline_results/detector_cache.json.
- Delete that file if you recalibrate for a different domain to avoid stale cache effects.


