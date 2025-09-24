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

# Full research run (scans all layers, selects top 3)
python3 he_steer_pipeline.py

# Custom configuration
python3 he_steer_pipeline.py --scan_all_layers --top_k_layers 5 --bonferroni --min_effect_size 0.5
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


