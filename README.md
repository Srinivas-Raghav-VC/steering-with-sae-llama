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
python he_steer_pipeline.py --debug

# Full research run (scans all layers, selects top 3)
python he_steer_pipeline.py

# Custom configuration
python he_steer_pipeline.py --scan_all_layers --top_k_layers 5 --bonferroni --min_effect_size 0.5
```

##  System Requirements

- **GPU**: A100 (80GB)
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ free space
- **Python**: 3.9+

Really cool Inference from Prime Intellect Guys!


