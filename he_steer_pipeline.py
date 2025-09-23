#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# he_steer_pipeline.py — SAE-only pipeline for Hindi–English steering
#
# What this script does:
# - Loads Wikipedia + Samanantar (streaming) for EN/HI
# - Trains JumpReLU SAEs per requested layer (online, no giant caches)
# - Discovers sparse features separating HI vs EN (streaming stats)
# - Labels features heuristically (no external API)
# - Steers generation by amplifying/suppressing SAE features
# - Evaluates shadowing (e.g., force English from Hindi prompts)
#
# Key choices:
# - SAEs only (no linear fallback)
# - Token-level (sequence-wide) intervention option
# - Dynamic batch sizing to actually use A100 memory
#
# Author: You, with assist
# Date: 2025-09

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import SGDClassifier
import json as _json
from pathlib import Path as _Path

# Optional high-performance attention kernels (auto-register fused ops if installed)
try:
    import lookahead_keys_attention  # noqa: F401
except Exception:
    lookahead_keys_attention = None  # type: ignore

try:
    import liger_kernel  # noqa: F401
except Exception:
    liger_kernel = None  # type: ignore

# Optional pretrained SAE loader (EleutherAI)
try:
    from eai_sparsify import load_sae  # type: ignore
except Exception:
    load_sae = None  # type: ignore

# -----------------------------
# Simple language/script utils
# -----------------------------

_DEV_RE = re.compile(r"[\u0900-\u097F]")


def devanagari_ratio(s: str) -> float:
    if not s:
        return 0.0
    dev = sum(1 for ch in s if _DEV_RE.match(ch))
    alpha = sum(1 for ch in s if ch.isalpha())
    if alpha == 0:
        return 0.0
    return dev / float(alpha)


def english_prob(s: str) -> float:
    try:
        from langid import classify  # type: ignore

        lang, conf = classify(s)
        return conf if lang == "en" else 1.0 - conf
    except Exception:
        r = devanagari_ratio(s)
        return float(max(0.0, min(1.0, 1.0 - r)))


def detect_language_simple(s: str) -> str:
    if devanagari_ratio(s) > 0.25:
        return "hindi"
    return "english" if english_prob(s) >= 0.8 else "unknown"


# -----------------------------
# Config
# -----------------------------


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = device_str()

    # Data
    samples_per_language: int = 10000
    max_sequence_length: int = 512
    min_sequence_length: int = 24
    romanized_hindi_ratio: float = 0.25

    # Layers to train/eval
    layer_range: Tuple[int, int] = (18, 23)
    test_layer_ranges: Optional[str] = None  # e.g. "18:23,29:32"
    top_k_layers: int = 3  # when using ranges, pick best by effectiveness

    # SAE
    sae_expansion_factor: int = 16
    sae_l0_target: int = 64
    train_epochs: int = 120
    sae_batch_size: int = 1024  # per SAE step (features batch)
    grad_accum_steps: int = 2
    lr: float = 1e-3
    warmup_steps: int = 200
    ckpt_dir: str = "he_checkpoints"

    # Steering
    default_strength: float = 1.8
    clamp_ratio: float = 0.35
    sup_factor: float = 1.0  # scales suppression relative to strength
    eval_mode: str = "shadow_hindi"  # "steer" | "shadow_hindi" | "shadow_english"
    eval_strength: float = 2.2
    eval_top_k_features: int = 32
    intervention_scope: str = "sequence"  # "sequence" | "last"
    pos_weight_start: float = 1.0
    pos_weight_end: float = 0.6

    # Eval prompts
    eval_prompts: int = 24

    # Performance
    start_extract_bs: int = 64
    max_extract_bs: int = 4096
    use_flash_attn: bool = True

    # Paths
    out_dir: str = "he_pipeline_results"

    # Debug
    debug: bool = False
    # Determinism (controls generation sampling)
    deterministic: bool = True
    # Optional quick sweep to tune steering params on a tiny validation subset
    tune_steering: bool = False
    # Optional bayesian-like tuning (random local search)
    tune_bayes: bool = False
    tune_steps: int = 12
    # Metric weights for tuning composite score
    w_flip: float = 0.7
    w_ppl: float = 0.3
    w_bleu: float = 0.0  # requires sacrebleu; default off
    # Hinglish mining/datasets
    use_hinglish_mining: bool = True
    hinglish_ratio_train: float = 0.15
    hinglish_eval_prompts: int = 12
    # External Hinglish prompts (val/test only)
    use_external_hinglish: bool = False
    external_hinglish_file: Optional[str] = None  # .txt (one per line), .jsonl/.json, or .csv
    external_hinglish_hf_spec: Optional[str] = None  # e.g., "microsoft/GLUECoS"
    external_hinglish_split: Optional[str] = None  # e.g., "train" or task split
    external_hinglish_text_field: Optional[str] = None  # name of text column if known
    external_hinglish_max: int = 500
    # Feature ranking method: "stats" (Cohen's d) or "probe"
    feature_ranking: str = "stats"
    # Stability selection over validation bootstraps (0 = off)
    stability_bootstrap: int = 0
    stability_frac: float = 0.6
    # Optional LLM feature labels to bias suppression/amplification
    use_feature_labels: bool = False
    feature_labels_path: str = "he_pipeline_results/feature_labels.json"
    hinglish_sup_boost: float = 1.2
    hinglish_amp_boost: float = 0.8
    # Training/Eval controls
    early_stop_patience: int = 0  # 0 = off
    ppl_over_continuation: bool = True
    # Telemetry & safety rails
    telemetry: bool = False
    min_clamp_scale: float = 1e-4
    # Distribution-aware shortlist thresholds (activation frequency in [0,1])
    dist_min_freq: Optional[float] = None
    dist_max_freq: Optional[float] = None

    # Optional: DISCO-style attention steering
    steer_qk: bool = False
    steer_qv_strength: float = 0.6

    # Optional: Use pretrained SAEs (EleutherAI) instead of training
    use_pretrained_sae: bool = False

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Environment & model
# -----------------------------


def setup_environment():
    print("🔧 Environment setup...")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=True)
            print("Hugging Face login OK")
        except Exception as e:
            print(f"HF login error: {e}")
    else:
        print("❌ HF_TOKEN not set (gated repos may be inaccessible)")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
        dev = torch.cuda.get_device_properties(0)
        total = dev.total_memory / (1024**3)
        print(f"CUDA: {dev.name} | VRAM: {total:.1f} GB")
    else:
        print("⚠️  CUDA not available; CPU fallback (slow).")


def load_model_and_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Use bf16 on GPU, fall back to fp32 on CPU for compatibility
    kwargs = dict(torch_dtype=(torch.bfloat16 if cfg.device != "cpu" else torch.float32), low_cpu_mem_usage=True)
    if cfg.use_flash_attn and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401  # type: ignore

            kwargs["attn_implementation"] = "flash_attention_2"
            print("✅ Flash Attention 2 available")
        except Exception:
            print("⚠️  Flash Attention not available; using standard attention")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.to(cfg.device)
    print(f"Model dtype: {next(model.parameters()).dtype}")
    return model, tok


# -----------------------------
# Datasets (streaming)
# -----------------------------


def _is_quality_english(txt: str, cfg: Config) -> bool:
    if not txt:
        return False
    n = len(txt.split())
    if n < cfg.min_sequence_length or n > cfg.max_sequence_length:
        return False
    if devanagari_ratio(txt) > 0.1:
        return False
    en_markers = {"the", "and", "is", "are", "this", "that", "in", "on", "at"}
    cnt = sum(1 for w in txt.split()[:64] if w.lower() in en_markers)
    return cnt >= 1


def _is_quality_hindi(txt: str, cfg: Config) -> bool:
    if not txt:
        return False
    n = len(txt.split())
    if n < cfg.min_sequence_length or n > cfg.max_sequence_length:
        return False
    return devanagari_ratio(txt) >= 0.3


def romanize_hindi_basic(text: str) -> str:
    mapping = {
        "अ": "a",
        "आ": "aa",
        "इ": "i",
        "ई": "ee",
        "उ": "u",
        "ऊ": "oo",
        "ए": "e",
        "ऐ": "ai",
        "ओ": "o",
        "औ": "au",
        "ा": "a",
        "ि": "i",
        "ी": "ee",
        "ु": "u",
        "ू": "oo",
        "े": "e",
        "ै": "ai",
        "ो": "o",
        "ौ": "au",
        "क": "k",
        "ख": "kh",
        "ग": "g",
        "घ": "gh",
        "च": "ch",
        "छ": "chh",
        "ज": "j",
        "झ": "jh",
        "ञ": "ny",
        "ट": "t",
        "ठ": "th",
        "ड": "d",
        "ढ": "dh",
        "ण": "n",
        "त": "t",
        "थ": "th",
        "द": "d",
        "ध": "dh",
        "न": "n",
        "प": "p",
        "फ": "ph",
        "ब": "b",
        "भ": "bh",
        "म": "m",
        "य": "y",
        "र": "r",
        "ल": "l",
        "व": "v",
        "श": "sh",
        "ष": "sh",
        "स": "s",
        "ह": "h",
        "ँ": "n",
        "ं": "n",
        "ः": "h",
        "ृ": "ri",
        "।": ".",
    }
    out = []
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            out.append(mapping.get(ch, ""))
        else:
            out.append(ch)
    return "".join(out)


def load_text_pairs(cfg: Config) -> Tuple[List[str], List[str]]:
    print("Loading datasets (streaming)...")
    H, E = [], []

    # Wikipedia EN
    try:
        ds_en = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for ex in tqdm(ds_en, desc="Wikipedia EN", total=250000):
            if len(E) >= cfg.samples_per_language:
                break
            txt = ex.get("text") or ""
            if _is_quality_english(txt, cfg):
                E.append(txt)
    except Exception as e:
        print(f"  Wikipedia EN load failed: {e}")

    # Wikipedia HI
    try:
        ds_hi = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.hi",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for ex in tqdm(ds_hi, desc="Wikipedia HI", total=250000):
            if len(H) >= int(cfg.samples_per_language * 0.5):
                break
            txt = ex.get("text") or ""
            if _is_quality_hindi(txt, cfg):
                H.append(txt)
    except Exception as e:
        print(f"  Wikipedia HI load failed: {e}")

    # Samanantar HI + EN(src)
    try:
        ds_sam_hi = load_dataset(
            "ai4bharat/samanantar", "hi", split="train", streaming=True
        )
        for ex in tqdm(ds_sam_hi, desc="Samanantar HI", total=500000):
            if len(H) >= cfg.samples_per_language:
                break
            hi = ex.get("tgt") or ""
            if _is_quality_hindi(hi, cfg):
                H.append(hi)
    except Exception as e:
        print(f"  Samanantar HI load failed: {e}")

    try:
        ds_sam_en = load_dataset(
            "ai4bharat/samanantar", "hi", split="train", streaming=True
        )
        for ex in tqdm(ds_sam_en, desc="Samanantar EN(src)", total=500000):
            if len(E) >= cfg.samples_per_language:
                break
            en = ex.get("src") or ""
            if _is_quality_english(en, cfg):
                E.append(en)
    except Exception as e:
        print(f"  Samanantar EN load failed: {e}")

    n = min(len(H), len(E), cfg.samples_per_language)
    H = H[:n]
    E = E[:n]

    # Romanized/Hinglish augmentation
    r = int(n * cfg.romanized_hindi_ratio)
    for i in range(r):
        rh = romanize_hindi_basic(H[i])
        H.append(rh)
        E.append(E[i])
        if i % 3 == 0 and len(rh.split()) > 4:
            words = rh.split()
            words2 = []
            for j, w in enumerate(words):
                if j % 5 == 0:
                    if w in ("hai", "hain", "hoon", "raha", "rahe", "rahi"):
                        words2.append("is")
                    elif w in ("mera", "meri", "mere"):
                        words2.append("my")
                    else:
                        words2.append(w)
                else:
                    words2.append(w)
            H.append(" ".join(words2))
            E.append(E[i])

    print(f"Loaded {len(H)} HI and {len(E)} EN samples")
    return H, E


# -----------------------------
# Strict splits with train-only augmentation
# -----------------------------


def _load_text_pairs_raw(cfg: Config) -> Tuple[List[str], List[str]]:
    """Load HI/EN texts without any augmentation, limited to samples_per_language.
    This replicates load logic but omits romanization/hinglish augmentation.
    """
    print("Loading datasets (streaming, raw)...")
    H, E = [], []

    # Wikipedia EN
    try:
        ds_en = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for ex in tqdm(ds_en, desc="Wikipedia EN", total=250000):
            if len(E) >= cfg.samples_per_language:
                break
            txt = ex.get("text") or ""
            if _is_quality_english(txt, cfg):
                E.append(txt)
    except Exception as e:
        print(f"  Wikipedia EN load failed: {e}")

    # Wikipedia HI
    try:
        ds_hi = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.hi",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for ex in tqdm(ds_hi, desc="Wikipedia HI", total=250000):
            if len(H) >= int(cfg.samples_per_language * 0.5):
                break
            txt = ex.get("text") or ""
            if _is_quality_hindi(txt, cfg):
                H.append(txt)
    except Exception as e:
        print(f"  Wikipedia HI load failed: {e}")

    # Samanantar HI + EN(src)
    try:
        ds_sam_hi = load_dataset(
            "ai4bharat/samanantar", "hi", split="train", streaming=True
        )
        for ex in tqdm(ds_sam_hi, desc="Samanantar HI", total=500000):
            if len(H) >= cfg.samples_per_language:
                break
            hi = ex.get("tgt") or ""
            if _is_quality_hindi(hi, cfg):
                H.append(hi)
    except Exception as e:
        print(f"  Samanantar HI load failed: {e}")

    try:
        ds_sam_en = load_dataset(
            "ai4bharat/samanantar", "hi", split="train", streaming=True
        )
        for ex in tqdm(ds_sam_en, desc="Samanantar EN(src)", total=500000):
            if len(E) >= cfg.samples_per_language:
                break
            en = ex.get("src") or ""
            if _is_quality_english(en, cfg):
                E.append(en)
    except Exception as e:
        print(f"  Samanantar EN load failed: {e}")

    n = min(len(H), len(E), cfg.samples_per_language)
    # De-duplicate within each language before pairing to reduce leakage
    def _uniq(lst: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for t in lst:
            key = (t or "").strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    H = _uniq(H)[:n]
    E = _uniq(E)[:n]
    print(f"[RAW] Loaded {len(H)} HI and {len(E)} EN samples (no augmentation)")
    return H, E


def load_text_pairs_splits(
    cfg: Config,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Load HI/EN and return strict train/val/test splits.

    Augmentation (romanized/Hinglish) is applied ONLY to training split.
    """
    H, E = _load_text_pairs_raw(cfg)

    # Shuffle pairs together to keep rough balance
    rng = random.Random(42)
    pairs = list(zip(H, E))
    rng.shuffle(pairs)
    H, E = map(list, zip(*pairs)) if pairs else ([], [])

    n = min(len(H), len(E))
    t_end = int(train_frac * n)
    v_end = t_end + int(val_frac * n)
    train_HI, train_EN = H[:t_end], E[:t_end]
    val_HI, val_EN = H[t_end:v_end], E[t_end:v_end]
    test_HI, test_EN = H[v_end:], E[v_end:]

    # Augmentation ONLY on training set
    r = int(len(train_HI) * cfg.romanized_hindi_ratio)
    for i in range(r):
        rh = romanize_hindi_basic(train_HI[i])
        train_HI.append(rh)
        train_EN.append(train_EN[i])
        if i % 3 == 0 and len(rh.split()) > 4:
            words = rh.split()
            words2 = []
            for j, w in enumerate(words):
                if j % 5 == 0:
                    if w in ("hai", "hain", "hoon", "raha", "rahe", "rahi"):
                        words2.append("is")
                    elif w in ("mera", "meri", "mere"):
                        words2.append("my")
                    else:
                        words2.append(w)
                else:
                    words2.append(w)
            train_HI.append(" ".join(words2))
            train_EN.append(train_EN[i])

    # Optional: Hinglish mining (train only)
    if cfg.use_hinglish_mining:
        def is_hinglish(txt: str) -> bool:
            if not txt:
                return False
            # English script but with Hindi roman markers
            return (devanagari_ratio(txt) < 0.2) and (roman_hi_ratio(txt) > 0.03)

        # Mine from existing train pools (cheap, no external datasets)
        cand = [t for t in (train_HI + train_EN) if is_hinglish(t)]
        if cand:
            add_n = int(len(train_HI) * cfg.hinglish_ratio_train)
            add = cand[: add_n]
            # Pair with repeating EN entries to keep lengths aligned
            for j, hgl in enumerate(add):
                train_HI.append(hgl)
                train_EN.append(train_EN[j % max(1, len(train_EN))])

    print(
        f"Splits → train: {len(train_HI)}/{len(train_EN)}, val: {len(val_HI)}/{len(val_EN)}, test: {len(test_HI)}/{len(test_EN)}"
    )
    return train_HI, train_EN, val_HI, val_EN, test_HI, test_EN


def _script_bucket_counts(texts: List[str]) -> Dict[str, int]:
    dev = sum(1 for t in texts if devanagari_ratio(t) >= 0.3)
    roman = sum(1 for t in texts if devanagari_ratio(t) < 0.1)
    mixed = max(0, len(texts) - dev - roman)
    return {"devanagari": dev, "roman": roman, "mixed": mixed}


def _run_lineage_meta(cfg: Config) -> Dict[str, str]:
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": cfg.model_name,
    }
    try:
        import subprocess, os as _os
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_root, stderr=subprocess.DEVNULL).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        pass
    return meta


# -----------------------------
# External Hinglish loader (VAL/TEST prompts only)
# -----------------------------


def _read_lines_txt(p: str) -> List[str]:
    out: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                out.append(t)
    return out


def _detect_text_field(sample: dict) -> Optional[str]:
    if not isinstance(sample, dict):
        return None
    candidates = [
        "text",
        "sentence",
        "input",
        "utterance",
        "content",
        "tweet",
        "hinglish",
        "source",
    ]
    for k in candidates:
        if k in sample and isinstance(sample[k], (str, bytes)):
            return k
    # Fallback: first string field
    for k, v in sample.items():
        if isinstance(v, str):
            return k
    return None


def load_external_hinglish(cfg: Config) -> List[str]:
    if not getattr(cfg, "use_external_hinglish", False):
        return []
    texts: List[str] = []
    try:
        # Local file path support
        if cfg.external_hinglish_file:
            p = cfg.external_hinglish_file
            if not _Path(p).exists():
                print(f"⚠️  External Hinglish file not found: {p}")
            else:
                low = p.lower()
                if low.endswith(".txt"):
                    texts.extend(_read_lines_txt(p))
                elif low.endswith(".jsonl"):
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = _json.loads(line)
                                fld = cfg.external_hinglish_text_field or _detect_text_field(obj)
                                if fld and isinstance(obj.get(fld), str):
                                    texts.append(obj[fld])
                            except Exception:
                                continue
                elif low.endswith(".json"):
                    with open(p, "r", encoding="utf-8") as f:
                        obj = _json.load(f)
                    if isinstance(obj, list):
                        if obj and isinstance(obj[0], str):
                            texts.extend([t for t in obj if isinstance(t, str)])
                        elif obj and isinstance(obj[0], dict):
                            fld = cfg.external_hinglish_text_field or _detect_text_field(obj[0])
                            for it in obj:
                                if fld and isinstance(it.get(fld), str):
                                    texts.append(it[fld])
                    elif isinstance(obj, dict):
                        # try a field with list of strings
                        for v in obj.values():
                            if isinstance(v, list) and v and isinstance(v[0], str):
                                texts.extend(v)
                                break
                elif low.endswith(".csv"):
                    import csv as _csv

                    with open(p, newline="", encoding="utf-8") as f:
                        reader = _csv.DictReader(f)
                        fld = cfg.external_hinglish_text_field or _detect_text_field(next(iter(reader), {}))
                        if fld:
                            texts.extend([row[fld] for row in reader if isinstance(row.get(fld), str)])
                else:
                    print(f"⚠️  Unsupported external file type: {p}")

        # Hugging Face dataset spec support
        if cfg.external_hinglish_hf_spec:
            try:
                split = cfg.external_hinglish_split or "train"
                ds = load_dataset(cfg.external_hinglish_hf_spec, split=split, streaming=True)
                cnt = 0
                for ex in ds:
                    fld = cfg.external_hinglish_text_field or _detect_text_field(ex)
                    if fld and isinstance(ex.get(fld), str):
                        texts.append(str(ex[fld]))
                        cnt += 1
                        if cnt >= max(1, cfg.external_hinglish_max):
                            break
            except Exception as e:
                print(f"⚠️  Failed to stream external HF Hinglish dataset: {e}")
    except Exception as e:
        print(f"⚠️  External Hinglish load error: {e}")

    # Basic filtering and de-dupe
    def _qual(t: str) -> bool:
        if not t:
            return False
        w = len(t.split())
        if w < 4 or w > cfg.max_sequence_length:
            return False
        return is_hinglish_text(t)

    seen = set()
    out: List[str] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        if _qual(t):
            seen.add(t)
            out.append(t)
        if len(out) >= cfg.external_hinglish_max:
            break
    if out:
        print(f"✅ Loaded {len(out)} external Hinglish prompts")
    else:
        print("ℹ️  No external Hinglish prompts loaded")
    return out


# -----------------------------
# Quick sweep for tuning steering params (optional)
# -----------------------------


@torch.no_grad()
def quick_sweep_steering(
    model,
    tok,
    cfg: Config,
    sel_saes: Dict[int, "JumpReLUSAE"],
    sel_hi: Dict[int, List[int]],
    sel_en: Dict[int, List[int]],
    eval_prompts: List[str],
    mode: str,
    target_lang: str,
):
    # Small grid; keep it cheap
    strengths = [1.4, 1.8, 2.2]
    clamps = [0.25, 0.35]
    topks = [16, 32]

    eval_prompts = eval_prompts[: min(24, len(eval_prompts))]
    best = None
    best_cfg = None
    for s in strengths:
        for cr in clamps:
            for tk in topks:
                # Temporarily use these settings
                orig_clamp = cfg.clamp_ratio
                cfg.clamp_ratio = cr
                steerer = MultiLayerSteerer(model, tok, cfg, sel_saes, sel_hi, sel_en)
                ok = 0
                tot = 0
                for p in eval_prompts:
                    st, bl = steerer.generate(
                        p,
                        target=target_lang.title(),
                        strength=s,
                        mode=mode,
                        topk=tk,
                        max_new_tokens=48,
                        deterministic=cfg.deterministic,
                    )
                    lang_b = detect_language_simple(bl)
                    lang_s = detect_language_simple(st)
                    if mode == "shadow_hindi":
                        ok += 1 if lang_s == "english" else 0
                    elif mode == "shadow_english":
                        ok += 1 if lang_s == "hindi" else 0
                    else:
                        ok += 1 if lang_s == target_lang.lower() else 0
                    tot += 1
                score = ok / max(1, tot)
                # restore
                cfg.clamp_ratio = orig_clamp
                if best is None or score > best:
                    best = score
                    best_cfg = (s, cr, tk)
    return best_cfg, best


@torch.no_grad()
def compute_perplexity(model, tok, device: str, text: str) -> float:
    # Compute per-token negative log-likelihood over the continuation part
    if not text:
        return float("inf")
    ids = tok(text, return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    labels = input_ids.clone()
    # Shift for causal LM
    out = model(input_ids=input_ids, labels=labels)
    loss = float(out.loss.detach().cpu().item())
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def try_bleu(candidate: str, reference: str) -> Optional[float]:
    try:
        import sacrebleu  # type: ignore

        # sacrebleu expects list of refs
        return float(sacrebleu.corpus_bleu([candidate], [[reference]]).score)
    except Exception:
        return None


@torch.no_grad()
def bayes_like_tuner(
    model,
    tok,
    cfg: Config,
    sel_saes: Dict[int, "JumpReLUSAE"],
    sel_hi: Dict[int, List[int]],
    sel_en: Dict[int, List[int]],
    eval_prompts: List[str],
    mode: str,
    target_lang: str,
    steps: int,
):
    import random as _random

    # Seed around current config
    best = None
    best_tuple = (cfg.eval_strength, cfg.clamp_ratio, cfg.eval_top_k_features, cfg.sup_factor)
    base_strength, base_clamp, base_topk, base_sup = best_tuple
    eval_prompts = eval_prompts[: min(32, len(eval_prompts))]

    def score_tuple(strength, clamp, topk, sup_factor):
        # Temporarily set
        orig_clamp, orig_sup = cfg.clamp_ratio, cfg.sup_factor
        cfg.clamp_ratio, cfg.sup_factor = float(clamp), float(sup_factor)
        steerer = MultiLayerSteerer(model, tok, cfg, sel_saes, sel_hi, sel_en)
        flips = 0
        total = 0
        ppl_vals = []
        bleu_vals = []
        for p in eval_prompts:
            st, bl = steerer.generate(
                p,
                target=target_lang.title(),
                strength=float(strength),
                mode=mode,
                topk=int(topk),
                max_new_tokens=48,
                deterministic=cfg.deterministic,
            )
            lang_s = detect_language_simple(st)
            if mode == "shadow_hindi":
                flips += 1 if lang_s == "english" else 0
            elif mode == "shadow_english":
                flips += 1 if lang_s == "hindi" else 0
            else:
                flips += 1 if lang_s == target_lang.lower() else 0
            total += 1
            # Perplexity on steered continuation (rough fluency proxy)
            ppl = compute_perplexity(model, tok, cfg.device, st)
            ppl_vals.append(ppl)
            # Optional BLEU if library present (baseline as pseudo-reference)
            b = try_bleu(st, bl)
            if b is not None:
                bleu_vals.append(b)
        cfg.clamp_ratio, cfg.sup_factor = orig_clamp, orig_sup
        flip_rate = flips / max(1, total)
        ppl_mean = float(np.mean(ppl_vals)) if ppl_vals else float("inf")
        bleu_mean = float(np.mean(bleu_vals)) if bleu_vals else None
        # Composite: higher is better; lower perplexity improves score, BLEU (if available) adds bonus
        # Normalize PPL by a soft transform: score_ppl = 1 / (1 + log(1+ppl))
        ppl_score = 1.0 / (1.0 + math.log1p(max(1e-6, ppl_mean)))
        bleu_score = (bleu_mean / 100.0) if (bleu_mean is not None) else 0.0
        composite = cfg.w_flip * flip_rate + cfg.w_ppl * ppl_score + cfg.w_bleu * bleu_score
        return composite, {
            "flip_rate": flip_rate,
            "ppl": ppl_mean,
            "bleu": bleu_mean,
            "strength": strength,
            "clamp_ratio": clamp,
            "top_k": int(topk),
            "sup_factor": sup_factor,
        }

    # Evaluate base
    best, best_rec = score_tuple(base_strength, base_clamp, base_topk, base_sup)
    best_tuple = (base_strength, base_clamp, base_topk, base_sup)

    # Random local search around base
    for _ in range(max(1, steps)):
        s = np.clip(np.random.normal(best_tuple[0], 0.4), 0.6, 3.0)
        cr = float(np.clip(np.random.normal(best_tuple[1], 0.08), 0.1, 0.7))
        tk = int(np.clip(int(best_tuple[2] + _random.choice([-16, 0, 16])), 8, 64))
        supf = float(np.clip(np.random.normal(best_tuple[3], 0.3), 0.2, 2.0))
        sc, rec = score_tuple(s, cr, tk, supf)
        if sc > best:
            best = sc
            best_tuple = (s, cr, tk, supf)
            best_rec = rec

    return best_tuple, best, best_rec


# -----------------------------
# Dynamic batch sizing
# -----------------------------


@torch.no_grad()
def find_max_batch_size(model, tok, texts: List[str], cfg: Config, seq_len: int) -> int:
    if not torch.cuda.is_available():
        return min(16, len(texts))
    lo, hi, best = 8, cfg.max_extract_bs, 8
    sample = texts[: max(8, min(128, len(texts)))]
    ids = tok(
        sample,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    ids = {k: v.to(cfg.device) for k, v in ids.items()}
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            rep = {k: v[:1].repeat(mid, 1) for k, v in ids.items()}
            _ = model(
                **rep,
                use_cache=False,
                output_hidden_states=False,
                attention_mask=rep.get("attention_mask"),
            )
            best = mid
            lo = mid + 8
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid - 8
        except Exception:
            hi = mid - 8
    return int(best)


# -----------------------------
# SAE definition
# -----------------------------


class JumpReLUSAE(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, l0_target: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.l0_target = l0_target
        self.pre_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.thresholds = nn.Parameter(torch.zeros(self.hidden_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.temperature = 0.30
        self._init()

    def _init(self):
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.thresholds.uniform_(0.3, 0.8)
        nn.init.zeros_(self.decoder_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        pre = self.encoder(x)  # [B,H]
        shifted = pre - self.thresholds.unsqueeze(0)
        hard = F.relu(shifted)
        if self.training:
            gate = torch.sigmoid(shifted / self.temperature)
            soft = gate * shifted
            feats = hard + (soft - soft.detach())
        else:
            feats = hard
        self._last_shifted = shifted
        return feats

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.encoder.weight.T, self.decoder_bias)

    def forward(self, x: torch.Tensor):
        feats = self.encode(x)
        recon = self.decode(feats)
        recon_loss = F.mse_loss(recon, x)

        shifted = self._last_shifted
        soft_gate = torch.sigmoid(shifted / self.temperature)
        l0_soft = soft_gate.sum(dim=-1).mean()
        l0_hard = (shifted > 0).float().sum(dim=-1).mean()

        l0_err = l0_soft - self.l0_target
        lam_up, lam_down = 1.5, 0.3
        l0_loss = lam_up * F.relu(l0_err) + lam_down * F.relu(-l0_err)

        l1_coeff = 1e-4
        l1_z = feats.abs().mean()

        total = recon_loss + l0_loss + l1_coeff * l1_z
        return recon, feats, {
            "reconstruction_loss": recon_loss.item(),
            "l0": l0_hard.item(),
            "l0_soft": l0_soft.item(),
            "l0_loss": l0_loss.item(),
            "total_loss": total.item(),
        }


# -----------------------------
# Pooled activations from a layer
# -----------------------------


def masked_pool(H: torch.Tensor) -> torch.Tensor:
    # No mask at generation; for training we use mean+last
    return 0.7 * H.mean(dim=1) + 0.3 * H[:, -1, :]


@torch.no_grad()
def pooled_from_layer_batch(model, layer: int, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
    pooled = {}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out  # [B,T,D]
        pooled[layer] = masked_pool(hs).detach().cpu()

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        _ = model(
            **inp, use_cache=False, output_hidden_states=False
        )
    finally:
        h.remove()
    return pooled.get(layer, torch.empty(0))


# -----------------------------
# Online SAE trainer (per layer)
# -----------------------------


class SAEOnlineTrainer:
    def __init__(self, cfg: Config):
        self.c = cfg

    def train_layer(
        self,
        model,
        tok,
        layer: int,
        train_texts: List[str],
        val_texts: List[str],
    ) -> JumpReLUSAE:
        d = model.config.hidden_size
        sae = JumpReLUSAE(d, self.c.sae_expansion_factor, self.c.sae_l0_target).to(
            self.c.device
        )
        sae.train()
        opt = torch.optim.AdamW(sae.parameters(), lr=self.c.lr, weight_decay=1e-4)

        steps_per_epoch = max(1, len(train_texts) // self.c.sae_batch_size)
        total_steps = self.c.train_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < self.c.warmup_steps:
                return step / max(1, self.c.warmup_steps)
            p = (step - self.c.warmup_steps) / max(1, total_steps - self.c.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        best = float("inf")
        best_l0_gap = float("inf")
        no_improve = 0
        best_path = Path(self.c.ckpt_dir) / f"sae_layer{layer}_best.pth"

        # Dynamic forward batch size (model)
        mbs = find_max_batch_size(
            model, tok, train_texts, self.c, self.c.max_sequence_length
        )
        mbs = max(self.c.start_extract_bs, mbs)
        if self.c.debug:
            print(f"[L{layer}] model batch size ~ {mbs}")

        step = 0
        for ep in range(self.c.train_epochs):
            random.shuffle(train_texts)
            acc = 0
            for i in range(0, len(train_texts), mbs):
                batch = train_texts[i : i + mbs]
                inp = tok(
                    batch,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.c.max_sequence_length,
                )
                inp = {k: v.to(self.c.device) for k, v in inp.items()}
                with torch.no_grad():
                    X = pooled_from_layer_batch(model, layer, inp)  # [B,D] CPU
                X = X.to(self.c.device, dtype=torch.float32)
                if X.numel() == 0:
                    continue
                # Split into SAE minibatches
                for j in range(0, X.size(0), self.c.sae_batch_size):
                    xb = X[j : j + self.c.sae_batch_size]
                    recon, feats, m = sae(xb)
                    shifted = sae._last_shifted
                    soft_gate = torch.sigmoid(shifted / sae.temperature)
                    l0_soft = soft_gate.sum(dim=-1).mean()
                    l0_err = l0_soft - sae.l0_target
                    lam_up, lam_down = 1.5, 0.3
                    l0_loss = lam_up * F.relu(l0_err) + lam_down * F.relu(-l0_err)
                    l1_coeff = 1e-4
                    l1_z = feats.abs().mean()
                    loss = F.mse_loss(recon, xb) + l0_loss + l1_coeff * l1_z
                    loss = loss / self.c.grad_accum_steps
                    loss.backward()
                    acc += 1
                    if acc % self.c.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad()
                        sched.step()
                        step += 1
            # Flush any remaining accumulation at epoch end
            if acc % self.c.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                sched.step()
                step += 1
            # Validation (small)
            sae.eval()
            with torch.no_grad():
                vs = min(8 * self.c.sae_batch_size, len(val_texts))
                val_sel = val_texts[:vs]
                Xv_list = []
                for k in range(0, len(val_sel), mbs):
                    b = val_sel[k : k + mbs]
                    inp = tok(
                        b,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=self.c.max_sequence_length,
                    )
                    inp = {k: v.to(self.c.device) for k, v in inp.items()}
                    Xv_list.append(pooled_from_layer_batch(model, layer, inp))
                if not Xv_list:
                    val_loss = float("inf")
                    vm = {"total_loss": val_loss, "l0_soft": 0.0}
                else:
                    Xv = torch.cat(Xv_list, dim=0).to(self.c.device, dtype=torch.float32)
                    if Xv.numel() > 0:
                        xb = Xv[: min(self.c.sae_batch_size, Xv.size(0))]
                        recon, feats, vm = sae(xb)
                        val_loss = F.mse_loss(recon, xb).item()
                    else:
                        val_loss = float("inf")
                        vm = {"total_loss": val_loss, "l0_soft": 0.0}
            sae.train()
            # Prefer lower val_loss; tie-break with l0 proximity
            l0_soft_val = float(vm.get('l0_soft', 0.0))
            l0_gap = abs(l0_soft_val - float(sae.l0_target))
            improved = val_loss < (best - 1e-6) or (abs(val_loss - best) <= 1e-6 and l0_gap < best_l0_gap)
            if improved:
                best = val_loss
                best_l0_gap = l0_gap
                torch.save(sae.state_dict(), best_path)
                no_improve = 0
            else:
                no_improve += 1
                if self.c.early_stop_patience and no_improve >= self.c.early_stop_patience:
                    print(f"Early stopping L{layer} at epoch {ep} (no improvement {no_improve} ≥ patience)")
                    break
            if ep % 10 == 0 or ep == self.c.train_epochs - 1:
                print(
                    f"  L{layer} Ep {ep:03d} val_loss={val_loss:.4f} "
                    f"l0_soft={vm.get('l0_soft', 0):.1f} "
                    f"LR={sched.get_last_lr()[0]:.2e}"
                )

        # Load best
        state = torch.load(best_path, map_location=self.c.device)
        sae.load_state_dict(state, strict=False)
        sae.eval()
        print(f"✅ Loaded best SAE for L{layer} from {best_path}")
        return sae


# -----------------------------
# Feature stats (streaming over SAE codes)
# -----------------------------


@torch.no_grad()
def feature_stats_streaming(
    model, tok, cfg: Config, sae: JumpReLUSAE, layer: int, HI: List[str], EN: List[str]
) -> Dict[str, np.ndarray]:
    H_dim = sae.hidden_dim
    # Running stats per feature for HI/EN
    st = {
        "hi": {
            "n": 0,
            "mean": torch.zeros(H_dim, dtype=torch.float64),
            "M2": torch.zeros(H_dim, dtype=torch.float64),
            "freq": torch.zeros(H_dim, dtype=torch.float64),
        },
        "en": {
            "n": 0,
            "mean": torch.zeros(H_dim, dtype=torch.float64),
            "M2": torch.zeros(H_dim, dtype=torch.float64),
            "freq": torch.zeros(H_dim, dtype=torch.float64),
        },
    }

    def update(side: str, Z: torch.Tensor):
        # Z: [B,H] fp32 on device; move to CPU float64 for stats
        Zc = Z.detach().cpu().to(torch.float64)
        B = Zc.size(0)
        if B == 0:
            return
        b_mean = Zc.mean(dim=0)  # [H]
        b_var = Zc.var(dim=0, unbiased=True)  # [H]
        b_M2 = b_var * max(1, B - 1)
        s = st[side]
        n, mean, M2 = s["n"], s["mean"], s["M2"]
        delta = b_mean - mean
        n_new = n + B
        mean_new = mean + delta * (B / max(1, n_new))
        M2_new = M2 + b_M2 + (delta * delta) * (n * B / max(1, n_new))
        s["n"], s["mean"], s["M2"] = n_new, mean_new, M2_new
        # Frequency (hard activation)
        s["freq"] += (Zc > 0).sum(dim=0)

    # Batch sizing
    mbs = find_max_batch_size(model, tok, HI[:2000] + EN[:2000], cfg, cfg.max_sequence_length)
    mbs = max(cfg.start_extract_bs, mbs)

    # Pass over HI
    for i in tqdm(range(0, len(HI), mbs), desc=f"FeatStats HI L{layer}"):
        batch = HI[i : i + mbs]
        inp = tok(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=cfg.max_sequence_length,
        )
        inp = {k: v.to(cfg.device) for k, v in inp.items()}
        X = pooled_from_layer_batch(model, layer, inp).to(cfg.device, dtype=torch.float32)
        if X.numel() == 0:
            continue
        Z = sae.encode(X)  # [B,H] fp32
        update("hi", Z)

    # Pass over EN
    for i in tqdm(range(0, len(EN), mbs), desc=f"FeatStats EN L{layer}"):
        batch = EN[i : i + mbs]
        inp = tok(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=cfg.max_sequence_length,
        )
        inp = {k: v.to(cfg.device) for k, v in inp.items()}
        X = pooled_from_layer_batch(model, layer, inp).to(cfg.device, dtype=torch.float32)
        if X.numel() == 0:
            continue
        Z = sae.encode(X)
        update("en", Z)

    # Compute stats
    n_hi, n_en = st["hi"]["n"], st["en"]["n"]
    mean_hi = st["hi"]["mean"].numpy()
    mean_en = st["en"]["mean"].numpy()
    var_hi = (st["hi"]["M2"] / max(1, n_hi - 1)).numpy()
    var_en = (st["en"]["M2"] / max(1, n_en - 1)).numpy()
    freq_hi = (st["hi"]["freq"].numpy() / max(1, n_hi)).astype(np.float64)
    freq_en = (st["en"]["freq"].numpy() / max(1, n_en)).astype(np.float64)

    # Welch t approx and Cohen's d
    # pooled std approx (per feature)
    pooled_std = np.sqrt(
        np.maximum(1e-12, ((n_hi - 1) * var_hi + (n_en - 1) * var_en) / max(1, (n_hi + n_en - 2)))
    )
    cohens_d = (mean_hi - mean_en) / np.maximum(1e-12, pooled_std)

    return {
        "mean_hi": mean_hi,
        "mean_en": mean_en,
        "var_hi": var_hi,
        "var_en": var_en,
        "freq_hi": freq_hi,
        "freq_en": freq_en,
        "cohens_d": cohens_d,
        "n_hi": np.array([n_hi]),
        "n_en": np.array([n_en]),
    }


def pick_top_features(stats: Dict[str, np.ndarray], top_n: int = 200):
    d = stats["cohens_d"]
    fh, fe = stats["freq_hi"], stats["freq_en"]
    # Score combines effect size and selectivity and activation frequency
    sel = np.abs(fh - fe)
    mag = np.abs(stats["mean_hi"]) + np.abs(stats["mean_en"])
    score = np.abs(d) + 0.3 * sel + 0.1 * mag
    idx = np.argsort(score)[::-1]
    # Split by sign of d
    hi_idx = [int(i) for i in idx if d[i] > 0][:top_n]
    en_idx = [int(i) for i in idx if d[i] < 0][:top_n]
    return hi_idx, en_idx


def filter_by_distribution(idx_list: List[int], stats: Dict[str, np.ndarray], side: str, cfg: Config) -> List[int]:
    if cfg.dist_min_freq is None and cfg.dist_max_freq is None:
        return idx_list
    fh, fe = stats.get("freq_hi"), stats.get("freq_en")
    out: List[int] = []
    for i in idx_list:
        freq = float(fh[i] if side == "hi" else fe[i])
        if cfg.dist_min_freq is not None and freq < cfg.dist_min_freq:
            continue
        if cfg.dist_max_freq is not None and freq > cfg.dist_max_freq:
            continue
        out.append(i)
    return out


@torch.no_grad()
def collect_codes_for_probe(
    model, tok, cfg: Config, sae: "JumpReLUSAE", layer: int, HI: List[str], EN: List[str], max_per_side: int = 4000
):
    # Sample up to max_per_side per class to fit a tiny probe
    HI = HI[: max_per_side]
    EN = EN[: max_per_side]
    mbs = find_max_batch_size(model, tok, HI[:2000] + EN[:2000], cfg, cfg.max_sequence_length)
    mbs = max(cfg.start_extract_bs, mbs)
    X_list, y_list = [], []
    for side, texts in [(0, HI), (1, EN)]:
        for i in range(0, len(texts), mbs):
            batch = texts[i : i + mbs]
            inp = tok(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=cfg.max_sequence_length,
            )
            inp = {k: v.to(cfg.device) for k, v in inp.items()}
            X = pooled_from_layer_batch(model, layer, inp).to(cfg.device, dtype=torch.float32)
            if X.numel() == 0:
                continue
            Z = sae.encode(X)  # [B,H]
            X_list.append(Z.detach().cpu().numpy())
            y_list.append(np.full(Z.size(0), side, dtype=np.int64))
    if not X_list:
        return np.empty((0, sae.hidden_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


def rank_features_by_probe(X: np.ndarray, y: np.ndarray, top_n: int = 200):
    if X.size == 0 or y.size == 0:
        return [], []
    # Simple linear classifier with L2; robust and fast
    clf = SGDClassifier(loss="log_loss", penalty="l2", max_iter=200, tol=1e-3)
    clf.fit(X, y)
    w = clf.coef_.reshape(-1)  # [H]
    # Positive weights → class 1 (EN), negative → class 0 (HI) depending on label mapping
    en_idx = list(np.argsort(-w)[:top_n].astype(int))
    hi_idx = list(np.argsort(w)[:top_n].astype(int))
    return hi_idx, en_idx


def stability_select(indices_list: List[List[int]], top_n: int) -> List[int]:
    if not indices_list:
        return []
    from collections import Counter

    cnt = Counter()
    for idxs in indices_list:
        cnt.update(idxs[: top_n * 2])
    ranked = [i for i, _ in cnt.most_common(top_n)]
    return ranked


# -----------------------------
# Causal feature ranking (optional)
# -----------------------------


@torch.no_grad()
def _plain_generate_continuation(model, tok, cfg: Config, prompt: str, max_new_tokens: int = 48, deterministic: bool = True) -> str:
    inp = tok(prompt, return_tensors="pt").to(cfg.device)
    gen_args = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if deterministic:
        g = model.generate(**inp, do_sample=False, **gen_args)
    else:
        g = model.generate(**inp, do_sample=True, temperature=0.8, top_p=0.9, **gen_args)
    prompt_len = int(inp["input_ids"].shape[1])
    cont_ids = g[0][prompt_len:]
    return tok.decode(cont_ids, skip_special_tokens=True).strip()


def _success_flag(text: str, mode: str) -> int:
    lang = detect_language_simple(text)
    if mode == "shadow_hindi":
        return 1 if lang == "english" else 0
    if mode == "shadow_english":
        return 1 if lang == "hindi" else 0
    # For generic steer, we don't define success here
    return 0


@torch.no_grad()
def _baseline_successes(model, tok, cfg: Config, prompts: List[str], mode: str) -> List[int]:
    flags: List[int] = []
    for p in prompts:
        base = _plain_generate_continuation(model, tok, cfg, p, max_new_tokens=48, deterministic=cfg.deterministic)
        flags.append(_success_flag(base, mode))
    return flags


@torch.no_grad()
def causal_rank_for_layer(
    model,
    tok,
    cfg: Config,
    sae: "JumpReLUSAE",
    layer: int,
    hi_candidates: List[int],
    en_candidates: List[int],
    eval_prompts: List[str],
    mode: str,
    top_n: int = 300,
):
    # Only define for shadow modes. For others, fall back outside.
    if mode not in ("shadow_hindi", "shadow_english"):
        return hi_candidates[:top_n], en_candidates[:top_n]

    prompts = eval_prompts[: min(16, len(eval_prompts))]
    base_flags = _baseline_successes(model, tok, cfg, prompts, mode)
    base_rate = float(sum(base_flags) / max(1, len(base_flags)))

    scores_hi: List[Tuple[int, float]] = []
    scores_en: List[Tuple[int, float]] = []

    # Helper to evaluate a single feature as a steered run
    def eval_feature(feat_idx: int, is_hi: bool) -> float:
        layer_saes = {layer: sae}
        if is_hi:
            layer_hi = {layer: [feat_idx]}
            layer_en = {layer: []}
        else:
            layer_hi = {layer: []}
            layer_en = {layer: [feat_idx]}
        steerer = MultiLayerSteerer(model, tok, cfg, layer_saes, layer_hi, layer_en)
        succ = 0
        for p in prompts:
            s, _ = steerer.generate(
                p,
                target=("English" if mode == "shadow_hindi" else "Hindi"),
                strength=cfg.eval_strength,
                mode=mode,
                topk=1,
                max_new_tokens=48,
                deterministic=cfg.deterministic,
            )
            succ += _success_flag(s, mode)
        rate = float(succ / max(1, len(prompts)))
        return rate - base_rate

    # Limit candidate set for compute
    hi_c = hi_candidates[: min(64, len(hi_candidates))]
    en_c = en_candidates[: min(64, len(en_candidates))]

    if mode == "shadow_hindi":
        # Suppress HI-like features only (amp list is disabled in shadow modes)
        for f in hi_c:
            try:
                d = eval_feature(f, is_hi=True)
                scores_hi.append((f, d))
            except Exception:
                continue
        scores_hi.sort(key=lambda x: x[1], reverse=True)
        hi_ranked = [fi for fi, _ in scores_hi][:top_n]
        # EN side unused for suppression in this mode; keep stats fallback order
        en_ranked = en_candidates[:top_n]
        hi_scores = {int(fi): float(sc) for fi, sc in scores_hi}
        en_scores = {int(fi): 0.0 for fi in en_ranked}
        return hi_ranked, en_ranked, hi_scores, en_scores

    if mode == "shadow_english":
        # Suppress EN-like features only
        for f in en_c:
            try:
                d = eval_feature(f, is_hi=False)
                scores_en.append((f, d))
            except Exception:
                continue
        scores_en.sort(key=lambda x: x[1], reverse=True)
        en_ranked = [fi for fi, _ in scores_en][:top_n]
        hi_ranked = hi_candidates[:top_n]
        hi_scores = {int(fi): 0.0 for fi in hi_ranked}
        en_scores = {int(fi): float(sc) for fi, sc in scores_en}
        return hi_ranked, en_ranked, hi_scores, en_scores
    hi_ranked = hi_candidates[:top_n]
    en_ranked = en_candidates[:top_n]
    return hi_ranked, en_ranked, {int(i): 0.0 for i in hi_ranked}, {int(i): 0.0 for i in en_ranked}


# -----------------------------
# Heuristic feature labeling
# -----------------------------


def roman_hi_ratio(text: str) -> float:
    # Simple marker set
    markers = {
        "hai",
        "hain",
        "hoon",
        "raha",
        "rahe",
        "rahi",
        "mera",
        "meri",
        "mere",
        "kya",
        "nahi",
        "ka",
        "ki",
        "ke",
        "mein",
        "hum",
        "aap",
        "tum",
        "bhai",
        "yaar",
        "ghar",
        "pyar",
    }
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in markers)
    return hits / max(1, len(toks))


def is_hinglish_text(text: str) -> bool:
    # English script but Hindi romanized markers present; low Devanagari
    return (devanagari_ratio(text) < 0.2) and (roman_hi_ratio(text) > 0.03)


def heuristic_label(feature_idx: int, top_hi: List[str], top_en: List[str]) -> str:
    hi_ratio = np.mean([devanagari_ratio(t) for t in top_hi]) if top_hi else 0.0
    en_ratio = np.mean([devanagari_ratio(t) for t in top_en]) if top_en else 0.0
    hi_roman = np.mean([roman_hi_ratio(t) for t in top_hi]) if top_hi else 0.0
    en_roman = np.mean([roman_hi_ratio(t) for t in top_en]) if top_en else 0.0

    if hi_roman > 0.05 and hi_ratio < 0.3:
        return f"F{feature_idx}: Hinglish/romanized-Hindi pattern"
    if hi_ratio > 0.6 and en_ratio < 0.1:
        return f"F{feature_idx}: Devanagari-heavy (Hindi script)"
    if en_ratio < 0.05 and any(w in " ".join(top_en).lower() for w in [" the ", " and ", " is "]):
        return f"F{feature_idx}: English lexical pattern"
    if hi_ratio > 0.3 and en_ratio < 0.1:
        return f"F{feature_idx}: Hindi morphological/syntactic pattern"
    return f"F{feature_idx}: Language-selective feature"


@torch.no_grad()
def top_activating_texts_for_feature(
    model, tok, cfg: Config, sae: JumpReLUSAE, layer: int, texts: List[str], feat: int, sample_n: int = 300
) -> List[str]:
    # Evaluate a small sample for top activations
    texts = texts[: min(sample_n, len(texts))]
    mbs = find_max_batch_size(model, tok, texts, cfg, cfg.max_sequence_length)
    mbs = max(cfg.start_extract_bs, mbs)
    acts: List[Tuple[int, float]] = []
    for i in range(0, len(texts), mbs):
        batch = texts[i : i + mbs]
        inp = tok(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=cfg.max_sequence_length,
        )
        inp = {k: v.to(cfg.device) for k, v in inp.items()}
        X = pooled_from_layer_batch(model, layer, inp).to(cfg.device, dtype=torch.float32)
        if X.numel() == 0:
            continue
        Z = sae.encode(X)  # [B,H]
        vals = Z[:, feat].detach().cpu().numpy()
        for j, v in enumerate(vals):
            acts.append((i + j, float(v)))
    acts.sort(key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx, _ in acts[:5]]
    return [texts[i] for i in top_idxs if i < len(texts)]


# -----------------------------
# Steering with SAEs
# -----------------------------


class MultiLayerSteerer:
    def __init__(
        self,
        model,
        tok,
        cfg: Config,
        layer_saes: Dict[int, JumpReLUSAE],
        layer_features_hi: Dict[int, List[int]],
        layer_features_en: Dict[int, List[int]],
        layer_weights: Optional[Dict[int, float]] = None,
    ):
        self.m = model
        self.tok = tok
        self.c = cfg
        self.saes = layer_saes
        self.hi = layer_features_hi
        self.en = layer_features_en
        if layer_weights and sum(layer_weights.values()) > 0:
            s = sum(layer_weights.values())
            self.w = {L: float(v / s) for L, v in layer_weights.items()}
        else:
            k = max(1, len(self.saes))
            self.w = {L: 1.0 / k for L in self.saes.keys()}

    def _pos_weights(self, T: int, start: float, end: float) -> torch.Tensor:
        if T <= 1:
            return torch.ones(T)
        return torch.linspace(start, end, steps=T)

    def _hook_sae(
        self, L: int, target: str, strength: float, mode: str, topk: int
    ):
        sae = self.saes[L]
        fhi = self.hi.get(L, [])[:topk]
        fen = self.en.get(L, [])[:topk]
        if target.lower() == "english":
            amp, sup = fen, fhi
        else:
            amp, sup = fhi, fen
        if mode == "shadow_hindi":
            amp, sup = [], fhi
        elif mode == "shadow_english":
            amp, sup = [], fen

        # Optional: label-driven per-feature scaling (e.g., Hinglish-specific features)
        # Compute per-feature multipliers for amplification/suppression based on labels
        amp_mul_t = sup_mul_t = None
        if getattr(self.c, "use_feature_labels", False):
            if not hasattr(self, "_label_cache"):
                self._label_cache = {}
                p = Path(getattr(self.c, "feature_labels_path", ""))
                if p.exists():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            self._label_cache = json.load(f)
                    except Exception:
                        self._label_cache = {}
            H = sae.hidden_dim
            amp_mul = torch.ones(H, dtype=torch.float32, device=sae.thresholds.device)
            sup_mul = torch.ones(H, dtype=torch.float32, device=sae.thresholds.device)
            for fi in set(amp + sup):
                key = f"L{L}:{fi}"
                lab = (self._label_cache.get(key) or "").lower()
                if "hinglish" in lab:
                    amp_mul[fi] = float(getattr(self.c, "hinglish_amp_boost", 0.8))
                    sup_mul[fi] = float(getattr(self.c, "hinglish_sup_boost", 1.2))
            amp_mul_t, sup_mul_t = amp_mul, sup_mul

        clamp_ratio = float(self.c.clamp_ratio)
        sup_factor = float(self.c.sup_factor)
        wL = float(self.w.get(L, 0.0))
        sae_dev = next(sae.parameters()).device
        scope = self.c.intervention_scope

        def fn(_, __, out):
            hs = out[0] if isinstance(out, tuple) else out  # [B,T,D]
            B, T, D = hs.shape

            if scope == "last":
                x = hs[:, -1, :]
                with torch.no_grad():
                    z = sae.encode(x.to(sae_dev, dtype=torch.float32))
                    zm = z.clone()
                    for i in amp:
                        if 0 <= i < zm.shape[1]:
                            zm[:, i] *= 1.0 + strength
                    for i in sup:
                        if 0 <= i < zm.shape[1]:
                            zm[:, i] *= max(0.0, 1.0 - sup_factor * strength)
                    xb = sae.decode(z)
                    xm = sae.decode(zm)
                    delta = (xm - xb).to(x.dtype)
                x_norm = x.float().norm(dim=1).clamp_min(1e-6)
                d_norm = delta.float().norm(dim=1).clamp_min(1e-6)
                scale = torch.clamp(
                    clamp_ratio * x_norm / d_norm, max=1.0
                ).unsqueeze(-1)
                delta = (delta.float() * scale).to(hs.dtype)
                hs[:, -1, :] = hs[:, -1, :] + wL * delta
                return (hs,) if isinstance(out, tuple) else hs

            # sequence-wide SAE intervention (vectorized over T)
            pos_w = self._pos_weights(T, self.c.pos_weight_start, self.c.pos_weight_end).to(hs.device)
            with torch.no_grad():
                x_bt = hs.reshape(B * T, D)
                z_bt = sae.encode(x_bt.to(sae_dev, dtype=torch.float32))  # [B*T, H]
                Hdim = z_bt.shape[1]
                # Build per-row per-feature scaling via deltaZ
                pos_w_rows = pos_w.repeat(B).to(z_bt.device, dtype=z_bt.dtype)  # [B*T]
                deltaZ = torch.zeros_like(z_bt)
                if amp:
                    idx = torch.tensor([i for i in amp if 0 <= i < Hdim], device=z_bt.device, dtype=torch.long)
                    if idx.numel() > 0:
                        base = 1.0 + strength * pos_w_rows.unsqueeze(1)
                        if amp_mul_t is not None:
                            base = base * amp_mul_t[idx].unsqueeze(0)
                        # delta factor = base - 1
                        delta_factor = (base - 1.0)
                        z_sel = z_bt[:, idx]
                        deltaZ[:, idx] = z_sel * delta_factor
                if sup:
                    idx = torch.tensor([i for i in sup if 0 <= i < Hdim], device=z_bt.device, dtype=torch.long)
                    if idx.numel() > 0:
                        base = 1.0 - sup_factor * strength * pos_w_rows.unsqueeze(1)
                        base = torch.clamp(base, min=0.0)
                        if sup_mul_t is not None:
                            base = base * sup_mul_t[idx].unsqueeze(0)
                        delta_factor = (base - 1.0)
                        z_sel = z_bt[:, idx]
                        deltaZ[:, idx] = z_sel * delta_factor
                # Decode only the change
                delta_bt = sae.decode(deltaZ).to(x_bt.dtype)  # [B*T, D]
                delta_all = delta_bt.reshape(B, T, D)
            x_norm = hs.float().norm(dim=2).clamp_min(1e-6)
            d_norm = delta_all.float().norm(dim=2).clamp_min(1e-6)
            scale = torch.clamp(self.c.clamp_ratio * x_norm / d_norm, max=1.0).unsqueeze(-1)
            if self.c.telemetry:
                # safety rail: detect pathological tiny scales
                min_scale = float(scale.min().detach().cpu().item())
                if min_scale < self.c.min_clamp_scale:
                    # zero out delta to avoid instability for this batch
                    delta_all = torch.zeros_like(delta_all)
            delta_all = (delta_all.float() * scale).to(hs.dtype)

            # Optional: DISCO-style query/value steering in addition to residual
            if getattr(self.c, "steer_qk", False) and hasattr(self.m.model.layers[L], "self_attn"):
                try:
                    attn = self.m.model.layers[L].self_attn
                    # Project queries/values, apply a gentle delta based on SAE residual
                    if hasattr(attn, "q_proj") and hasattr(attn, "v_proj") and hasattr(attn, "o_proj"):
                        with torch.no_grad():
                            q = attn.q_proj(hs)
                            v = attn.v_proj(hs)
                            q = q + float(self.c.steer_qv_strength) * delta_all
                            v = v + float(self.c.steer_qv_strength) * delta_all
                            # map back via output projection to residual space
                            hv = attn.o_proj(v)
                        hs = hs + wL * hv
                except Exception:
                    # If anything fails, fall back to residual-only edits
                    pass

            # Always apply residual delta
            hs = hs + wL * delta_all
            return (hs,) if isinstance(out, tuple) else hs

        return fn

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        target: str,
        strength: float,
        mode: str,
        topk: int = 32,
        max_new_tokens: int = 64,
        deterministic: bool = True,
    ):
        hooks = []
        old_use_cache = bool(getattr(self.m.config, "use_cache", True))
        if self.c.intervention_scope == "sequence":
            self.m.config.use_cache = False
        try:
            for L in self.saes.keys():
                h = self.m.model.layers[L].register_forward_hook(
                    self._hook_sae(L, target, strength, mode, topk)
                )
                hooks.append(h)
            inp = self.tok(prompt, return_tensors="pt").to(self.c.device)
            gen_args = dict(
                max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id
            )
            if deterministic:
                g = self.m.generate(**inp, do_sample=False, **gen_args)
            else:
                g = self.m.generate(
                    **inp, do_sample=True, temperature=0.8, top_p=0.9, **gen_args
                )
            # Slice continuation by token length
            prompt_len = int(inp["input_ids"].shape[1])
            steered_cont_ids = g[0][prompt_len:]
            steered = self.tok.decode(steered_cont_ids, skip_special_tokens=True)
        finally:
            for h in hooks:
                h.remove()
            self.m.config.use_cache = old_use_cache

        # Baseline
        inp = self.tok(prompt, return_tensors="pt").to(self.c.device)
        gen_args = dict(
            max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id
        )
        if deterministic:
            b = self.m.generate(**inp, do_sample=False, **gen_args)
        else:
            b = self.m.generate(
                **inp, do_sample=True, temperature=0.8, top_p=0.9, **gen_args
            )
        prompt_len = int(inp["input_ids"].shape[1])
        base_cont_ids = b[0][prompt_len:]
        baseline = self.tok.decode(base_cont_ids, skip_special_tokens=True)
        return steered.strip(), baseline.strip()


# -----------------------------
# Effectiveness scan (SAE only)
# -----------------------------


def build_eval_prompts(HI: List[str], EN: List[str], k: int, HGL: Optional[List[str]] = None, k_hinglish: int = 0) -> List[str]:
    def shorten(text):
        words = text.split()
        return " ".join(words[: min(18, len(words))])

    rng = random.Random(42)
    k_each = max(1, k // 2)
    hi = [shorten(t) for t in rng.sample(HI, min(k_each, len(HI)))]
    en = [shorten(t) for t in rng.sample(EN, min(k_each, len(EN)))]
    hgl = []
    if HGL and k_hinglish > 0:
        hgl = [shorten(t) for t in rng.sample(HGL, min(k_hinglish, len(HGL)))]
    extra = [
        "आज का मौसम कैसा है?",
        "भारत की राजधानी क्या है?",
        "Explain photosynthesis briefly.",
        "What is the capital of France?",
        "kal party hai kya? let's go",
        "main ghar ja raha hoon",
    ]
    return hi + en + hgl + extra


@torch.no_grad()
def scan_effectiveness_sae(
    model,
    tok,
    cfg: Config,
    layer_saes: Dict[int, JumpReLUSAE],
    layer_hi: Dict[int, List[int]],
    layer_en: Dict[int, List[int]],
    eval_prompts: List[str],
    target: str,
    mode: str,
) -> Dict[int, Dict[str, float]]:
    results = {}
    for L, sae in layer_saes.items():
        steerer = MultiLayerSteerer(
            model, tok, cfg, {L: sae}, {L: layer_hi.get(L, [])}, {L: layer_en.get(L, [])}
        )
        success = 0
        changed = 0
        total = 0
        for p in eval_prompts:
            s, b = steerer.generate(
                p,
                target=target,
                strength=cfg.eval_strength,
                mode=mode,
                topk=cfg.eval_top_k_features,
                max_new_tokens=48,
                deterministic=cfg.deterministic,
            )
            lang_b = detect_language_simple(b)
            lang_s = detect_language_simple(s)
            total += 1
            changed += 1 if lang_b != lang_s else 0
            if mode == "shadow_hindi":
                success += 1 if lang_s == "english" else 0
            elif mode == "shadow_english":
                success += 1 if lang_s == "hindi" else 0
            else:
                success += 1 if lang_s == target.lower() else 0
        results[L] = {
            "success_rate": success / max(1, total),
            "change_rate": changed / max(1, total),
            "eval_count": total,
        }
    return results


def choose_layers_by_effectiveness(
    eff: Dict[int, Dict[str, float]], k: int
) -> List[int]:
    if not eff:
        return []
    ranked = sorted(
        eff.items(), key=lambda x: x[1].get("success_rate", 0.0), reverse=True
    )
    chosen = [L for L, _ in ranked[: max(1, k)]]
    print("Layer effectiveness ranking (top first):")
    for L, m in ranked:
        print(
            f"  L{L}: success={m['success_rate']:.3f} change={m['change_rate']:.3f} "
            f"N={m['eval_count']}"
        )
    print(f"Selected by effectiveness: {chosen}")
    return chosen


# -----------------------------
# Orchestrator (SAE-only)
# -----------------------------


class Pipeline:
    def __init__(self, cfg: Config):
        self.c = cfg
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        setup_environment()
        self.model, self.tok = load_model_and_tokenizer(cfg)
        total = len(self.model.model.layers)
        start, end = self.c.layer_range
        assert 0 <= start < end <= total, f"Invalid layer_range {self.c.layer_range}"
        print(f"✅ Default layer range: [{start}, {end}) with {total} total layers")

    def run(self):
        c = self.c
        t0 = time.time()
        timings = {}
        # Strict splits with train-only aug
        tr_hi, tr_en, va_hi, va_en, te_hi, te_en = load_text_pairs_splits(c)
        timings["data_load_s"] = round(time.time() - t0, 3)
        print(f"Data load time: {timings['data_load_s']:.1f}s")

        # Determine candidate layers
        candidate_layers: List[int] = list(range(*c.layer_range))
        if c.test_layer_ranges:
            cand = []
            for spec in c.test_layer_ranges.split(","):
                if ":" in spec:
                    a, b = spec.split(":")
                    cand.extend(list(range(int(a), int(b))))
            candidate_layers = sorted(set(cand))
            print(f"Testing custom ranges: {candidate_layers}")

        # Build corpora for SAE training/validation and separate eval prompts for val/test
        train_texts = tr_hi + tr_en
        val_texts = va_hi[:2000] + va_en[:2000]
        if c.use_hinglish_mining:
            val_hgl = [t for t in (va_hi + va_en) if is_hinglish_text(t)]
            test_hgl = [t for t in (te_hi + te_en) if is_hinglish_text(t)]
        else:
            val_hgl, test_hgl = [], []
        # External Hinglish (val/test only)
        if c.use_external_hinglish:
            ext_hgl = load_external_hinglish(c)
            if ext_hgl:
                val_hgl = (val_hgl or []) + ext_hgl
                test_hgl = (test_hgl or []) + ext_hgl
        eval_prompts_val = build_eval_prompts(va_hi, va_en, c.eval_prompts, val_hgl, c.hinglish_eval_prompts)
        eval_prompts_test = build_eval_prompts(te_hi, te_en, c.eval_prompts, test_hgl, c.hinglish_eval_prompts)

        # Train or load SAE per candidate layer
        trainer = SAEOnlineTrainer(c)
        layer_saes: Dict[int, JumpReLUSAE] = {}
        t_train = time.time()
        for L in candidate_layers:
            ck = Path(c.ckpt_dir) / f"sae_layer{L}_best.pth"
            d = self.model.config.hidden_size
            sae = JumpReLUSAE(d, c.sae_expansion_factor, c.sae_l0_target).to(c.device)
            if getattr(c, "use_pretrained_sae", False) and (load_sae is not None):
                try:
                    print(f"⚙️  Loading pretrained SAE for L{L} via eai-sparsify...")
                    sae_pre = load_sae(c.model_name, layer=L, device=c.device)
                    if isinstance(sae_pre, nn.Module):
                        sae = sae_pre.to(c.device)
                        sae.eval()
                        print(f"✅ Pretrained SAE loaded for L{L}")
                    else:
                        print(f"ℹ️  eai-sparsify returned unexpected object for L{L}; training instead")
                except Exception as e:
                    print(f"⚠️  Failed to load pretrained SAE L{L}: {e}; will check local ckpt/train")
            if ck.exists():
                try:
                    state = torch.load(ck, map_location=c.device)
                    sae.load_state_dict(state, strict=False)
                    sae.eval()
                    print(f"✅ Loaded existing SAE L{L} from {ck}")
                except Exception as e:
                    print(f"⚠️  Failed to load SAE L{L}: {e}, retraining.")
                    sae = trainer.train_layer(self.model, self.tok, L, train_texts, val_texts)
            else:
                sae = trainer.train_layer(self.model, self.tok, L, train_texts, val_texts)
            layer_saes[L] = sae
        timings["sae_train_total_s"] = round(time.time() - t_train, 3)

        # Feature stats + selection per layer (use validation split only)
        layer_feat_hi: Dict[int, List[int]] = {}
        layer_feat_en: Dict[int, List[int]] = {}
        layer_causal_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
        label_map: Dict[str, str] = {}

        t_select = time.time()
        for L, sae in layer_saes.items():
            print(f"Selecting features for L{L} (method={c.feature_ranking})...")
            if c.feature_ranking == "probe":
                X, y = collect_codes_for_probe(self.model, self.tok, c, sae, L, va_hi[:4000], va_en[:4000])
                if c.stability_bootstrap and len(X) > 0:
                    idxs_hi, idxs_en = [], []
                    for _ in range(max(1, c.stability_bootstrap)):
                        n = int(max(10, c.stability_frac * len(X)))
                        sel = np.random.choice(len(X), size=n, replace=False)
                        hi_i, en_i = rank_features_by_probe(X[sel], y[sel], top_n=300)
                        idxs_hi.append(hi_i)
                        idxs_en.append(en_i)
                    hi_idx = stability_select(idxs_hi, 300)
                    en_idx = stability_select(idxs_en, 300)
                else:
                    hi_idx, en_idx = rank_features_by_probe(X, y, top_n=300)
                # Optional distribution-aware pruning for probe mode as well
                if (c.dist_min_freq is not None) or (c.dist_max_freq is not None):
                    stats_small = feature_stats_streaming(
                        self.model, self.tok, c, sae, L, va_hi[:2000], va_en[:2000]
                    )
                    hi_idx = filter_by_distribution(hi_idx, stats_small, "hi", c)
                    en_idx = filter_by_distribution(en_idx, stats_small, "en", c)
            elif c.feature_ranking == "causal":
                # Start from stats shortlist, then re-rank with causal deltas on a tiny val set
                stats = feature_stats_streaming(
                    self.model, self.tok, c, sae, L, va_hi[:2000], va_en[:2000]
                )
                base_hi, base_en = pick_top_features(stats, top_n=300)
                # Optional distribution-aware pruning
                base_hi = filter_by_distribution(base_hi, stats, "hi", c)
                base_en = filter_by_distribution(base_en, stats, "en", c)
                hi_idx, en_idx, hi_scores, en_scores = causal_rank_for_layer(
                    self.model,
                    self.tok,
                    c,
                    sae,
                    L,
                    base_hi,
                    base_en,
                    eval_prompts_val,
                    c.eval_mode,
                    top_n=300,
                )
                layer_causal_scores[f"L{L}"] = {
                    "hi": {str(k): float(v) for k, v in hi_scores.items()},
                    "en": {str(k): float(v) for k, v in en_scores.items()},
                }
            else:
                stats = feature_stats_streaming(
                    self.model, self.tok, c, sae, L, va_hi[:4000], va_en[:4000]
                )
                hi_idx, en_idx = pick_top_features(stats, top_n=300)
                # Optional distribution-aware pruning
                hi_idx = filter_by_distribution(hi_idx, stats, "hi", c)
                en_idx = filter_by_distribution(en_idx, stats, "en", c)
            layer_feat_hi[L] = hi_idx
            layer_feat_en[L] = en_idx

            # Heuristic labeling for top 10 per side (optional)
            for idx in hi_idx[:10]:
                top_hi = top_activating_texts_for_feature(
                    self.model, self.tok, c, sae, L, va_hi, idx, sample_n=300
                )
                label = heuristic_label(idx, top_hi, [])
                label_map[f"L{L}:{idx}"] = label
            for idx in en_idx[:10]:
                top_en = top_activating_texts_for_feature(
                    self.model, self.tok, c, sae, L, va_en, idx, sample_n=300
                )
                label = heuristic_label(idx, [], top_en)
                label_map[f"L{L}:{idx}"] = label

        # Effectiveness scan with SAEs only (use validation prompts)
        timings["feature_select_total_s"] = round(time.time() - t_select, 3)
        print("Scanning effectiveness (SAE-based steering)...")
        eff = scan_effectiveness_sae(
            self.model,
            self.tok,
            c,
            layer_saes,
            layer_feat_hi,
            layer_feat_en,
            eval_prompts_val,
            target="English" if c.eval_mode == "shadow_hindi" else "Hindi",
            mode=c.eval_mode,
        )
        chosen = choose_layers_by_effectiveness(eff, c.top_k_layers)
        if not chosen:
            chosen = [candidate_layers[-1]]
            print(f"Fallback chosen: {chosen}")

        # Normalize weights from success rate
        ssum = sum(eff[L]["success_rate"] for L in chosen) or 1.0
        layer_weights = {L: float(eff[L]["success_rate"] / ssum) for L in chosen}

        # Final steering with selected layers
        sel_saes = {L: layer_saes[L] for L in chosen}
        sel_hi = {L: layer_feat_hi[L] for L in chosen}
        sel_en = {L: layer_feat_en[L] for L in chosen}
        steerer = MultiLayerSteerer(self.model, self.tok, c, sel_saes, sel_hi, sel_en, layer_weights)

        # Optional quick sweep to tune steering params on validation prompts
        tuned = None
        if c.tune_steering or c.debug:
            tgt = "English" if c.eval_mode == "shadow_hindi" else "Hindi"
            cfg_choice, score = quick_sweep_steering(
                self.model, self.tok, c, sel_saes, sel_hi, sel_en, eval_prompts_val, c.eval_mode, tgt
            )
            if cfg_choice is not None:
                s_best, cr_best, tk_best = cfg_choice
                c.eval_strength = s_best
                c.clamp_ratio = cr_best
                c.eval_top_k_features = tk_best
                tuned = {"eval_strength": s_best, "clamp_ratio": cr_best, "top_k": tk_best, "val_score": score}
                print(f"Tuned steering on VAL → strength={s_best} clamp={cr_best} top_k={tk_best} (score={score:.3f})")

        # Optional bayesian-like tuning (random local search) that also considers perplexity/BLEU
        t_tune = time.time()
        if c.tune_bayes:
            tgt = "English" if c.eval_mode == "shadow_hindi" else "Hindi"
            best_tuple, best_score, best_rec = bayes_like_tuner(
                self.model, self.tok, c, sel_saes, sel_hi, sel_en, eval_prompts_val, c.eval_mode, tgt, c.tune_steps
            )
            s_best, cr_best, tk_best, sup_best = best_tuple
            c.eval_strength = float(s_best)
            c.clamp_ratio = float(cr_best)
            c.eval_top_k_features = int(tk_best)
            c.sup_factor = float(sup_best)
            tuned = tuned or {}
            tuned.update({
                "bayes_strength": c.eval_strength,
                "bayes_clamp": c.clamp_ratio,
                "bayes_top_k": c.eval_top_k_features,
                "bayes_sup_factor": c.sup_factor,
                "bayes_score": best_score,
                "bayes_record": best_rec,
            })
            print(
                f"Bayes-like tuned on VAL → strength={c.eval_strength:.2f} clamp={c.clamp_ratio:.2f} top_k={c.eval_top_k_features} sup={c.sup_factor:.2f} (score={best_score:.3f})"
            )

        timings["tuning_total_s"] = round(max(0.0, time.time() - t_tune), 3)
        print("Final evaluation on mixed prompts (TEST split)...")
        t_eval = time.time()
        ok = 0
        recs = []
        for p in eval_prompts_test:
            s, b = steerer.generate(
                p,
                target="English" if c.eval_mode == "shadow_hindi" else "Hindi",
                strength=c.eval_strength,
                mode=c.eval_mode,
                topk=c.eval_top_k_features,
                max_new_tokens=64,
                deterministic=c.deterministic,
            )
            lang_b = detect_language_simple(b)
            lang_s = detect_language_simple(s)
            if c.eval_mode == "shadow_hindi":
                ok += 1 if lang_s == "english" else 0
            elif c.eval_mode == "shadow_english":
                ok += 1 if lang_s == "hindi" else 0
            recs.append(
                {
                    "prompt": p,
                    "steered": s,
                    "baseline": b,
                    "steered_lang": lang_s,
                    "baseline_lang": lang_b,
                    "steered_dev_ratio": float(devanagari_ratio(s)),
                    "baseline_dev_ratio": float(devanagari_ratio(b)),
                }
            )
            print("\nPrompt:", p)
            print(" Steered:", s[:200])
            print(" Baseline:", b[:200])

        print(
            f"\nFinal success on TEST ({c.eval_mode}): {ok}/{len(eval_prompts_test)} "
            f"({100.0 * ok / max(1,len(eval_prompts_test)):.1f}%)"
        )
        timings["eval_total_s"] = round(time.time() - t_eval, 3)

        # Data balance & lineage metadata
        balance = {
            "train_hi": _script_bucket_counts(tr_hi),
            "train_en": _script_bucket_counts(tr_en),
            "val_hi": _script_bucket_counts(va_hi),
            "val_en": _script_bucket_counts(va_en),
            "test_hi": _script_bucket_counts(te_hi),
            "test_en": _script_bucket_counts(te_en),
        }
        lineage = _run_lineage_meta(c)

        out = {
            "config": asdict(c),
            "candidate_layers": candidate_layers,
            "effectiveness": eff,
            "selected_layers": chosen,
            "layer_weights": layer_weights,
            "top_features_hi": {str(L): sel_hi[L][:50] for L in chosen},
            "top_features_en": {str(L): sel_en[L][:50] for L in chosen},
            "causal_scores": layer_causal_scores,
            "labels": label_map,
            "results": recs,
            "timings_s": timings,
            "data_balance": balance,
            "lineage": lineage,
            "data_splits": {
                "train_hi": len(tr_hi),
                "train_en": len(tr_en),
                "val_hi": len(va_hi),
                "val_en": len(va_en),
                "test_hi": len(te_hi),
                "test_en": len(te_en),
            },
        }
        if tuned is not None:
            out["tuned_params"] = tuned
        with open(Path(c.out_dir) / "results_sae_only.json", "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {Path(c.out_dir) / 'results_sae_only.json'}")


# -----------------------------
# Main / CLI
# -----------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    p.add_argument("--layers", type=str, default=None)
    p.add_argument("--test-layer-ranges", type=str, default=None)
    p.add_argument("--mode", type=str, default=None)
    p.add_argument("--steer-alpha", type=float, default=None)
    p.add_argument("--norm-clamp-ratio", type=float, default=None)
    p.add_argument("--top-k-per-layer", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--samples-per-language", type=int, default=None)
    p.add_argument("--eval-prompts", type=int, default=None)
    p.add_argument("--intervention-scope", type=str, default=None)
    p.add_argument("--pos-weight-start", type=float, default=None)
    p.add_argument("--pos-weight-end", type=float, default=None)
    p.add_argument("--train-epochs", type=int, default=None)
    p.add_argument("--tune-steering", action="store_true")
    p.add_argument("--tune-bayes", action="store_true")
    p.add_argument("--tune-steps", type=int, default=None)
    p.add_argument("--sup-factor", type=float, default=None)
    p.add_argument("--w-flip", type=float, default=None)
    p.add_argument("--w-ppl", type=float, default=None)
    p.add_argument("--w-bleu", type=float, default=None)
    p.add_argument("--feature-ranking", type=str, default=None, help='"stats", "probe", or "causal"')
    p.add_argument("--stability-bootstrap", type=int, default=None)
    p.add_argument("--stability-frac", type=float, default=None)
    p.add_argument("--use-feature-labels", action="store_true")
    p.add_argument("--feature-labels-path", type=str, default=None)
    # DISCO-style attention steering
    p.add_argument("--steer-qk", action="store_true", help="Enable DISCO-style Q/V steering (experimental)")
    p.add_argument("--steer-qv-strength", type=float, default=None, help="Scale for Q/V deltas from SAE residual")
    # Determinism & telemetry
    p.add_argument("--nondeterministic", action="store_true", help="Disable deterministic generation for variability")
    p.add_argument("--telemetry", action="store_true", help="Enable steering telemetry and safety rails")
    p.add_argument("--min-clamp-scale", type=float, default=None)
    # Distribution-aware shortlist thresholds
    p.add_argument("--dist-min-freq", type=float, default=None)
    p.add_argument("--dist-max-freq", type=float, default=None)
    # GPU presets
    p.add_argument("--a100", action="store_true", help="Apply A100-friendly performance presets")
    p.add_argument("--h100", action="store_true", help="Apply H100-friendly performance presets")
    # External Hinglish options
    p.add_argument("--use-external-hinglish", action="store_true")
    p.add_argument("--external-hinglish-file", type=str, default=None)
    p.add_argument("--external-hinglish-hf-spec", type=str, default=None)
    p.add_argument("--external-hinglish-split", type=str, default=None)
    p.add_argument("--external-hinglish-text-field", type=str, default=None)
    p.add_argument("--external-hinglish-max", type=int, default=None)
    # Pretrained SAEs
    p.add_argument("--use-pretrained-sae", action="store_true", help="Load SAEs via eai-sparsify when available")
    args = p.parse_args()

    if torch.cuda.is_available():
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:256",
        )
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass

    cfg = Config()
    if args.debug:
        cfg.debug = True
        cfg.samples_per_language = 4000
        cfg.max_sequence_length = 256
        cfg.top_k_layers = 3
        cfg.eval_prompts = 16
        cfg.train_epochs = 60
        cfg.sae_batch_size = 512

    # GPU performance presets (H100 takes precedence if both are set)
    if getattr(args, "a100", False):
        cfg.use_flash_attn = True
        cfg.max_extract_bs = max(cfg.max_extract_bs, 8192)
        cfg.start_extract_bs = max(cfg.start_extract_bs, 64)
        cfg.sae_batch_size = max(cfg.sae_batch_size, 1024)
    if getattr(args, "h100", False):
        cfg.use_flash_attn = True
        cfg.max_extract_bs = max(cfg.max_extract_bs, 12288)
        cfg.start_extract_bs = max(cfg.start_extract_bs, 96)
        cfg.sae_batch_size = max(cfg.sae_batch_size, 1536)

    if args.mode:
        cfg.eval_mode = args.mode
    if args.steer_alpha is not None:
        cfg.eval_strength = float(args.steer_alpha)
    if args.norm_clamp_ratio is not None:
        cfg.clamp_ratio = float(args.norm_clamp_ratio)
    if args.top_k_per_layer is not None:
        cfg.eval_top_k_features = int(args.top_k_per_layer)
    if args.max_seq_length is not None:
        cfg.max_sequence_length = int(args.max_seq_length)
    if args.samples_per_language is not None:
        cfg.samples_per_language = int(args.samples_per_language)
    if args.eval_prompts is not None:
        cfg.eval_prompts = int(args.eval_prompts)
    if args.intervention_scope is not None:
        s = args.intervention_scope.lower().strip()
        cfg.intervention_scope = s if s in ("sequence", "last") else "sequence"
    if args.pos_weight_start is not None:
        cfg.pos_weight_start = float(args.pos_weight_start)
    if args.pos_weight_end is not None:
        cfg.pos_weight_end = float(args.pos_weight_end)
    if args.train_epochs is not None:
        cfg.train_epochs = int(args.train_epochs)
    if getattr(args, "tune_steering", False):
        cfg.tune_steering = True
    if getattr(args, "tune_bayes", False):
        cfg.tune_bayes = True
    if args.tune_steps is not None:
        cfg.tune_steps = int(args.tune_steps)
    if args.sup_factor is not None:
        cfg.sup_factor = float(args.sup_factor)
    if args.w_flip is not None:
        cfg.w_flip = float(args.w_flip)
    if args.w_ppl is not None:
        cfg.w_ppl = float(args.w_ppl)
    if args.w_bleu is not None:
        cfg.w_bleu = float(args.w_bleu)
    if args.feature_ranking is not None:
        s = args.feature_ranking.strip().lower()
        cfg.feature_ranking = s if s in ("stats", "probe", "causal") else "stats"
    if args.stability_bootstrap is not None:
        cfg.stability_bootstrap = int(args.stability_bootstrap)
    if args.stability_frac is not None:
        cfg.stability_frac = float(args.stability_frac)
    if getattr(args, "use_feature_labels", False):
        cfg.use_feature_labels = True
    if args.feature_labels_path is not None:
        cfg.feature_labels_path = args.feature_labels_path
    # DISCO-style steering
    if getattr(args, "steer_qk", False):
        cfg.steer_qk = True
    if args.steer_qv_strength is not None:
        cfg.steer_qv_strength = float(args.steer_qv_strength)
    # Determinism & telemetry
    if getattr(args, "nondeterministic", False):
        cfg.deterministic = False
    if getattr(args, "telemetry", False):
        cfg.telemetry = True
    if args.min_clamp_scale is not None:
        cfg.min_clamp_scale = float(args.min_clamp_scale)
    # Distribution-aware thresholds
    if args.dist_min_freq is not None:
        cfg.dist_min_freq = float(args.dist_min_freq)
    if args.dist_max_freq is not None:
        cfg.dist_max_freq = float(args.dist_max_freq)
    # External Hinglish
    if getattr(args, "use_external_hinglish", False):
        cfg.use_external_hinglish = True
    if args.external_hinglish_file is not None:
        cfg.external_hinglish_file = args.external_hinglish_file
    if args.external_hinglish_hf_spec is not None:
        cfg.external_hinglish_hf_spec = args.external_hinglish_hf_spec
    if args.external_hinglish_split is not None:
        cfg.external_hinglish_split = args.external_hinglish_split
    if args.external_hinglish_text_field is not None:
        cfg.external_hinglish_text_field = args.external_hinglish_text_field
    if args.external_hinglish_max is not None:
        cfg.external_hinglish_max = int(args.external_hinglish_max)
    # Pretrained SAEs
    if getattr(args, "use_pretrained_sae", False):
        cfg.use_pretrained_sae = True

    if args.layers:
        try:
            lst = sorted({int(x) for x in args.layers.split(",") if x.strip()})
            if lst:
                cfg.layer_range = (min(lst), max(lst) + 1)
                cfg.top_k_layers = len(lst)
                print(
                    f"Using user-specified layers: {lst} | "
                    f"layer_range={cfg.layer_range}"
                )
        except Exception as e:
            print(f"⚠️  Invalid --layers value '{args.layers}': {e}. Ignoring.")
    if args.test_layer_ranges:
        cfg.test_layer_ranges = args.test_layer_ranges

    pl = Pipeline(cfg)
    pl.run()


if __name__ == "__main__":
    main()
