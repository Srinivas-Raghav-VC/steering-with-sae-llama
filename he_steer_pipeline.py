#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# he_steer_pipeline.py
#
# End-to-end Hindi–English language steering on Llama-3.1-8B:
# - Gating checks for model/datasets
# - Data loading (EN: LMSYS gated + Wikipedia/OWT; HI: IITB + Samanantar + romanized)
# - Layer discovery (18–22 by default) with objective consensus:
#     LAPE + Probe + MMD (entropy-weighted, no manual weights)
# - JumpReLU SAE (stable dtypes on A100)
# - Feature discovery (stats + Goodfire-style gradients)
# - Gemini 2.x/1.5 Flash auto-interpretation of top features
# - Gradio UI: steer/shadow live; toggle features
#
# First principles + research:
# - Residual stream encodes language features linearly; local feature interventions
#   steer outputs (Anthropic 2024; Goodfire 2024)
# - MMD for geometric separation (Gretton et al. JMLR 2012)
# - Entropy weighting (objective MCDM), no arbitrary weights (Roszkowska & Wachowicz 2024)
#
# A100-80GB:
# - Single-GPU device_map="cuda" to keep hooks local
# - TF32 + bf16 + big batches + autocast + batch finder to fill VRAM
#
# Author: You, with assist
# Date: 2025-09

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, login
from scipy.stats import entropy as shannon_entropy
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Utilities / Environment
# -----------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def bytes_gb(x: int) -> float:
    return x / (1024**3)


class Environment:
    @staticmethod
    def setup():
        print("🔧 Environment setup...")
        # HF auth
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("❌ HF_TOKEN not set. Set: export HF_TOKEN=...")
        else:
            try:
                login(token=hf_token, add_to_git_credential=True)
                print("Hugging Face login OK")
            except Exception as e:
                print(f"HF login error: {e}")

        # CUDA
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except Exception:
                pass
            dev = torch.cuda.get_device_properties(0)
            print(
                f"CUDA: {dev.name} | VRAM: {bytes_gb(dev.total_memory):.1f} GB | "
                f"CC: {dev.major}.{dev.minor}"
            )
        else:
            print(" CUDA not available; CPU fallback (slow).")

        # Gating checks
        api = HfApi()
        statuses = {}

        def check_model_access(model_id: str):
            try:
                _ = api.model_info(model_id, token=hf_token)
                statuses[model_id] = "accessible"
            except Exception:
                statuses[model_id] = "requires_request"

        def check_dataset_access(ds_id: str):
            try:
                _ = api.dataset_info(ds_id, token=hf_token)
                statuses[ds_id] = "accessible_or_gated_ok"
            except Exception:
                statuses[ds_id] = "requires_agreement"

        check_model_access("meta-llama/Llama-3.1-8B-Instruct")
        check_dataset_access("lmsys/lmsys-chat-1m")
        check_dataset_access("cfilt/iitb-english-hindi")
        check_dataset_access("ai4bharat/samanantar")

        print("Access status:")
        for k, v in statuses.items():
            print(f"  - {k}: {v}")

        if statuses.get("meta-llama/Llama-3.1-8B-Instruct") != "accessible":
            print(
                "❗ Llama-3.1-8B-Instruct access not confirmed. "
                "Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct "
                "and request/accept terms."
            )
        if statuses.get("lmsys/lmsys-chat-1m") == "requires_agreement":
            print(
                "  LMSYS-Chat-1M is gated; accept terms at "
                "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
            )

        return statuses


# -----------------------------
# Config
# -----------------------------


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = device_str()

    # Data
    samples_per_language: int = 4000
    max_sequence_length: int = 256
    min_sequence_length: int = 24
    romanized_hindi_ratio: float = 0.2  # add romanized slice to avoid script-only cues

    # SAE
    sae_expansion_factor: int = 32
    sae_l0_target: int = 100

    # Training
    training_epochs: int = 300
    batch_size: int = 1024
    grad_accum_steps: int = 2
    lr: float = 1e-3
    warmup_steps: int = 80

    # Layer discovery (default mid/late; will also verify empirically)
    layer_range: Tuple[int, int] = (18, 23)  # scans 18..22 inclusive

    # Layer scan configuration
    scan_all_layers: bool = (
        True  # if True, scan all model layers instead of layer_range
    )
    top_k_layers: int = 3  # number of top layers to select from scan

    # Objective consensus (no arbitrary weights)
    geometric_metric: str = "mmd"  # mmd | centroid
    consensus_weighting: str = "entropy"  # entropy | equal

    # Stats
    significance_threshold: float = 0.01
    min_effect_size: float = 0.5
    bonferroni: bool = True

    # Steering
    default_strength: float = 1.5
    top_k_features: int = 5

    # Paths
    out_dir: str = "he_pipeline_results"
    ckpt_dir: str = "he_checkpoints"

    # Gemini
    gemini_model: Optional[str] = None  # from env or fallback
    gemini_timeout: int = 30

    # Debug
    debug: bool = False

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Romanization (simple)
# -----------------------------


def romanize_hindi_basic(text: str) -> str:
    # Very rough transliteration; enough to reduce pure script cues.
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
        "ङ": "ng",
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
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
        "।": ".",
        "”": '"',
        "“": '"',
    }
    out = []
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            out.append(mapping.get(ch, ""))
        else:
            out.append(ch)
    return "".join(out)


# -----------------------------
# Datasets
# -----------------------------


class DatasetLoader:
    def __init__(self, cfg: Config, access: Dict[str, str]):
        self.c = cfg
        self.access = access

    def _is_quality_english(self, txt: str) -> bool:
        if not txt:
            return False
        words = txt.split()
        n = len(words)
        if n < self.c.min_sequence_length or n > self.c.max_sequence_length:
            return False
        # Low ratio of Devanagari
        if any("\u0900" <= c <= "\u097f" for c in txt[:200]):
            return False
        # Basic English markers
        en_markers = {"the", "and", "is", "are", "was", "were", "this", "that"}
        cnt = sum(1 for w in words[:50] if w.lower() in en_markers)
        if cnt < 2:
            return False
        return True

    def _is_quality_hindi(self, txt: str) -> bool:
        if not txt:
            return False
        words = txt.split()
        n = len(words)
        if n < self.c.min_sequence_length or n > self.c.max_sequence_length:
            return False
        alpha = sum(1 for c in txt if c.isalpha())
        dev = sum(1 for c in txt if "\u0900" <= c <= "\u097f")
        if alpha > 0 and (dev / alpha) < 0.5:
            return False
        # Hindi markers
        for m in ["है", "हैं", "में", "और", "का", "की", "के", "से"]:
            if m in txt:
                return True
        return False

    def load(self) -> Tuple[List[str], List[str]]:
        print(" Loading datasets...")
        H, E = [], []

        # English
        if self.access.get("lmsys/lmsys-chat-1m") != "requires_agreement":
            try:
                ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
                cnt = 0
                for ex in tqdm(ds, desc="LMSYS", total=200000):
                    if cnt >= int(self.c.samples_per_language * 0.6):
                        break
                    conv = ex.get("conversation")
                    if isinstance(conv, list) and len(conv) >= 1:
                        text = " ".join([t.get("content", "") for t in conv[:2]])
                        if self._is_quality_english(text):
                            E.append(text)
                            cnt += 1
            except Exception as e:
                print(f"   LMSYS stream failed: {e}")

        # Wikipedia EN fallback
        if len(E) < int(self.c.samples_per_language * 0.8):
            try:
                ds = load_dataset(
                    "wikipedia", "20220301.en", split="train", streaming=True
                )
                tgt = int(self.c.samples_per_language * 0.3)
                cnt = 0
                for ex in tqdm(ds, desc="WikiEN", total=tgt * 2):
                    if cnt >= tgt:
                        break
                    txt = ex.get("text", "")
                    paras = [p for p in txt.split("\n\n") if 40 < len(p.split()) < 200]
                    if not paras:
                        continue
                    para = paras[0]
                    if self._is_quality_english(para):
                        E.append(para)
                        cnt += 1
            except Exception as e:
                print(f"   Wikipedia EN failed: {e}")

        # OpenWebText fallback
        if len(E) < self.c.samples_per_language:
            try:
                ds = load_dataset("openwebtext", split="train", streaming=True)
                need = self.c.samples_per_language - len(E)
                cnt = 0
                for ex in tqdm(ds, desc="OWT", total=need * 2):
                    if cnt >= need:
                        break
                    txt = ex.get("text", "")
                    if self._is_quality_english(txt):
                        words = txt.split()
                        chunk = " ".join(words[30:180]) if len(words) > 200 else txt
                        E.append(chunk)
                        cnt += 1
            except Exception as e:
                print(f"   OWT failed: {e}")

        # Hindi
        # IITB
        try:
            ds = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)
            tgt = int(self.c.samples_per_language * 0.4)
            cnt = 0
            for ex in tqdm(ds, desc="IITB", total=tgt * 2):
                if cnt >= tgt:
                    break
                hi = ex.get("translation", {}).get("hi", "")
                if self._is_quality_hindi(hi):
                    H.append(hi)
                    cnt += 1
        except Exception as e:
            print(f"   IITB failed: {e}")

        # Samanantar (hi)
        try:
            ds = load_dataset(
                "ai4bharat/samanantar", "hi", split="train", streaming=True
            )
            tgt = int(self.c.samples_per_language * 0.4)
            cnt = 0
            for ex in tqdm(ds, desc="Samanantar(hi)", total=tgt * 2):
                if cnt >= tgt:
                    break
                hi = ex.get("tgt", "")
                if self._is_quality_hindi(hi):
                    H.append(hi)
                    cnt += 1
        except Exception as e:
            print(f"   Samanantar failed: {e}")

        # Indic News (if available)
        try:
            ds = load_dataset(
                "ai4bharat/IndicNLP-News-Article-Classification",
                split="train",
                streaming=True,
            )
            tgt = int(self.c.samples_per_language * 0.2)
            cnt = 0
            for ex in tqdm(ds, desc="IndicNews", total=tgt * 2):
                if cnt >= tgt:
                    break
                if ex.get("language") == "hi":
                    text = (ex.get("headline", "") + " " + ex.get("content", ""))[:1000]
                    if self._is_quality_hindi(text):
                        H.append(text)
                        cnt += 1
        except Exception as e:
            print(f"   IndicNews failed: {e}")

        # Balance and romanize slice
        n = min(len(H), len(E), self.c.samples_per_language)
        H = H[:n]
        E = E[:n]

        # Romanized Hindi slice
        r = int(n * self.c.romanized_hindi_ratio)
        for i in range(r):
            H.append(romanize_hindi_basic(H[i]))
            E.append(E[i])  # keep balance

        print(f"Loaded {len(H)} HI and {len(E)} EN samples")
        return H, E


# -----------------------------
# JumpReLU SAE
# -----------------------------


class JumpReLUSAE(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, l0_target: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.l0_target = l0_target

        self.pre_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.thresholds = nn.Parameter(torch.zeros(self.hidden_dim))  # fp32
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.temperature = 0.1
        self._init()

    def _init(self):
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.thresholds.uniform_(0.5, 1.0)
        nn.init.zeros_(self.decoder_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x is fp32
        x = self.pre_norm(x)
        pre = self.encoder(x)  # [B, H]
        shifted = pre - self.thresholds.unsqueeze(0)  # [B, H]

        # JumpReLU features = ReLU(pre - theta)
        hard = F.relu(shifted)

        if self.training:
            gate = torch.sigmoid(shifted / self.temperature)  # soft gate for grad
            soft = gate * shifted
            feats = hard + (soft - soft.detach())
        else:
            feats = hard

        # stash for l0 computation in forward
        self._last_shifted = shifted
        return feats

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.encoder.weight.T, self.decoder_bias)

    def forward(self, x: torch.Tensor):
        feats = self.encode(x)
        recon = self.decode(feats)

        recon_loss = F.mse_loss(recon, x)

        # soft L0 (for gradients)
        shifted = getattr(self, "_last_shifted", None)
        if shifted is None:
            shifted = self.encoder(self.pre_norm(x)) - self.thresholds.unsqueeze(0)
        soft_gate = torch.sigmoid(shifted / self.temperature)
        l0_soft = soft_gate.sum(dim=-1).mean()

        # hard L0 (for reporting)
        l0_hard = (shifted > 0).float().sum(dim=-1).mean()

        l0_loss = F.relu(l0_soft - self.l0_target) * 1.0
        total = recon_loss + l0_loss

        metrics = {
            "reconstruction_loss": recon_loss.item(),
            "l0": l0_hard.item(),
            "l0_soft": l0_soft.item(),
            "l0_loss": l0_loss.item(),
            "total_loss": total.item(),
        }
        return recon, feats, metrics


# -----------------------------
# Layer Discovery (LAPE + Probe + MMD) with Entropy weighting
# -----------------------------


class LayerDiscovery:
    def __init__(self, model, tok, cfg: Config):
        self.m = model
        self.tok = tok
        self.c = cfg

    def run(self, H: List[str], E: List[str]) -> Tuple[List[int], Dict[int, float]]:
        print("Layer discovery...")
        sample_n = min(1200 if not self.c.debug else 300, len(H), len(E))
        h = H[:sample_n]
        e = E[:sample_n]
        texts = h + e
        layers = list(range(*self.c.layer_range))
        acts = self._extract(texts, layers)

        scores = {}
        labels = [0] * len(h) + [1] * len(e)

        for L, A in tqdm(acts.items(), desc="Scoring layers"):
            lape = self._lape(A[: len(h)], A[len(h) :])
            probe = self._probe(A, labels)
            if self.c.geometric_metric == "mmd":
                geom = self._mmd(A[: len(h)], A[len(h) :])
            else:
                geom = self._centroid_sep(A[: len(h)], A[len(h) :])
            scores[L] = {"lape": lape, "probing": probe, "geometric": geom}

        weights = (
            self._entropy_weights(scores)
            if self.c.consensus_weighting == "entropy"
            else {"lape": 1 / 3, "probing": 1 / 3, "geometric": 1 / 3}
        )
        print(f"  Consensus weighting: {weights}")

        for L, s in scores.items():
            s["consensus"] = (
                weights["lape"] * s["lape"]
                + weights["probing"] * s["probing"]
                + weights["geometric"] * s["geometric"]
            )
            print(
                f"  L{L}: LAPE={s['lape']:.3f} Probe={s['probing']:.3f} "
                f"Geom={s['geometric']:.4f} -> Consensus={s['consensus']:.3f}"
            )

        best = sorted(scores.items(), key=lambda x: x[1]["consensus"], reverse=True)
        k = max(1, int(self.c.top_k_layers))
        chosen = [lid for (lid, _) in best[:k]]
        with open(Path(self.c.out_dir) / "layer_scores.json", "w") as f:
            json.dump(
                {"scores": scores, "chosen": chosen, "weights": weights}, f, indent=2
            )
        print(f"Selected layer(s): {chosen}")
        return chosen, {L: scores[L]["consensus"] for L in chosen}

    def _extract(self, texts: List[str], layers: List[int]) -> Dict[int, torch.Tensor]:
        m, tok, dev = self.m, self.tok, self.c.device
        activations: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}

        # 1) Find batch size BEFORE registering hooks (prevents double-collection)
        bs = self._find_max_batch_size_for_extract(texts)
        print(f"  Using batch size {bs} for extraction")

        # 2) Now register hooks
        def hook(L):
            def fn(_, __, out):
                hs = out[0] if isinstance(out, tuple) else out
                last = hs[:, -1, :]
                mean = hs.mean(dim=1)
                pooled = 0.7 * mean + 0.3 * last
                activations[L].append(pooled.detach().cpu())

            return fn

        hooks = []
        for L in layers:
            hooks.append(m.model.layers[L].register_forward_hook(hook(L)))

        # 3) Run extraction once
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                inp = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.c.max_sequence_length,
                )
                inp = {k: v.to(dev) for k, v in inp.items()}
                _ = m(**inp)
                if torch.cuda.is_available() and (i // bs) % 2 == 0:
                    torch.cuda.empty_cache()

        # 4) Remove hooks
        for h in hooks:
            h.remove()

        # 5) Concatenate per layer
        out = {}
        for L in layers:
            if activations[L]:
                out[L] = torch.cat(activations[L], dim=0)
        return out

    def _find_max_batch_size_for_extract(self, texts: List[str]) -> int:
        if not torch.cuda.is_available():
            return 64
        lo, hi = 64, 2048
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                with torch.no_grad():
                    batch = texts[:mid]
                    inp = self.tok(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.c.max_sequence_length,
                    )
                    _ = self.m(**{k: v.to(self.c.device) for k, v in inp.items()})
                best = mid
                lo = mid + 64
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hi = mid - 64
        return best

    def _lape(self, A_hi: torch.Tensor, A_en: torch.Tensor) -> float:
        hi = A_hi.float().cpu().numpy()
        en = A_en.float().cpu().numpy()
        n = min(1000, hi.shape[0], en.shape[0])
        hi, en = hi[:n], en[:n]
        activity = np.mean(np.abs(hi), axis=0) + np.mean(np.abs(en), axis=0)
        k = min(2000, hi.shape[1])
        idx = np.argsort(activity)[-k:]
        vals = []
        for i in idx:
            hai = (hi[:, i] > 0).astype(float)
            eai = (en[:, i] > 0).astype(float)
            if hai.sum() == 0 and eai.sum() == 0:
                continue
            p = np.array([hai.mean(), eai.mean()])
            p = p / (p.sum() + 1e-8)
            H = shannon_entropy(p + 1e-8, base=2)
            sep = 1.0 - H
            mag = np.mean(np.abs(hi[:, i])) + np.mean(np.abs(en[:, i]))
            vals.append(sep * mag)
        return float(np.mean(vals) if vals else 0.0)

    def _probe(self, A: torch.Tensor, labels: List[int]) -> float:
        X = A.float().cpu().numpy()
        y = np.array(labels)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        return float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())

    def _centroid_sep(self, A: torch.Tensor, B: torch.Tensor) -> float:
        a = A.mean(dim=0)
        b = B.mean(dim=0)
        dist = torch.norm(a - b).item()
        sA = torch.trace(torch.cov(A.float().T)).item()
        sB = torch.trace(torch.cov(B.float().T)).item()
        return dist / (1.0 + 0.5 * (sA + sB))

    def _mmd(self, A: torch.Tensor, B: torch.Tensor) -> float:
        with torch.no_grad():
            X = A.float()
            Y = B.float()
            n = min(512, X.shape[0], Y.shape[0])
            X = X[:n]
            Y = Y[:n]
            Z = torch.cat([X, Y], dim=0)
            D = torch.cdist(Z, Z, p=2)
            med = torch.median(D[D > 0]).item()
            sigma2 = max(med**2, 1e-6)
            gamma = 1.0 / (2.0 * sigma2)

            def k(U, V):
                return torch.exp(-gamma * torch.cdist(U, V, p=2) ** 2)

            Kxx = k(X, X)
            Kyy = k(Y, Y)
            Kxy = k(X, Y)
            nf = float(n)
            mmd2 = (
                (Kxx.sum() - torch.diagonal(Kxx).sum()) / (nf * (nf - 1.0))
                + (Kyy.sum() - torch.diagonal(Kyy).sum()) / (nf * (nf - 1.0))
                - 2.0 * Kxy.mean()
            )
            return float(max(mmd2.item(), 0.0))

    def _entropy_weights(
        self, per_layer: Dict[int, Dict[str, float]]
    ) -> Dict[str, float]:
        metrics = ["lape", "probing", "geometric"]
        Ls = list(per_layer.keys())
        M = np.array([[per_layer[L][m] for m in metrics] for L in Ls], dtype=np.float64)
        lo = M.min(axis=0)
        hi = M.max(axis=0)
        rng = np.maximum(hi - lo, 1e-8)
        X = (M - lo) / rng
        X = X + 1e-12
        P = X / np.sum(X, axis=0, keepdims=True)
        k = 1.0 / np.log(P.shape[0])
        E = -k * np.sum(P * np.log(P), axis=0)
        D = 1.0 - E
        w = D / (np.sum(D) + 1e-12)
        return {"lape": float(w[0]), "probing": float(w[1]), "geometric": float(w[2])}


# -----------------------------
# SAE Training
# -----------------------------


class SAETrainer:
    def __init__(self, cfg: Config):
        self.c = cfg
        self.best = float("inf")

    def extract_layer_acts(
        self, model, tok, texts: List[str], layer: int
    ) -> torch.Tensor:
        acts: List[torch.Tensor] = []

        def hook_fn(_, __, out):
            hs = out[0] if isinstance(out, tuple) else out
            last = hs[:, -1, :]
            mean = hs.mean(dim=1)
            pooled = 0.7 * mean + 0.3 * last
            acts.append(pooled.detach().cpu())

        h = model.model.layers[layer].register_forward_hook(hook_fn)
        bs = self._find_bs(model, tok, texts)
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), bs), desc=f"Acts L{layer}"):
                batch = texts[i : i + bs]
                inp = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.c.max_sequence_length,
                )
                _ = model(**{k: v.to(self.c.device) for k, v in inp.items()})
                if torch.cuda.is_available() and (i // bs) % 2 == 0:
                    torch.cuda.empty_cache()
        h.remove()
        return torch.cat(acts, dim=0)

    def _find_bs(self, model, tok, texts: List[str]) -> int:
        if not torch.cuda.is_available():
            return 128
        lo, hi, best = 128, 2048, 128
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                batch = texts[:mid]
                inp = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.c.max_sequence_length,
                )
                _ = model(**{k: v.to(self.c.device) for k, v in inp.items()})
                best = mid
                lo = mid + 64
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hi = mid - 64
            except Exception:
                # If anything else happens, be conservative
                hi = mid - 64
        return best

    def train(self, model, tok, H: List[str], E: List[str], layer: int) -> JumpReLUSAE:
        texts = H + E
        X = self.extract_layer_acts(model, tok, texts, layer)
        d = X.shape[1]
        sae = JumpReLUSAE(d, self.c.sae_expansion_factor, self.c.sae_l0_target).to(
            self.c.device
        )
        # SAE params in fp32; inputs in fp32; use autocast in forward below
        n = X.shape[0]
        ntr = int(0.9 * n)
        tr = X[:ntr].to(self.c.device, dtype=torch.float32)
        va = X[ntr:].to(self.c.device, dtype=torch.float32)

        opt = torch.optim.AdamW(sae.parameters(), lr=self.c.lr, weight_decay=1e-4)

        steps_per_epoch = max(1, len(tr) // self.c.batch_size)
        total_steps = self.c.training_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < self.c.warmup_steps:
                return step / max(1, self.c.warmup_steps)
            p = (step - self.c.warmup_steps) / max(1, total_steps - self.c.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        sae.train()
        idx = torch.arange(len(tr), device=self.c.device)
        step = 0
        for ep in range(self.c.training_epochs):
            perm = idx[torch.randperm(len(idx))]
            opt.zero_grad()
            acc = 0
            for i in range(0, len(perm), self.c.batch_size):
                b = tr[perm[i : i + self.c.batch_size]]
                # autocast to bf16 in matmul; SAE params remain fp32
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    recon, feats, m = sae(b)
                    recon_loss = F.mse_loss(recon, b)

                # Differentiable sparsity (soft gate) outside autocast - keep in fp32
                shifted = sae._last_shifted  # [B, H], fp32
                soft_gate = torch.sigmoid(shifted / sae.temperature)
                l0_soft = soft_gate.sum(dim=-1).mean()
                l0_loss = F.relu(l0_soft - sae.l0_target) * 1.0
                loss = (recon_loss + l0_loss) / self.c.grad_accum_steps
                loss.backward()
                acc += 1
                if acc % self.c.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                    step += 1
            sae.eval()
            with torch.no_grad():
                _, _, vm = sae(va[: min(256, len(va))])
            sae.train()
            if vm["total_loss"] < self.best:
                self.best = vm["total_loss"]
                torch.save(
                    sae.state_dict(),
                    Path(self.c.ckpt_dir) / f"sae_layer{layer}_best.pth",
                )
            if ep % 25 == 0 or ep == self.c.training_epochs - 1:
                print(
                    f"  Ep {ep:03d} trainL0={m['l0']:.0f} valTot={vm['total_loss']:.4f} "
                    f"LR={sched.get_last_lr()[0]:.2e}"
                )

        ck = Path(self.c.ckpt_dir) / f"sae_layer{layer}_best.pth"
        if ck.exists():
            state = torch.load(ck, map_location=self.c.device)
            sae.load_state_dict(state, strict=False)
            print(f"Loaded best SAE from {ck}")
        else:
            print(" Best checkpoint not found; using current weights.")
        return sae


# -----------------------------
# Feature Analysis
# -----------------------------


class FeatureAnalysis:
    def __init__(self, cfg: Config):
        self.c = cfg

    def extract_features(
        self, sae, model, tok, texts: List[str], layer: int
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        def hook_fn(_, __, out):
            hs = out[0] if isinstance(out, tuple) else out
            last = hs[:, -1, :]
            mean = hs.mean(dim=1)
            pooled = 0.7 * mean + 0.3 * last
            with torch.no_grad():
                dev = next(sae.parameters()).device
                f = sae.encode(pooled.to(dev, dtype=torch.float32))
            feats.append(f.cpu())

        h = model.model.layers[layer].register_forward_hook(hook_fn)
        bs = 1024
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), bs), desc=f"Feats L{layer}"):
                batch = texts[i : i + bs]
                inp = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.c.max_sequence_length,
                )
                _ = model(**{k: v.to(self.c.device) for k, v in inp.items()})
                if torch.cuda.is_available() and (i // bs) % 2 == 0:
                    torch.cuda.empty_cache()
        h.remove()
        return torch.cat(feats, dim=0)

    def stats(self, F_hi: torch.Tensor, F_en: torch.Tensor) -> List[Dict]:
        res = []
        H = F_hi.float().numpy()
        E = F_en.float().numpy()
        for i in range(H.shape[1]):
            hi = H[:, i]
            en = E[:, i]
            t, p = ttest_ind(hi, en, equal_var=False)
            m_hi, m_en = hi.mean(), en.mean()
            v_hi, v_en = hi.var(), en.var()
            n_hi, n_en = len(hi), len(en)
            pooled = np.sqrt(
                ((n_hi - 1) * v_hi + (n_en - 1) * v_en) / max(1, (n_hi + n_en - 2))
            )
            d = (m_hi - m_en) / (pooled + 1e-8)
            res.append(
                {
                    "idx": int(i),
                    "p_value": float(p),
                    "cohens_d": float(d),
                    "hindi_mean": float(m_hi),
                    "english_mean": float(m_en),
                    "hindi_freq": float((hi > 0).mean()),
                    "english_freq": float((en > 0).mean()),
                    "importance": float(abs(d) * (1 - min(1.0, p))),
                }
            )
        return res

    def gradients(self, F_hi: torch.Tensor, F_en: torch.Tensor) -> Dict:
        hi = F_hi.clone().float()
        en = F_en.clone().float()
        allF = torch.cat([hi, en], dim=0).requires_grad_(True)
        labels = torch.cat([torch.zeros(hi.shape[0]), torch.ones(en.shape[0])]).to(
            allF.device
        )
        W = torch.randn(allF.shape[1], 1, device=allF.device, requires_grad=True)
        b = torch.zeros(1, device=allF.device, requires_grad=True)
        logits = (allF @ W).squeeze(-1) + b
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        grads = torch.autograd.grad(
            outputs=loss, inputs=allF, create_graph=False, retain_graph=False
        )[0]
        imp = grads.abs().mean(dim=0)  # [D]
        return {"importance": imp.detach().cpu().tolist()}


# -----------------------------
# Gemini Interpreter
# -----------------------------


class Gemini:
    def __init__(
        self, api_key: Optional[str], model_name: Optional[str], timeout: int = 30
    ):
        self.key = api_key
        self.model = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        self.timeout = timeout
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "x-goog-api-key": self.key or ""}
        )

    def label(self, features: List[Dict], language: str) -> List[str]:
        out = []
        for f in tqdm(features, desc=f"Gemini label {language}"):
            idx = f.get("idx")
            d = f.get("cohens_d")
            hm = f.get("hindi_mean", 0)
            em = f.get("english_mean", 0)
            freq_h = f.get("hindi_freq", 0)
            freq_e = f.get("english_freq", 0)
            top_texts = f.get("top_activating_texts", [])

            # Build enhanced prompt with text snippets
            prompt = (
                f"You are an expert linguist. Interpret a neural feature for {language}.\n"
                f"Feature idx: {idx}\nCohen's d: {d:.3f}\n"
                f"Mean activations -> HI: {hm:.3f}, EN: {em:.3f}\n"
                f"Freq -> HI: {freq_h:.2f}, EN: {freq_e:.2f}\n"
            )

            if top_texts:
                prompt += f"\nTop-activating {language} text snippets:\n"
                for i, text in enumerate(top_texts, 1):
                    prompt += f'{i}. "{text}"\n'
                prompt += (
                    f"\nBased on these examples, explain the likely linguistic pattern "
                    f"(script, morphology, syntax, semantics) and why it's characteristic of {language}. "
                    f"One short paragraph focusing on what these texts have in common."
                )
            else:
                prompt += (
                    f"\nExplain the likely linguistic pattern (script, morphology, syntax, semantics) "
                    f"and why it's characteristic of {language}. One short paragraph."
                )

            text = self._call(prompt)
            out.append(text)
            time.sleep(0.2)
        return out

    def _call(self, prompt: str) -> str:
        if not self.key:
            return "(Gemini API key not set)"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 250},
        }
        try:
            r = self.session.post(self.url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                j = r.json()
                return (
                    j.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
            # Fallback to 1.5 flash if 2.x fails
            if "2.0" in self.model or "2.5" in self.model:
                self.model = "gemini-1.5-flash"
                self.url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model}:generateContent"
                )
                return self._call(prompt)
            return f"(API error {r.status_code})"
        except Exception as e:
            return f"(Error {e})"


# -----------------------------
# Multi-Layer Steering (shadowing, toggles)
# -----------------------------


class MultiLayerSteering:
    def __init__(
        self,
        model,
        tok,
        layer_saes: Dict[int, JumpReLUSAE],
        layer_features_hi: Dict[int, List[int]],
        layer_features_en: Dict[int, List[int]],
        layer_weights: Dict[int, float],
        cfg: Config,
    ):
        self.m = model
        self.tok = tok
        self.saes = layer_saes
        self.hi = layer_features_hi
        self.en = layer_features_en
        self.w = layer_weights
        self.c = cfg
        self._last_tel: Dict[int, Dict[str, float]] = {}

    @staticmethod
    def detect_lang(text: str) -> str:
        dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
        alpha = sum(1 for c in text if c.isalpha())
        if alpha == 0:
            return "unknown"
        return "hindi" if dev / alpha > 0.3 else "english"

    def _make_hook(
        self,
        L: int,
        target: str,
        strength: float,
        mode: str,
        topk: int,
        toggle: Optional[set],
        custom_scales: Optional[Dict[int, float]] = None,
    ):
        sae = self.saes[L]
        wL = self.w.get(L, 0.0)
        tgt = target.lower()
        feat_hi = self.hi[L][:topk]
        feat_en = self.en[L][:topk]
        amp = feat_hi if tgt == "hindi" else feat_en
        sup = feat_en if tgt == "hindi" else feat_hi
        if mode == "shadow_hindi":
            amp = []
            sup = feat_hi
        if mode == "shadow_english":
            amp = []
            sup = feat_en
        if toggle:
            amp = [i for i in amp if i in toggle]
            sup = [i for i in sup if i in toggle]
        sae_dev = next(sae.parameters()).device
        beta = wL  # weight the delta by layer quality

        def hook(_, __, out):
            hs = out[0] if isinstance(out, tuple) else out
            B, T, D = hs.shape
            flat = hs.reshape(-1, D)
            with torch.no_grad():
                f_base = sae.encode(flat.to(sae_dev, dtype=torch.float32))
                f_mod = f_base.clone()
                for i in amp:
                    if 0 <= i < f_mod.shape[1]:
                        f_mod[:, i] *= 1.0 + strength
                for i in sup:
                    if 0 <= i < f_mod.shape[1]:
                        f_mod[:, i] *= max(0.0, 1.0 - strength)

                # Per-feature custom overrides (layer-scoped)
                scales_for_L = (custom_scales or {}).get(L, {})
                if scales_for_L:
                    for i, gain in scales_for_L.items():
                        if 0 <= i < f_mod.shape[1]:
                            # gain = +0.5 -> multiply by 1.5 ; gain = -0.3 -> multiply by 0.7
                            f_mod[:, i] *= max(0.0, 1.0 + float(gain))

                rec_base = sae.decode(f_base)
                rec_mod = sae.decode(f_mod)
                delta = (rec_mod - rec_base).to(hs.dtype).reshape(B, T, D)

                # Telemetry (aggregates)
                amp_idx = torch.tensor(amp, device=f_base.device) if amp else None
                sup_idx = torch.tensor(sup, device=f_base.device) if sup else None
                base_amp = (
                    f_base[:, amp_idx].mean().item()
                    if amp_idx is not None and amp_idx.numel() > 0
                    else 0.0
                )
                mod_amp = (
                    f_mod[:, amp_idx].mean().item()
                    if amp_idx is not None and amp_idx.numel() > 0
                    else 0.0
                )
                base_sup = (
                    f_base[:, sup_idx].mean().item()
                    if sup_idx is not None and sup_idx.numel() > 0
                    else 0.0
                )
                mod_sup = (
                    f_mod[:, sup_idx].mean().item()
                    if sup_idx is not None and sup_idx.numel() > 0
                    else 0.0
                )
                delta_l2 = delta.reshape(-1, D).float().norm(dim=1).mean().item()
                hs_l2 = hs.reshape(-1, D).float().norm(dim=1).mean().item()
                self._last_tel[L] = {
                    "amp_count": len(amp),
                    "sup_count": len(sup),
                    "base_amp": base_amp,
                    "mod_amp": mod_amp,
                    "base_sup": base_sup,
                    "mod_sup": mod_sup,
                    "delta_l2": delta_l2,
                    "hs_l2": hs_l2,
                    "weight": beta,
                }
            new_hs = hs + beta * delta
            return (new_hs,) if isinstance(out, tuple) else new_hs

        return hook

    def generate(
        self,
        prompt: str,
        target: str,
        strength: float,
        mode: str,
        topk: int = 5,
        toggle_ids: Optional[List[str]] = None,
        max_new_tokens: int = 64,
        deterministic: bool = False,
        custom_scales_text: Optional[str] = None,
    ):
        # Build per-layer toggle map from ["L{layer}:{feat} — label...", ...]
        toggle_map: Dict[int, set] = {}
        if toggle_ids:
            for t in toggle_ids:
                if isinstance(t, str):
                    # Keep only the left token "L{layer}:{feat}" before any label text
                    t = t.split(" ", 1)[0].strip()
                if t.startswith("L") and ":" in t:
                    try:
                        lpart, fpart = t[1:].split(":", 1)  # strip leading L
                        L = int(lpart)
                        f = int(fpart)
                        toggle_map.setdefault(L, set()).add(f)
                    except Exception:
                        pass
                elif str(t).isdigit():
                    # apply numeric feature id to all layers (rare use)
                    for L in self.saes.keys():
                        toggle_map.setdefault(L, set()).add(int(t))

        # Parse custom scales like "L18:130821=+0.6; L19:117955=-0.4"
        custom_scales = {}
        if custom_scales_text:
            items = [x.strip() for x in custom_scales_text.split(";") if x.strip()]
            for it in items:
                try:
                    key, val = it.split("=")
                    val = float(val)
                    if key.startswith("L") and ":" in key:
                        Ls, fs = key[1:].split(":")
                        L = int(Ls)
                        f = int(fs)
                        custom_scales.setdefault(L, {})[f] = val
                except Exception:
                    pass

        hooks = []
        try:
            for L in self.saes.keys():
                layer_toggle = toggle_map.get(L, None)
                h = self.m.model.layers[L].register_forward_hook(
                    self._make_hook(
                        L, target, strength, mode, topk, layer_toggle, custom_scales
                    )
                )
                hooks.append(h)
            inp = self.tok(prompt, return_tensors="pt").to(self.c.device)
            with torch.no_grad():
                if deterministic:
                    g = self.m.generate(
                        **inp,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tok.eos_token_id,
                    )
                else:
                    g = self.m.generate(
                        **inp,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tok.eos_token_id,
                    )
            steered = self.tok.decode(g[0], skip_special_tokens=True)
        finally:
            for h in hooks:
                h.remove()
        with torch.no_grad():
            if deterministic:
                b = self.m.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
            else:
                b = self.m.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tok.eos_token_id,
                )
        baseline = self.tok.decode(b[0], skip_special_tokens=True)
        st = steered[len(prompt) :].strip()
        bt = baseline[len(prompt) :].strip()

        # Print steering telemetry
        print("Steering telemetry per layer:")
        for L, t in self._last_tel.items():
            print(
                f"  L{L}: amp={t['amp_count']} sup={t['sup_count']} "
                f"amp_mean {t['base_amp']:.4f}->{t['mod_amp']:.4f} "
                f"sup_mean {t['base_sup']:.4f}->{t['mod_sup']:.4f} "
                f"deltaL2={t['delta_l2']:.4f} hsL2={t['hs_l2']:.4f} w={t['weight']:.3f}"
            )

        return {
            "steered": st,
            "baseline": bt,
            "steered_lang": self.detect_lang(st),
            "baseline_lang": self.detect_lang(bt),
        }


class Steering:
    def __init__(self, model, tok, sae, layer: int, cfg: Config):
        self.m = model
        self.tok = tok
        self.sae = sae
        self.layer = layer
        self.c = cfg
        self.custom_feature_set: Optional[List[int]] = None

    @staticmethod
    def detect_lang(text: str) -> str:
        dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
        alpha = sum(1 for c in text if c.isalpha())
        if alpha == 0:
            return "unknown"
        return "hindi" if dev / alpha > 0.3 else "english"

    def _hook(self, target, strength, mode, feats_hi, feats_en, toggle_set):
        tgt = target.lower()
        amp = feats_hi if tgt == "hindi" else feats_en
        sup = feats_en if tgt == "hindi" else feats_hi
        if mode == "shadow_hindi":
            amp = []
            sup = feats_hi
        if mode == "shadow_english":
            amp = []
            sup = feats_en

        if toggle_set:
            amp = [i for i in amp if i in toggle_set]
            sup = [i for i in sup if i in toggle_set]

        sae_dev = next(self.sae.parameters()).device
        beta = 1.0  # set to 0.5 if you want to be more conservative

        def fn(_, __, out):
            hs = out[0] if isinstance(out, tuple) else out  # [B, T, D], bf16
            B, T, D = hs.shape
            flat = hs.reshape(-1, D)

            with torch.no_grad():
                f_base = self.sae.encode(
                    flat.to(sae_dev, dtype=torch.float32)
                )  # [-1, H]
                f_mod = f_base.clone()

                for i in amp[: self.c.top_k_features]:
                    if 0 <= i < f_mod.shape[1]:
                        f_mod[:, i] *= 1.0 + strength
                for i in sup[: self.c.top_k_features]:
                    if 0 <= i < f_mod.shape[1]:
                        f_mod[:, i] *= max(0.0, 1.0 - strength)

                rec_base = self.sae.decode(f_base)  # [-1, D] fp32
                rec_mod = self.sae.decode(f_mod)  # [-1, D] fp32
                delta = rec_mod - rec_base  # fp32
                delta = delta.to(hs.dtype).reshape(B, T, D)  # bf16

            new_hs = hs + beta * delta
            return (new_hs,) if isinstance(out, tuple) else new_hs

        return fn

    def generate(
        self,
        prompt: str,
        target: str,
        strength: float,
        mode: str,
        feats_hi: List[int],
        feats_en: List[int],
        toggle_set: Optional[List[int]] = None,
        max_new_tokens: int = 64,
    ) -> Dict:
        hook_fn = self._hook(target, strength, mode, feats_hi, feats_en, toggle_set)
        h = self.m.model.layers[self.layer].register_forward_hook(hook_fn)
        try:
            inp = self.tok(prompt, return_tensors="pt").to(self.c.device)
            with torch.no_grad():
                gen = self.m.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tok.eos_token_id,
                )
            steered = self.tok.decode(gen[0], skip_special_tokens=True)
        finally:
            h.remove()
        with torch.no_grad():
            base = self.m.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tok.eos_token_id,
            )
        baseline = self.tok.decode(base[0], skip_special_tokens=True)
        st = steered[len(prompt) :].strip()
        bt = baseline[len(prompt) :].strip()
        return {
            "steered": st,
            "baseline": bt,
            "steered_lang": self.detect_lang(st),
            "baseline_lang": self.detect_lang(bt),
        }


# -----------------------------
# Orchestration
# -----------------------------


class Pipeline:
    def __init__(self, cfg: Config):
        self.c = cfg
        set_seed(42)
        self.access = Environment.setup()

        # Load model
        tok = AutoTokenizer.from_pretrained(self.c.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.c.model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.to(self.c.device)
        self.tok, self.model = tok, model

        # Derive layer scan range from model if requested
        if self.c.scan_all_layers:
            try:
                n_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
            except Exception:
                n_layers = 0
            if n_layers and n_layers > 0:
                # scan all layers [0..n_layers-1]
                self.c.layer_range = (0, n_layers)

    def run(self):
        c = self.c
        start_time = time.time()

        # Load datasets
        print("Loading datasets...")
        loader = DatasetLoader(c, self.access)
        HI, EN = loader.load()
        print(f"Dataset loading completed in {time.time() - start_time:.1f}s")

        # Layer discovery (objective consensus)
        print("Starting layer discovery...")
        layer_start = time.time()
        discover = LayerDiscovery(self.model, self.tok, c)
        sel_layers, layer_consensus = discover.run(HI, EN)
        print(f"Layer discovery completed in {time.time() - layer_start:.1f}s")

        # Train a SAE per selected layer
        print("Starting SAE training...")
        sae_start = time.time()
        trainer = SAETrainer(c)
        FA = FeatureAnalysis(c)
        layer_saes = {}
        layer_feats = {}
        for L in sel_layers:
            print(f"\nTraining SAE for layer {L}")
            saeL = trainer.train(self.model, self.tok, HI, EN, L)
            layer_saes[L] = saeL
            print(f"  Extracting features for layer {L}")
            F_hi = FA.extract_features(saeL, self.model, self.tok, HI[:2000], L)
            F_en = FA.extract_features(saeL, self.model, self.tok, EN[:2000], L)
            layer_feats[L] = (F_hi, F_en)
        print(f"SAE training completed in {time.time() - sae_start:.1f}s")

        # Stats + gradients per layer; collect union for steering
        print("Starting feature analysis...")
        analysis_start = time.time()
        union_hi, union_en = [], []
        for L, (F_hi, F_en) in layer_feats.items():
            stats = FA.stats(F_hi, F_en)
            if c.bonferroni:
                thr = c.significance_threshold / max(1, F_hi.shape[1])
            else:
                thr = c.significance_threshold
            hi_feats = [
                s
                for s in stats
                if s["p_value"] < thr and s["cohens_d"] > c.min_effect_size
            ]
            en_feats = [
                s
                for s in stats
                if s["p_value"] < thr and s["cohens_d"] < -c.min_effect_size
            ]
            hi_feats.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
            en_feats.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
            grads = FA.gradients(F_hi, F_en)
            gimp = np.array(grads["importance"])

            def re_rank(feats):
                idxs = [f["idx"] for f in feats[:200]]
                pairs = [(i, float(gimp[i])) for i in idxs]
                pairs.sort(key=lambda x: x[1], reverse=True)
                return [i for i, _ in pairs]

            top_hi = re_rank(hi_feats)
            top_en = re_rank(en_feats)

            # Fallback if statistical sets are empty
            if not hi_feats:
                # choose top 200 gradient features with hindi preference
                means_hi = F_hi.float().mean(dim=0).cpu().numpy()
                means_en = F_en.float().mean(dim=0).cpu().numpy()
                pref = means_hi > means_en
                top_idx = np.argsort(gimp)[::-1]  # sort all by gradient
                top_hi = [int(i) for i in top_idx if pref[i]][:200]
            if not en_feats:
                means_hi = F_hi.float().mean(dim=0).cpu().numpy()
                means_en = F_en.float().mean(dim=0).cpu().numpy()
                pref = means_en > means_hi
                top_idx = np.argsort(gimp)[::-1]
                top_en = [int(i) for i in top_idx if pref[i]][:200]

            # FINAL GUARANTEE: if still empty, take top by gradient anyway
            if len(top_hi) == 0:
                top_hi = [int(i) for i in np.argsort(gimp)[::-1][:200]]
            if len(top_en) == 0:
                top_en = [int(i) for i in np.argsort(gimp)[::-1][:200]]

            union_hi.append((L, top_hi))
            union_en.append((L, top_en))
        print(f"Feature analysis completed in {time.time() - analysis_start:.1f}s")

        # --- Build Gemini labels for multiple layers (limit to top_n per layer) ---
        top_n_labels = 10  # per lang per layer; adjust as needed to control cost
        label_map: Dict[Tuple[int, int], str] = {}

        def build_stats_index_map(stats_list):
            return {s["idx"]: s for s in stats_list}

        print("Building Gemini labels for all selected layers...")
        for L, (F_hi, F_en) in layer_feats.items():
            stats_all = FA.stats(F_hi, F_en)
            stats_idx = build_stats_index_map(stats_all)

            # Filter for significant features (abs(d) >= 0.3) first
            hi_significant = [
                s
                for s in stats_all
                if s["idx"] in union_hi[[l for l, _ in union_hi].index(L)][1]
                and abs(s["cohens_d"]) >= 0.3
            ]
            en_significant = [
                s
                for s in stats_all
                if s["idx"] in union_en[[l for l, _ in union_en].index(L)][1]
                and abs(s["cohens_d"]) >= 0.3
            ]

            # Sort by Cohen's d magnitude and take top features
            hi_significant.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
            en_significant.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

            # If not enough significant features, backfill with gradient importance
            hi_top = [s["idx"] for s in hi_significant[:top_n_labels]]
            en_top = [s["idx"] for s in en_significant[:top_n_labels]]

            if len(hi_top) < top_n_labels:
                # Get remaining features by gradient importance
                grads = FA.gradients(F_hi, F_en)
                gimp = np.array(grads["importance"])
                remaining_hi = [
                    i
                    for i in union_hi[[l for l, _ in union_hi].index(L)][1]
                    if i not in hi_top
                ]
                remaining_hi.sort(key=lambda x: gimp[x], reverse=True)
                hi_top.extend(remaining_hi[: top_n_labels - len(hi_top)])

            if len(en_top) < top_n_labels:
                grads = FA.gradients(F_hi, F_en)
                gimp = np.array(grads["importance"])
                remaining_en = [
                    i
                    for i in union_en[[l for l, _ in union_en].index(L)][1]
                    if i not in en_top
                ]
                remaining_en.sort(key=lambda x: gimp[x], reverse=True)
                en_top.extend(remaining_en[: top_n_labels - len(en_top)])

            # Build enhanced stats with text snippets for Gemini
            hi_stats_subset = []
            en_stats_subset = []

            # Get text samples for activation analysis
            hi_texts_sample = HI[: min(1000, len(HI))]  # Use subset for efficiency
            en_texts_sample = EN[: min(1000, len(EN))]

            for idx in hi_top:
                if idx in stats_idx:
                    stat = stats_idx[idx].copy()

                    # Find top-activating Hindi texts for this feature
                    with torch.no_grad():
                        # Extract activations for this feature across sample texts
                        acts = []
                        for i in range(0, len(hi_texts_sample), 64):  # Small batches
                            batch = hi_texts_sample[i : i + 64]
                            inp = self.tok(
                                batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=c.max_sequence_length,
                            )
                            inp = {k: v.to(c.device) for k, v in inp.items()}

                            def hook_fn(_, __, out):
                                hs = out[0] if isinstance(out, tuple) else out
                                last = hs[:, -1, :]
                                mean = hs.mean(dim=1)
                                pooled = 0.7 * mean + 0.3 * last
                                with torch.no_grad():
                                    dev = next(layer_saes[L].parameters()).device
                                    f = layer_saes[L].encode(
                                        pooled.to(dev, dtype=torch.float32)
                                    )
                                acts.extend(f[:, idx].cpu().numpy())

                            h = self.model.model.layers[L].register_forward_hook(
                                hook_fn
                            )
                            _ = self.model(**inp)
                            h.remove()

                    # Get top 3-5 activating texts
                    if acts:
                        top_indices = np.argsort(acts)[-5:][
                            ::-1
                        ]  # Top 5, highest first
                        top_texts = [
                            hi_texts_sample[i]
                            for i in top_indices
                            if i < len(hi_texts_sample)
                        ]
                        stat["top_activating_texts"] = [
                            t[:200] for t in top_texts[:3]
                        ]  # First 200 chars of top 3
                    else:
                        stat["top_activating_texts"] = []

                    hi_stats_subset.append(stat)

            for idx in en_top:
                if idx in stats_idx:
                    stat = stats_idx[idx].copy()

                    # Find top-activating English texts for this feature
                    with torch.no_grad():
                        acts = []
                        for i in range(0, len(en_texts_sample), 64):
                            batch = en_texts_sample[i : i + 64]
                            inp = self.tok(
                                batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=c.max_sequence_length,
                            )
                            inp = {k: v.to(c.device) for k, v in inp.items()}

                            def hook_fn(_, __, out):
                                hs = out[0] if isinstance(out, tuple) else out
                                last = hs[:, -1, :]
                                mean = hs.mean(dim=1)
                                pooled = 0.7 * mean + 0.3 * last
                                with torch.no_grad():
                                    dev = next(layer_saes[L].parameters()).device
                                    f = layer_saes[L].encode(
                                        pooled.to(dev, dtype=torch.float32)
                                    )
                                acts.extend(f[:, idx].cpu().numpy())

                            h = self.model.model.layers[L].register_forward_hook(
                                hook_fn
                            )
                            _ = self.model(**inp)
                            h.remove()

                    if acts:
                        top_indices = np.argsort(acts)[-5:][::-1]
                        top_texts = [
                            en_texts_sample[i]
                            for i in top_indices
                            if i < len(en_texts_sample)
                        ]
                        stat["top_activating_texts"] = [t[:200] for t in top_texts[:3]]
                    else:
                        stat["top_activating_texts"] = []

                    en_stats_subset.append(stat)

            # Call Gemini (handle empty subsets)
            hi_texts = (
                Gemini(
                    os.getenv("GEMINI_API_KEY"), c.gemini_model, c.gemini_timeout
                ).label(hi_stats_subset, "Hindi")
                if hi_stats_subset and os.getenv("GEMINI_API_KEY")
                else []
            )
            en_texts = (
                Gemini(
                    os.getenv("GEMINI_API_KEY"), c.gemini_model, c.gemini_timeout
                ).label(en_stats_subset, "English")
                if en_stats_subset and os.getenv("GEMINI_API_KEY")
                else []
            )

            # Map back to (L, idx)
            for idx, txt in zip(hi_top[: len(hi_texts)], hi_texts):
                label_map[(L, idx)] = (txt or "").strip()
            for idx, txt in zip(en_top[: len(en_texts)], en_texts):
                label_map[(L, idx)] = (txt or "").strip()

        # Normalize layer consensus to weights for steering blend
        ssum = sum(layer_consensus.values()) or 1.0
        layer_weights = {
            L: float(layer_consensus.get(L, 0.0) / ssum) for L in sel_layers
        }

        # Build multi-layer steering maps BEFORE Gemini section
        layer_hi_map = {L: feats for (L, feats) in union_hi}
        layer_en_map = {L: feats for (L, feats) in union_en}

        # Gemini interpretation on BEST layer only (first is the best by consensus)
        print("Starting Gemini interpretation...")
        gemini_start = time.time()
        gemini = Gemini(os.getenv("GEMINI_API_KEY"), c.gemini_model, c.gemini_timeout)
        best_layer = sel_layers[0]
        best_hi_idx = {i for i in layer_hi_map[best_layer][:10]}
        best_en_idx = {i for i in layer_en_map[best_layer][:10]}
        best_stats_all = FA.stats(*layer_feats[best_layer])
        best_hi_stats = [s for s in best_stats_all if s["idx"] in best_hi_idx]
        best_en_stats = [s for s in best_stats_all if s["idx"] in best_en_idx]
        hi_labels = (
            gemini.label(best_hi_stats[:10], "Hindi")
            if os.getenv("GEMINI_API_KEY")
            else []
        )
        en_labels = (
            gemini.label(best_en_stats[:10], "English")
            if os.getenv("GEMINI_API_KEY")
            else []
        )
        print(f"Gemini interpretation completed in {time.time() - gemini_start:.1f}s")

        # Save results
        per_layer = {}
        for L, (F_hi, F_en) in layer_feats.items():
            # we already computed union features per layer earlier in union_hi/union_en
            # build maps for convenience
            per_layer[L] = {
                "top_hindi_features": next(
                    (feats for (l, feats) in union_hi if l == L), []
                )[:50],
                "top_english_features": next(
                    (feats for (l, feats) in union_en if l == L), []
                )[:50],
                "sae_checkpoint": str(Path(self.c.ckpt_dir) / f"sae_layer{L}_best.pth"),
            }
        # Save label_map snippet in results for reproducibility (optional)
        # (Serialize as strings "L{layer}:{idx}" -> text)
        labels_serializable = {f"L{L}:{i}": t for (L, i), t in label_map.items()}

        out = {
            "config": asdict(c),
            "selected_layers": sel_layers,
            "layer_weights": layer_weights,  # consensus-normalized weights
            "per_layer": per_layer,
            "gemini_labels": labels_serializable,
            "gemini_interpretations": {
                "best_layer": best_layer,
                "best_hindi_feature_stats_top10": best_hi_stats[:10],
                "best_english_feature_stats_top10": best_en_stats[:10],
                "hindi_labels": hi_labels,
                "english_labels": en_labels,
            },
        }
        with open(Path(c.out_dir) / "results.json", "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {Path(c.out_dir) / 'results.json'}")

        # Build multi-layer steering
        msteer = MultiLayerSteering(
            self.model,
            self.tok,
            layer_saes,
            layer_hi_map,
            layer_en_map,
            layer_weights,
            c,
        )

        # Shadowing evaluation (suppress Hindi)
        print("Starting shadowing evaluation...")
        eval_start = time.time()
        test_prompts = [
            "The capital of France is",
            "What is 12 times 17?",
            "Explain photosynthesis briefly.",
            "यह वाक्य हिंदी में है, देखें क्या होता है",
            "Tell me about the Indian National Congress.",
        ]
        ok, trials = 0, 0
        eval_records = []
        for p in test_prompts:
            r = msteer.generate(
                p,
                target="English",
                strength=c.default_strength,
                mode="shadow_hindi",
                topk=c.top_k_features,
                max_new_tokens=48,
                deterministic=True,  # For debugging - remove sampling noise
            )
            trials += 1
            if r["steered_lang"] == "english":
                ok += 1

            # Copy per-layer telemetry
            tel_copy = {str(L): dict(t) for L, t in msteer._last_tel.items()}
            eval_records.append(
                {
                    "prompt": p,
                    "steered": r["steered"],
                    "baseline": r["baseline"],
                    "steered_lang": r["steered_lang"],
                    "baseline_lang": r["baseline_lang"],
                    "telemetry": tel_copy,
                }
            )

            print(
                f"\nPrompt: {p}\n"
                f" Steered({r['steered_lang']}): {r['steered'][:180]}\n"
                f" Baseline({r['baseline_lang']}): {r['baseline'][:180]}"
            )

        print(f"\nShadowing score (english after shadow_hindi): {ok}/{trials}")
        print(f"Shadowing evaluation completed in {time.time() - eval_start:.1f}s")

        # Save evaluation telemetry for report generation
        eval_obj = {
            "mode": "shadow_hindi",
            "target": "english",
            "records": eval_records,
        }
        with open(Path(c.out_dir) / "evaluation_results.json", "w") as f:
            json.dump(eval_obj, f, ensure_ascii=False, indent=2)
        print(
            f"Saved evaluation telemetry to {Path(c.out_dir) / 'evaluation_results.json'}"
        )

        # Timing information
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"\nTotal pipeline time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Time breakdown:")
        print(f"   Dataset loading: {layer_start - start_time:.1f}s")
        print(f"   Layer discovery: {sae_start - layer_start:.1f}s")
        print(f"   SAE training: {analysis_start - sae_start:.1f}s")
        print(f"   Feature analysis: {gemini_start - analysis_start:.1f}s")
        print(f"   Gemini interpretation: {eval_start - gemini_start:.1f}s")
        print(f"   Shadowing evaluation: {time.time() - eval_start:.1f}s")

        # Launch Gradio UI (play with features)
        self._launch_ui(msteer, union_hi, union_en, label_map)

    def _launch_ui(
        self,
        msteer: MultiLayerSteering,
        union_hi: List[Tuple[int, List[int]]],
        union_en: List[Tuple[int, List[int]]],
        label_map: Dict[Tuple[int, int], str],
    ):
        c = self.c

        # Prepare feature choices for UI
        all_hi_feats = []
        all_en_feats = []

        def pretty_label(L, idx):
            base = f"L{L}:{idx}"
            desc = label_map.get((L, idx), "")
            if desc:
                desc = desc.replace("\n", " ").strip()
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                return f"{base} — {desc}"
            return base

        for L, feats in union_hi:
            all_hi_feats.extend([pretty_label(L, f) for f in feats[:15]])

        for L, feats in union_en:
            all_en_feats.extend([pretty_label(L, f) for f in feats[:15]])

        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Hindi-English Language Steering",
            css="""
            /* Only target specific sections that need black text */
            .gradio-container .contain .example-prompts,
            .gradio-container .contain .example-prompts *,
            .gradio-container .contain .example-prompts h1,
            .gradio-container .contain .example-prompts h2,
            .gradio-container .contain .example-prompts h3,
            .gradio-container .contain .example-prompts h4,
            .gradio-container .contain .example-prompts h5,
            .gradio-container .contain .example-prompts h6,
            .gradio-container .contain .example-prompts p,
            .gradio-container .contain .example-prompts li,
            .gradio-container .contain .example-prompts ul,
            .gradio-container .contain .example-prompts ol,
            .gradio-container .contain .example-prompts strong,
            .gradio-container .contain .example-prompts em,
            .gradio-container .contain .example-prompts span,
            .gradio-container .contain .example-prompts div {
                color: #000 !important;
            }

            .gradio-container .contain .troubleshooting,
            .gradio-container .contain .troubleshooting *,
            .gradio-container .contain .troubleshooting h1,
            .gradio-container .contain .troubleshooting h2,
            .gradio-container .contain .troubleshooting h3,
            .gradio-container .contain .troubleshooting h4,
            .gradio-container .contain .troubleshooting h5,
            .gradio-container .contain .troubleshooting h6,
            .gradio-container .contain .troubleshooting p,
            .gradio-container .contain .troubleshooting li,
            .gradio-container .contain .troubleshooting ul,
            .gradio-container .contain .troubleshooting ol,
            .gradio-container .contain .troubleshooting strong,
            .gradio-container .contain .troubleshooting em,
            .gradio-container .contain .troubleshooting span,
            .gradio-container .contain .troubleshooting div {
                color: #000 !important;
            }

            /* Target mode explanation sections */
            .gradio-container .contain .mode-explanation,
            .gradio-container .contain .mode-explanation *,
            .gradio-container .contain .mode-explanation h1,
            .gradio-container .contain .mode-explanation h2,
            .gradio-container .contain .mode-explanation h3,
            .gradio-container .contain .mode-explanation h4,
            .gradio-container .contain .mode-explanation h5,
            .gradio-container .contain .mode-explanation h6,
            .gradio-container .contain .mode-explanation p,
            .gradio-container .contain .mode-explanation li,
            .gradio-container .contain .mode-explanation ul,
            .gradio-container .contain .mode-explanation ol,
            .gradio-container .contain .mode-explanation strong,
            .gradio-container .contain .mode-explanation em,
            .gradio-container .contain .mode-explanation span,
            .gradio-container .contain .mode-explanation div {
                color: #000 !important;
            }

            /* Keep original styling for other elements */
            .main-header { text-align: center; margin-bottom: 2rem; }
            .section-header { margin-top: 1.5rem; margin-bottom: 1rem; }
            .example-prompts { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
            .troubleshooting { background: #fff3cd; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
            .feature-toggle { max-height: 200px; overflow-y: auto; }
            """,
        ) as demo:

            # Header
            gr.HTML(
                """
            <div class="main-header">
                <h1>Hindi-English Language Steering</h1>
                <p style="color: #666; font-size: 1.1em;">Multi-layer feature manipulation on Llama-3.1-8B</p>
            </div>
            """
            )

            with gr.Row():
                # Left Column - Controls
                with gr.Column(scale=1):

                    # Basic Controls Section
                    gr.HTML('<div class="section-header"><h3>Basic Controls</h3></div>')

                    with gr.Group():
                        prompt = gr.Textbox(
                            label="📝 Input Prompt",
                            value="The weather today is",
                            lines=3,
                            placeholder="Enter your prompt here...",
                        )

                        with gr.Row():
                            target = gr.Dropdown(
                                ["Hindi", "English"],
                                value="Hindi",
                                label="Target Language",
                                scale=1,
                            )
                            mode = gr.Dropdown(
                                ["steer", "shadow_hindi", "shadow_english"],
                                value="steer",
                                label="Mode",
                                scale=1,
                            )

                    # Advanced Controls Section
                    gr.HTML(
                        '<div class="section-header"><h3>Advanced Controls</h3></div>'
                    )

                    with gr.Group():
                        with gr.Row():
                            strength = gr.Slider(
                                0.5,
                                2.0,
                                value=c.default_strength,
                                step=0.1,
                                label="Strength",
                                scale=1,
                            )
                            topk = gr.Slider(
                                1,
                                10,
                                value=c.top_k_features,
                                step=1,
                                label="Top-k Features",
                                scale=1,
                            )

                        max_tokens = gr.Slider(
                            16, 200, value=64, step=8, label="Max New Tokens"
                        )

                    # Feature Controls Section
                    gr.HTML(
                        '<div class="section-header"><h3>Feature Controls</h3></div>'
                    )

                    with gr.Group():
                        with gr.Tabs():
                            with gr.Tab("Hindi Features"):
                                hi_check = gr.CheckboxGroup(
                                    choices=all_hi_feats[:50],
                                    label="Select Hindi Features",
                                    elem_classes=["feature-toggle"],
                                )

                            with gr.Tab("English Features"):
                                en_check = gr.CheckboxGroup(
                                    choices=all_en_feats[:50],
                                    label="Select English Features",
                                    elem_classes=["feature-toggle"],
                                )

                        custom_gains = gr.Textbox(
                            label="Custom Feature Gains",
                            placeholder="L18:130821=+0.6; L19:117955=-0.4",
                            lines=2,
                            info="Format: L{layer}:{feature}={gain}; separate multiple with semicolons",
                        )

                    # Quick Examples Section
                    gr.HTML('<div class="section-header"><h3>Quick Examples</h3></div>')

                    gr.HTML(
                        """
                    <div class="example-prompts">
                        <h4>Try these prompts:</h4>
                        <ul>
                            <li><strong>English:</strong> "The weather today is", "What is the capital of France?"</li>
                            <li><strong>Hindi:</strong> "आज का मौसम", "फ्रांस की राजधानी क्या है?"</li>
                            <li><strong>Mixed:</strong> "Hello नमस्ते", "I love भारत"</li>
                        </ul>
                    </div>
                    """
                    )

                    # Generate Button
                    go = gr.Button("Generate", variant="primary", size="lg")

                # Right Column - Results
                with gr.Column(scale=2):

                    # Results Section
                    gr.HTML('<div class="section-header"><h3>Results</h3></div>')

                    with gr.Tabs():
                        with gr.Tab("Steered Output"):
                            steered = gr.Markdown(
                                value="**Steered Output will appear here...**",
                                label="Steered Generation",
                            )

                        with gr.Tab("Baseline Output"):
                            baseline = gr.Markdown(
                                value="**Baseline Output will appear here...**",
                                label="Baseline Generation",
                            )

                        with gr.Tab("Comparison"):
                            comparison = gr.Markdown(
                                value="**Side-by-side comparison will appear here...**",
                                label="Side-by-Side Comparison",
                            )

                    # Generation Info
                    info = gr.Markdown(
                        value="**Generation information will appear here...**",
                        label="Generation Info",
                    )

            # Mode Explanations
            gr.HTML('<div class="section-header"><h3>Mode Explanations</h3></div>')

            with gr.Row():
                with gr.Column():
                    gr.HTML(
                        """
                    <div style="background: #e7f3ff; padding: 1rem; border-radius: 8px; margin: 0.5rem; color: #000;">
                        <h4><strong>steer</strong></h4>
                        <p>Full steering: amplifies target language features AND suppresses the other language. Best for strong language switching.</p>
                    </div>
                    """
                    )

                with gr.Column():
                    gr.HTML(
                        """
                    <div style="background: #fff2e7; padding: 1rem; border-radius: 8px; margin: 0.5rem; color: #000;">
                        <h4><strong>shadow_hindi</strong></h4>
                        <p>Shadow mode: suppresses ONLY Hindi features. Perfect for forcing English output from Hindi prompts.</p>
                    </div>
                    """
                    )

                with gr.Column():
                    gr.HTML(
                        """
                    <div style="background: #f0fff4; padding: 1rem; border-radius: 8px; margin: 0.5rem; color: #000;">
                        <h4><strong>shadow_english</strong></h4>
                        <p>Shadow mode: suppresses ONLY English features. Perfect for forcing Hindi output from English prompts.</p>
                    </div>
                    """
                    )

            # Troubleshooting Section
            gr.HTML('<div class="section-header"><h3>Troubleshooting</h3></div>')

            gr.HTML(
                """
            <div class="troubleshooting">
                <h4>Common Issues & Solutions:</h4>
                <ul>
                    <li><strong>Same outputs?</strong> Try increasing strength (1.2-1.6) or top-k features (5-7)</li>
                    <li><strong>Force Hindi?</strong> Use Target=Hindi, Mode=steer, Strength=1.5-2.0</li>
                    <li><strong>Force English?</strong> Use Target=English, Mode=shadow_hindi, Strength=1.2-1.6</li>
                    <li><strong>Slow generation?</strong> Normal for research-grade multi-layer steering</li>
                </ul>
            </div>
            """
            )

            # Event Handlers
            def run_ui(
                prompt,
                target,
                mode,
                strength,
                topk,
                max_tokens,
                hi_ids,
                en_ids,
                custom_gains,
            ):
                if not prompt.strip():
                    return (
                        "**Please enter a prompt**",
                        "**Please enter a prompt**",
                        "**Please enter a prompt**",
                        "**No prompt provided**",
                    )

                try:
                    res = msteer.generate(
                        prompt,
                        target=target,
                        strength=float(strength),
                        mode=mode,
                        topk=int(topk),
                        toggle_ids=(hi_ids or []) + (en_ids or []),
                        max_new_tokens=int(max_tokens),
                        custom_scales_text=custom_gains,
                    )

                    # Format outputs
                    steered_text = f"**Steered Output ({res['steered_lang']}):**\n\n{res['steered']}"
                    baseline_text = f"**Baseline Output ({res['baseline_lang']}):**\n\n{res['baseline']}"

                    # Side-by-side comparison
                    comparison_text = f"""
                    | **Steered ({res['steered_lang']})** | **Baseline ({res['baseline_lang']})** |
                    |---|---|
                    | {res['steered']} | {res['baseline']} |
                    """

                    # Generation info
                    info_text = f"""
                    **Configuration:**
                    - Target: {target} | Mode: {mode} | Strength: {strength:.2f}
                    - Top-k Features: {topk} | Max Tokens: {max_tokens}

                    **Results:**
                    - Steered Language: {res['steered_lang']} | Baseline Language: {res['baseline_lang']}
                    - Feature Toggles: {len(hi_ids or [])} Hindi, {len(en_ids or [])} English
                    """

                    return steered_text, baseline_text, comparison_text, info_text

                except Exception as e:
                    error_msg = f"**Error:** {str(e)}"
                    return (
                        error_msg,
                        error_msg,
                        error_msg,
                        f"**Generation failed:** {str(e)}",
                    )

            # Connect the generate button
            go.click(
                fn=run_ui,
                inputs=[
                    prompt,
                    target,
                    mode,
                    strength,
                    topk,
                    max_tokens,
                    hi_check,
                    en_check,
                    custom_gains,
                ],
                outputs=[steered, baseline, comparison, info],
            )

        print("Launching improved UI (public link enabled)")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Quick run")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass

    cfg = Config()
    if args.debug:
        cfg.debug = True
        cfg.samples_per_language = 600
        cfg.training_epochs = 60
        cfg.layer_range = (18, 20)  # fewer layers
        cfg.batch_size = 512
        cfg.max_sequence_length = 192
        cfg.scan_all_layers = False
        cfg.bonferroni = False
        cfg.min_effect_size = 0.3

    # Gemini model name from env, with safe default/fallbacks
    cfg.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    pipe = Pipeline(cfg)
    pipe.run()
