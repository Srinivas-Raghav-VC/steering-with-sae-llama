#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrated language detection utilities shared across the repo.

Provides:
- classify_language(text) -> (lang, confidence)
- devanagari_ratio(text)
- normalized_edit_distance(a, b)

Uses a robust heuristic with optional Gemini 2.0 Flash fallback when GEMINI_API_KEY
is configured. Safe to import without the API; it will use heuristics only.
"""
from __future__ import annotations

import json
import os
import re

# Simple process-wide cache with optional on-disk persistence
_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_PATH = os.getenv("DETECTOR_CACHE", os.path.join("he_pipeline_results", "detector_cache.json"))
try:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r", encoding="utf-8") as _cf:
            _loaded = json.load(_cf)
            if isinstance(_loaded, dict):
                # Ensure types
                for k, v in _loaded.items():
                    if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str):
                        _CACHE[k] = (v[0], float(v[1]))
except Exception:
    pass
from typing import Optional, Tuple
import difflib

_DEV_RE = re.compile(r"[\u0900-\u097F]")
_WORD_RE = re.compile(r"[A-Za-z']+")
_ROMAN_HI_MARKERS = set(
    """
hai hain hoon raha rahe rahi mera meri mere kya nahi ka ki ke mein hum aap tum
bhai yaar ghar pyar tera tere tumhara tumhari kuch bahut accha theek
""".split()
)

TEXT_CHANGE_THRESHOLD: float = 0.05


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    dev = sum(1 for ch in text if _DEV_RE.match(ch))
    alpha = sum(1 for ch in text if ch.isalpha())
    return dev / max(1, alpha)


def romanized_hindi_ratio(text: str) -> float:
    tokens = _WORD_RE.findall((text or "").lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _ROMAN_HI_MARKERS)
    return hits / len(tokens)


def english_marker_ratio(text: str) -> float:
    english_markers = {
        "the",
        "and",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "been",
        "will",
        "would",
        "could",
        "should",
        "this",
        "that",
        "these",
        "those",
        "which",
        "what",
        "when",
        "where",
    }
    tokens = _WORD_RE.findall((text or "").lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in english_markers)
    return hits / len(tokens)


class _GeminiFlashDetector:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.enabled = False
        self.model_name = model
        self.model = None
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.enabled = True
        except Exception:
            self.enabled = False

    def classify(self, text: str) -> Optional[Tuple[str, float]]:
        if not self.enabled or not text or not text.strip():
            return None
        key = text.strip()[:1200]
        if key in _CACHE:
            return _CACHE[key]
        payload = json.dumps({"text": key}, ensure_ascii=False)
        prompt = (
            "You are an expert linguist. Classify the primary language of the JSON 'text' value below.\n"
            "Respond with one token: english, hindi, mixed, unknown.\n\n"
            f"Payload:\n{payload}\n\n"
            "Rules: english (>70% English), hindi (>70% Hindi/Devanagari), mixed (30-70% each), unknown (other/short).\n"
            "Answer:"
        )
        try:
            resp = self.model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 10})
            out = (getattr(resp, "text", "") or "").strip().lower()
            conf = {"english": 0.9, "hindi": 0.9, "mixed": 0.75, "unknown": 0.4}.get(out)
            if conf is not None:
                _CACHE[key] = (out, conf)
                try:
                    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
                    tmp_path = _CACHE_PATH + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as _cf:
                        json.dump(_CACHE, _cf, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, _CACHE_PATH)
                except Exception:
                    pass
                return (out, conf)
            if "english" in out:
                _CACHE[key] = ("english", 0.7)
                return _CACHE[key]
            if "hindi" in out:
                _CACHE[key] = ("hindi", 0.7)
                return _CACHE[key]
            if "mixed" in out:
                _CACHE[key] = ("mixed", 0.6)
                return _CACHE[key]
            _CACHE[key] = ("unknown", 0.3)
            return _CACHE[key]
        except Exception:
            return None


_FLASH = _GeminiFlashDetector()


def gemini_enabled() -> bool:
    """Return True iff the Gemini fallback is active and not force-disabled."""
    disabled_via_env = os.getenv("DISABLE_GEMINI_DETECTOR")
    return _FLASH.enabled and not disabled_via_env


def classify_language(text: str) -> Tuple[str, float]:
    """Return (language, confidence). Languages: english, hindi, mixed, unknown."""
    text = (text or "").strip()
    if not text:
        return ("unknown", 0.0)

    # Preferred: Gemini Flash (unless disabled via env/availability)
    if gemini_enabled():
        guess = _FLASH.classify(text)
        if guess is not None and guess[1] >= 0.55:
            return (guess[0], float(guess[1]))

    # Heuristics fallback
    tokens = text.split()
    is_short = len(tokens) < 5
    is_very_short = len(tokens) < 3
    dev = devanagari_ratio(text)
    rom = romanized_hindi_ratio(text)
    eng = english_marker_ratio(text)

    if is_very_short:
        if dev > 0.5:
            return ("hindi", min(0.8, dev + 0.2))
        if eng > 0.4:
            return ("english", min(0.8, eng * 2))
        return ("unknown", 0.4)

    if dev > 0.3:
        conf = min(1.0, dev + 0.3 + (0.1 if rom > 0.05 else 0.0))
        return ("hindi", conf)

    if rom > 0.1:
        if eng < 0.05:
            return ("hindi", min(0.9, rom * 4))
        if eng < rom:
            return ("hindi", min(0.8, rom * 3))

    if eng > 0.15 and dev < 0.05 and rom < 0.05:
        return ("english", min(1.0, eng * 3))

    if rom > 0.03 and eng > 0.03:
        if is_short:
            if rom > eng * 1.5:
                return ("hindi", 0.6)
            if eng > rom * 1.5:
                return ("english", 0.6)
            return ("mixed", 0.5)
        return ("mixed", 0.6)

    if is_short:
        if dev > 0.1:
            return ("hindi", min(0.6, dev * 2))
        if eng > 0.1:
            return ("english", min(0.6, eng * 2))
        return ("unknown", 0.3)

    if dev > 0.05:
        return ("hindi", min(0.7, dev * 2))
    if rom > 0.02:
        return ("hindi", min(0.6, rom * 4))
    return ("english", 0.6)


def normalized_edit_distance(a: str, b: str) -> float:
    if not (a or b):
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()
