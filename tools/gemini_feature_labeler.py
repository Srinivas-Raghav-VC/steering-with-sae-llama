#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained Gemini feature labeler utility.

Provides a small wrapper class `GeminiFeatureLabeler` that labels SAE features
based on a handful of example texts. Falls back to a simple heuristic if the
Gemini API or package isn't available.

Usage:
    from tools.gemini_feature_labeler import GeminiFeatureLabeler
    labeler = GeminiFeatureLabeler(api_key=os.getenv("GEMINI_API_KEY"))
    label = labeler.label_feature("L20:123", examples=["main ghar ja raha hoon", ...])

Requirements (optional fast path):
- google-generativeai (listed in requirements.txt)
- GEMINI_API_KEY env var
"""
from __future__ import annotations
import os
from typing import List, Optional

class GeminiFeatureLabeler:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._client = None
        try:
            import google.generativeai as genai  # type: ignore
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(model)
        except Exception:
            self._client = None

    def label_feature(self, feature_id: str, examples: List[str]) -> str:
        # If Gemini client is not available, fallback heuristic
        if self._client is None:
            return self._heuristic_label(feature_id, examples)
        prompt = self._build_prompt(feature_id, examples)
        try:
            resp = self._client.generate_content(prompt)
            txt = getattr(resp, 'text', None) or "".join(getattr(resp, 'candidates', []) or [])
            txt = (txt or "").strip()
            if not txt:
                return self._heuristic_label(feature_id, examples)
            # Keep it short
            return self._postprocess_label(txt)
        except Exception:
            return self._heuristic_label(feature_id, examples)

    def _build_prompt(self, feature_id: str, examples: List[str]) -> str:
        ex = "\n".join(f"- {t}" for t in examples[:4])
        return (
            "You are labeling a sparse autoencoder feature discovered in a language model.\n"
            f"Feature: {feature_id}\n"
            "Provide a short, human-readable label (<=8 words) describing the linguistic pattern.\n"
            "Focus on language identity, script, morphology, syntax, or code-mixing (Hinglish).\n"
            "Examples where this feature activates strongly:\n"
            f"{ex}\n"
            "Return only the label, no explanations."
        )

    def _heuristic_label(self, feature_id: str, examples: List[str]) -> str:
        text = " ".join(examples).lower()
        dev = any("\u0900" <= ch <= "\u097f" for ch in text)
        markers = {"hai","hain","hoon","raha","rahe","rahi","mera","meri","mere","kya","nahi","ka","ki","ke","mein","hum","aap","tum","bhai"}
        roman_hits = sum(1 for w in text.split() if w in markers)
        if roman_hits > 2 and not dev:
            return f"{feature_id}: Hinglish / romanized Hindi"
        if dev:
            return f"{feature_id}: Devanagari / Hindi script"
        if any(w in text for w in [" the ", " and ", " is "]):
            return f"{feature_id}: English lexical pattern"
        return f"{feature_id}: Language-selective feature"

    def _postprocess_label(self, s: str) -> str:
        s = s.strip().splitlines()[0]
        if len(s) > 64:
            s = s[:64].rstrip()
        return s
