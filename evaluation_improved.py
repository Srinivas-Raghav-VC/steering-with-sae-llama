#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved evaluation metrics for language steering experiments.
Measures both target language shadowing AND source language preservation.
"""

import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

# Language detection utilities
_DEV_RE = re.compile(r"[\u0900-\u097F]")
_ROMAN_HI_MARKERS = set("""
hai hain hoon raha rahe rahi mera meri mere kya nahi ka ki ke mein hum aap tum
bhai yaar ghar pyar tera tere tumhara tumhari kuch bahut accha theek
""".split())
_WORD_RE = re.compile(r"[A-Za-z']+")

def devanagari_ratio(text: str) -> float:
    """Calculate ratio of Devanagari characters in text."""
    if not text:
        return 0.0
    dev_chars = sum(1 for ch in text if _DEV_RE.match(ch))
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    return dev_chars / max(1, alpha_chars)

def romanized_hindi_ratio(text: str) -> float:
    """Detect romanized Hindi markers in text."""
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    marker_hits = sum(1 for t in tokens if t in _ROMAN_HI_MARKERS)
    return marker_hits / len(tokens)

def english_marker_ratio(text: str) -> float:
    """Detect English-specific markers."""
    english_markers = {"the", "and", "is", "are", "was", "were", "have", "has",
                      "been", "will", "would", "could", "should", "this", "that",
                      "these", "those", "which", "what", "when", "where"}
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    marker_hits = sum(1 for t in tokens if t in english_markers)
    return marker_hits / len(tokens)

def classify_language(text: str) -> Tuple[str, float]:
    """
    Classify text language with confidence score.
    Returns: (language, confidence)
    """
    dev_ratio = devanagari_ratio(text)
    rom_hi_ratio = romanized_hindi_ratio(text)
    eng_ratio = english_marker_ratio(text)

    # Strong Hindi indicators
    if dev_ratio > 0.3:
        return ("hindi", min(1.0, dev_ratio + 0.3))

    # Romanized Hindi
    if rom_hi_ratio > 0.1 and eng_ratio < 0.05:
        return ("hindi", min(1.0, rom_hi_ratio * 3))

    # Strong English indicators
    if eng_ratio > 0.15 and dev_ratio < 0.05 and rom_hi_ratio < 0.05:
        return ("english", min(1.0, eng_ratio * 3))

    # Mixed or unclear
    if rom_hi_ratio > 0.05 and eng_ratio > 0.05:
        return ("mixed", 0.5)

    # Default to English with low confidence
    if dev_ratio < 0.1 and rom_hi_ratio < 0.05:
        return ("english", 0.6)

    return ("unknown", 0.3)

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result for a single example."""
    prompt: str
    baseline: str
    steered: str
    prompt_lang: str
    baseline_lang: str
    baseline_conf: float
    steered_lang: str
    steered_conf: float
    is_flip: bool
    is_preservation: bool
    text_changed: bool
    dev_ratio_baseline: float
    dev_ratio_steered: float

    def to_dict(self) -> Dict:
        return {
            'prompt': self.prompt,
            'baseline': self.baseline,
            'steered': self.steered,
            'prompt_lang': self.prompt_lang,
            'baseline_lang': self.baseline_lang,
            'baseline_conf': float(self.baseline_conf),
            'steered_lang': self.steered_lang,
            'steered_conf': float(self.steered_conf),
            'is_flip': self.is_flip,
            'is_preservation': self.is_preservation,
            'text_changed': self.text_changed,
            'dev_ratio_baseline': float(self.dev_ratio_baseline),
            'dev_ratio_steered': float(self.dev_ratio_steered)
        }

class ComprehensiveEvaluator:
    """Evaluator that measures both shadowing effectiveness and language preservation."""

    def __init__(self, mode: str = "shadow_hindi"):
        """
        Args:
            mode: "shadow_hindi" or "shadow_english"
        """
        self.mode = mode
        self.target_lang = "english" if mode == "shadow_hindi" else "hindi"
        self.suppress_lang = "hindi" if mode == "shadow_hindi" else "english"

    def evaluate_single(self, prompt: str, baseline: str, steered: str) -> EvaluationResult:
        """Evaluate a single example."""
        # Language classification
        prompt_lang, _prompt_conf = classify_language(prompt)
        baseline_lang, baseline_conf = classify_language(baseline)
        steered_lang, steered_conf = classify_language(steered)

        # Check if text actually changed
        text_changed = baseline.strip() != steered.strip()

        # Determine if this is a successful flip
        is_flip = False
        if self.mode == "shadow_hindi":
            # Success: Hindi baseline -> English steered
            is_flip = (baseline_lang == "hindi" and steered_lang == "english" and text_changed)
        else:
            # Success: English baseline -> Hindi steered
            is_flip = (baseline_lang == "english" and steered_lang == "hindi" and text_changed)

        # Determine if language capability is preserved
        is_preservation = False
        if prompt_lang == self.target_lang:
            # If prompt is already in target language, output should remain in target
            is_preservation = (steered_lang == self.target_lang)

        return EvaluationResult(
            prompt=prompt,
            baseline=baseline,
            steered=steered,
            prompt_lang=prompt_lang,
            baseline_lang=baseline_lang,
            baseline_conf=baseline_conf,
            steered_lang=steered_lang,
            steered_conf=steered_conf,
            is_flip=is_flip,
            is_preservation=is_preservation,
            text_changed=text_changed,
            dev_ratio_baseline=devanagari_ratio(baseline),
            dev_ratio_steered=devanagari_ratio(steered)
        )

    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """Evaluate a batch of results and compute aggregate metrics."""
        evaluations = []

        for r in results:
            prompt = r.get('prompt', '')
            baseline = r.get('baseline', '')
            steered = r.get('steered', '')

            if not all([prompt, baseline, steered]):
                continue

            eval_result = self.evaluate_single(prompt, baseline, steered)
            evaluations.append(eval_result)

        if not evaluations:
            return {'error': 'No valid examples to evaluate'}

        # Compute aggregate metrics
        total = len(evaluations)

        # Shadowing metrics (for prompts in suppress_lang)
        shadow_eligible = [e for e in evaluations if e.baseline_lang == self.suppress_lang]
        shadow_success = sum(1 for e in shadow_eligible if e.is_flip)
        shadow_total = len(shadow_eligible)
        shadow_rate = shadow_success / max(1, shadow_total)

        # Preservation metrics (for prompts already in target_lang)
        preserve_eligible = [e for e in evaluations if e.prompt_lang == self.target_lang]
        preserve_success = sum(1 for e in preserve_eligible if e.steered_lang == self.target_lang)
        preserve_total = len(preserve_eligible)
        preserve_rate = preserve_success / max(1, preserve_total)

        # Text change metrics
        text_changed = sum(1 for e in evaluations if e.text_changed)
        change_rate = text_changed / max(1, total)

        # Language confidence metrics
        avg_baseline_conf = np.mean([e.baseline_conf for e in evaluations])
        avg_steered_conf = np.mean([e.steered_conf for e in evaluations])

        # Devanagari ratio changes
        dev_ratio_changes = [e.dev_ratio_steered - e.dev_ratio_baseline for e in evaluations]
        avg_dev_change = np.mean(dev_ratio_changes)

        # Combined effectiveness score
        # Weighted combination of shadowing and preservation
        effectiveness = 0.6 * shadow_rate + 0.4 * preserve_rate

        return {
            'mode': self.mode,
            'total_examples': total,

            # Core metrics
            'shadow_rate': float(shadow_rate),
            'shadow_success': shadow_success,
            'shadow_eligible': shadow_total,

            'preservation_rate': float(preserve_rate),
            'preservation_success': preserve_success,
            'preservation_eligible': preserve_total,

            'effectiveness_score': float(effectiveness),

            # Additional metrics
            'text_change_rate': float(change_rate),
            'avg_baseline_confidence': float(avg_baseline_conf),
            'avg_steered_confidence': float(avg_steered_conf),
            'avg_devanagari_change': float(avg_dev_change),

            # Detailed results
            'evaluations': [e.to_dict() for e in evaluations]
        }

def evaluate_english_preservation(results: List[Dict]) -> Dict:
    """
    Specifically evaluate English language preservation.
    This is critical for the stated goal of not affecting English capability.
    """
    english_prompts = []

    for r in results:
        prompt = r.get('prompt', '')
        baseline = r.get('baseline', '')
        steered = r.get('steered', '')

        prompt_lang, conf = classify_language(prompt)

        # Only consider prompts that are clearly English
        if prompt_lang == "english" and conf > 0.7:
            english_prompts.append({
                'prompt': prompt,
                'baseline': baseline,
                'steered': steered,
                'baseline_lang': classify_language(baseline)[0],
                'steered_lang': classify_language(steered)[0]
            })

    if not english_prompts:
        return {'error': 'No English prompts found for preservation testing'}

    # Measure preservation
    total = len(english_prompts)
    preserved = sum(1 for e in english_prompts if e['steered_lang'] == 'english')
    degraded = sum(1 for e in english_prompts if e['steered_lang'] != 'english')

    preservation_rate = preserved / max(1, total)

    return {
        'english_prompts_tested': total,
        'english_preserved': preserved,
        'english_degraded': degraded,
        'preservation_rate': float(preservation_rate),
        'examples': english_prompts[:5]  # Show first 5 examples
    }

def main(results_path: str, output_path: Optional[str] = None):
    """Main evaluation function."""
    # Load results
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    config = data.get('config', {})
    mode = config.get('eval_mode', 'shadow_hindi')

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(mode=mode)

    # Run comprehensive evaluation
    eval_results = evaluator.evaluate_batch(results)

    # Run specific English preservation test
    english_preservation = evaluate_english_preservation(results)
    eval_results['english_preservation'] = english_preservation

    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)

    print(f"\nMode: {mode}")
    print(f"Total examples: {eval_results['total_examples']}")

    print("\n--- SHADOWING EFFECTIVENESS ---")
    print(f"Shadow rate: {eval_results['shadow_rate']:.1%} ({eval_results['shadow_success']}/{eval_results['shadow_eligible']})")

    print("\n--- LANGUAGE PRESERVATION ---")
    print(f"Preservation rate: {eval_results['preservation_rate']:.1%} ({eval_results['preservation_success']}/{eval_results['preservation_eligible']})")

    print("\n--- ENGLISH CAPABILITY PRESERVATION ---")
    eng_pres = eval_results['english_preservation']
    if 'error' not in eng_pres:
        print(f"English preservation: {eng_pres['preservation_rate']:.1%} ({eng_pres['english_preserved']}/{eng_pres['english_prompts_tested']})")
        print(f"English degraded: {eng_pres['english_degraded']} examples")
    else:
        print(f"Error: {eng_pres['error']}")

    print("\n--- OVERALL EFFECTIVENESS ---")
    print(f"Combined effectiveness score: {eval_results['effectiveness_score']:.1%}")
    print(f"Text change rate: {eval_results['text_change_rate']:.1%}")

    # Save detailed results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")

    return eval_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation_improved.py <results.json> [output.json]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    main(results_path, output_path)
