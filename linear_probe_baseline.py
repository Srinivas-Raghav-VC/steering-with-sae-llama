#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Probe Baseline for Language Steering
A simpler alternative to SAE-based steering for comparison.
"""

import argparse
import json
import difflib
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utilities from main pipeline
from he_steer_pipeline import (
    Config,
    setup_environment,
    load_model_and_tokenizer,
    load_text_pairs,
    devanagari_ratio,
    build_eval_prompts,
    find_max_batch_size
)
from tools.lang_detect import classify_language  # calibrated detector
from tools.quality_metrics import compute_perplexity, cosine_similarity_model

# Threshold for considering the steered text meaningfully changed
TEXT_CHANGE_THRESHOLD = 0.05

def normalized_edit_distance(a: str, b: str) -> float:
    """Return normalized edit distance between two strings in [0, 1]."""
    if not (a or b):
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def summarize_quality_metrics(records: List[Dict]) -> Optional[Dict[str, float]]:
    if not records:
        return None
    ppl_baseline: List[float] = []
    ppl_steered: List[float] = []
    delta_ppl: List[float] = []
    semantic_sim: List[float] = []
    edit_vals: List[float] = []
    changed = 0
    for rec in records:
        pb = rec.get("ppl_baseline")
        ps = rec.get("ppl_steered")
        sim = rec.get("semantic_sim")
        if isinstance(pb, (int, float)) and isinstance(ps, (int, float)):
            pb_f = float(pb)
            ps_f = float(ps)
            ppl_baseline.append(pb_f)
            ppl_steered.append(ps_f)
            delta_ppl.append(ps_f - pb_f)
        if isinstance(sim, (int, float)):
            semantic_sim.append(float(sim))
        if isinstance(rec.get("edit_distance"), (int, float)):
            edit_vals.append(float(rec["edit_distance"]))
        if rec.get("text_changed"):
            changed += 1
    summary: Dict[str, float] = {"count": float(len(records))}
    if ppl_baseline:
        pb_arr = np.array(ppl_baseline, dtype=float)
        ps_arr = np.array(ppl_steered, dtype=float)
        dp_arr = np.array(delta_ppl, dtype=float)
        summary.update(
            {
                "mean_ppl_baseline": float(pb_arr.mean()),
                "mean_ppl_steered": float(ps_arr.mean()),
                "mean_delta_ppl": float(dp_arr.mean()),
                "median_delta_ppl": float(np.median(dp_arr)),
                "delta_ppl_count": float(len(delta_ppl)),
            }
        )
    if semantic_sim:
        sim_arr = np.array(semantic_sim, dtype=float)
        summary.update(
            {
                "mean_semantic_similarity": float(sim_arr.mean()),
                "median_semantic_similarity": float(np.median(sim_arr)),
                "semantic_similarity_count": float(len(semantic_sim)),
            }
        )
    if edit_vals:
        edit_arr = np.array(edit_vals, dtype=float)
        summary.update(
            {
                "mean_edit_distance": float(edit_arr.mean()),
                "median_edit_distance": float(np.median(edit_arr)),
            }
        )
    summary["text_changed_rate"] = float(changed / max(1, len(records)))
    return summary if len(summary) > 1 else None

@dataclass
class LinearProbeConfig(Config):
    """Extended config for linear probe baseline."""
    probe_regularization: float = 1.0  # C parameter for LogisticRegression
    probe_max_iter: int = 1000
    steering_method: str = "direction"  # "direction" or "difference"
    probe_train_samples: int = 5000  # Samples per language for probe training
    compute_quality_metrics: bool = False


class LinearProbe:
    """Simple linear probe for language classification."""

    def __init__(self, input_dim: int, regularization: float = 1.0):
        self.input_dim = input_dim
        self.probe = LogisticRegression(
            C=regularization,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the probe."""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        # Train probe
        self.probe.fit(X_scaled, y)
        self.is_fitted = True

    def get_direction(self) -> np.ndarray:
        """Get the direction vector for steering (probe weights)."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted first")
        # For binary classification, this is the weight vector
        return self.probe.coef_[0]  # Shape: (input_dim,)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted first")
        X_scaled = self.scaler.transform(X)
        return self.probe.predict_proba(X_scaled)


@torch.no_grad()
def extract_representations(
    model,
    tokenizer,
    texts: List[str],
    layer: int,
    max_batch_size: int = 32,
    max_seq_len: int = 512,
    pooling: str = "mean"
) -> np.ndarray:
    """Extract representations from a specific layer."""
    device = next(model.parameters()).device
    representations = []

    for i in tqdm(range(0, len(texts), max_batch_size), desc=f"Extracting L{layer}"):
        batch = texts[i:i + max_batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len
        ).to(device)

        # Hook to capture representations
        captured = {}
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if pooling == "mean":
                # Mean pooling over sequence
                mask = inputs.attention_mask.unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            elif pooling == "last":
                # Last token
                pooled = hidden_states[:, -1, :]
            else:
                # Weighted combination
                mask = inputs.attention_mask.unsqueeze(-1)
                mean_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
                last_pooled = hidden_states[:, -1, :]
                pooled = 0.7 * mean_pooled + 0.3 * last_pooled
            captured['pooled'] = pooled.cpu().numpy()

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            _ = model(**inputs, use_cache=False)
            representations.append(captured['pooled'])
        finally:
            handle.remove()

    return np.concatenate(representations, axis=0)


class LinearProbeSteerer:
    """Steering using linear probe directions."""

    def __init__(
        self,
        model,
        tokenizer,
        config: LinearProbeConfig,
        layer_probes: Dict[int, LinearProbe],
        layer_weights: Optional[Dict[int, float]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.probes = layer_probes

        # Normalize layer weights
        if layer_weights and sum(layer_weights.values()) > 0:
            total = sum(layer_weights.values())
            self.layer_weights = {L: w/total for L, w in layer_weights.items()}
        else:
            n_layers = len(layer_probes)
            self.layer_weights = {L: 1.0/n_layers for L in layer_probes}

    def _create_steering_hook(self, layer: int, target_lang: str, strength: float):
        """Create a hook function for steering at a specific layer."""
        probe = self.probes[layer]
        direction = probe.get_direction()  # Shape: (hidden_dim,)
        direction = torch.tensor(direction, dtype=torch.float32)

        # Determine steering direction based on target
        if target_lang.lower() == "english":
            # Negative direction points toward English (class 0)
            direction = -direction
        # else: positive direction points toward Hindi (class 1)

        layer_weight = self.layer_weights.get(layer, 1.0)
        clamp_ratio = self.config.clamp_ratio

        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            B, T, D = hidden_states.shape
            device = hidden_states.device

            # Move direction to correct device
            steering_vec = direction.to(device).to(hidden_states.dtype)

            if self.config.intervention_scope == "last":
                # Only modify last token
                delta = strength * layer_weight * steering_vec.unsqueeze(0)

                # Norm clamping
                h_norm = hidden_states[:, -1, :].norm(dim=1, keepdim=True).clamp(min=1e-6)
                d_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
                scale = torch.clamp(clamp_ratio * h_norm / d_norm, max=1.0)
                delta = delta * scale

                hidden_states[:, -1, :] = hidden_states[:, -1, :] + delta
            else:
                # Sequence-wide intervention with position weighting
                pos_weights = torch.linspace(
                    self.config.pos_weight_start,
                    self.config.pos_weight_end,
                    steps=T
                ).to(device)

                for t in range(T):
                    delta = strength * layer_weight * pos_weights[t] * steering_vec.unsqueeze(0)

                    # Norm clamping
                    h_norm = hidden_states[:, t, :].norm(dim=1, keepdim=True).clamp(min=1e-6)
                    d_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    scale = torch.clamp(clamp_ratio * h_norm / d_norm, max=1.0)
                    delta = delta * scale

                    hidden_states[:, t, :] = hidden_states[:, t, :] + delta

            return (hidden_states,) if isinstance(output, tuple) else hidden_states

        return hook

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        target_lang: str,
        strength: float,
        max_new_tokens: int = 64,
        deterministic: bool = True
    ) -> Tuple[str, str]:
        """Generate with steering and return (steered, baseline)."""
        device = next(self.model.parameters()).device

        # Steered generation
        hooks = []
        old_use_cache = self.model.config.use_cache
        if self.config.intervention_scope == "sequence":
            self.model.config.use_cache = False

        try:
            for layer in self.probes:
                hook = self.model.model.layers[layer].register_forward_hook(
                    self._create_steering_hook(layer, target_lang, strength)
                )
                hooks.append(hook)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id
            }

            if deterministic:
                outputs = self.model.generate(**inputs, do_sample=False, **gen_kwargs)
            else:
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    **gen_kwargs
                )

            steered = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        finally:
            for hook in hooks:
                hook.remove()
            self.model.config.use_cache = old_use_cache

        # Baseline generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        if deterministic:
            outputs = self.model.generate(**inputs, do_sample=False, **gen_kwargs)
        else:
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                **gen_kwargs
            )
        baseline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return only the generated part (remove prompt)
        steered = steered[len(prompt):].strip()
        baseline = baseline[len(prompt):].strip()

        return steered, baseline


def train_linear_probes(
    model,
    tokenizer,
    config: LinearProbeConfig,
    hindi_texts: List[str],
    english_texts: List[str],
    layers: List[int]
) -> Dict[int, LinearProbe]:
    """Train linear probes for specified layers."""
    probes = {}

    # Sample training data
    n_samples = min(config.probe_train_samples, len(hindi_texts), len(english_texts))
    hindi_train = random.sample(hindi_texts, n_samples)
    english_train = random.sample(english_texts, n_samples)

    # Create labels (0 for English, 1 for Hindi)
    texts = english_train + hindi_train
    labels = np.array([0] * len(english_train) + [1] * len(hindi_train))

    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = labels[indices]

    # Find batch size for extraction
    batch_size = find_max_batch_size(
        model, tokenizer, texts[:100], config, config.max_sequence_length
    )
    batch_size = max(16, batch_size)

    # Train probe for each layer
    for layer in layers:
        print(f"\nTraining probe for layer {layer}...")

        # Extract representations
        representations = extract_representations(
            model, tokenizer, texts, layer,
            max_batch_size=batch_size,
            max_seq_len=config.max_sequence_length,
            pooling="mean"
        )

        # Train probe
        probe = LinearProbe(
            input_dim=representations.shape[1],
            regularization=config.probe_regularization
        )
        probe.fit(representations, labels)

        # Evaluate probe accuracy
        probs = probe.predict_proba(representations)
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == labels)
        print(f"  Layer {layer} probe accuracy: {accuracy:.3f}")

        probes[layer] = probe

    return probes


def evaluate_probe_steering(
    model,
    tokenizer,
    config: LinearProbeConfig,
    probes: Dict[int, LinearProbe],
    eval_prompts: List[str],
    target_lang: str = "english",
    strength: float = 2.0
) -> Dict:
    """Evaluate probe-based steering."""
    steerer = LinearProbeSteerer(model, tokenizer, config, probes)

    results = []
    success_count = 0

    for prompt in tqdm(eval_prompts, desc="Evaluating"):
        steered, baseline = steerer.generate(
            prompt,
            target_lang=target_lang,
            strength=strength,
            max_new_tokens=48,
            deterministic=True
        )

        baseline_lang, baseline_conf = classify_language(baseline)
        steered_lang, steered_conf = classify_language(steered)
        # Guard against trivial copies using normalized edit distance
        edit_delta = normalized_edit_distance(baseline.strip(), steered.strip())
        text_changed = edit_delta >= TEXT_CHANGE_THRESHOLD

        # Check success based on mode
        if config.eval_mode == "shadow_hindi":
            is_success = (baseline_lang == "hindi" and steered_lang == "english" and text_changed)
        elif config.eval_mode == "shadow_english":
            is_success = (baseline_lang == "english" and steered_lang == "hindi" and text_changed)
        else:
            is_success = (steered_lang == target_lang.lower() and text_changed)

        if is_success:
            success_count += 1

        # Optional quality metrics
        ppl_b = ppl_s = sim_bs = None
        if getattr(config, "compute_quality_metrics", False):
            try:
                ppl_b = compute_perplexity(model, tokenizer, baseline, max_length=config.max_sequence_length)
                ppl_s = compute_perplexity(model, tokenizer, steered, max_length=config.max_sequence_length)
                sim_bs = cosine_similarity_model(model, tokenizer, baseline, steered, max_length=min(256, config.max_sequence_length))
            except Exception:
                ppl_b = ppl_s = sim_bs = None

        results.append({
            "prompt": prompt,
            "baseline": baseline,
            "steered": steered,
            "baseline_lang": baseline_lang,
            "steered_lang": steered_lang,
            "baseline_conf": float(baseline_conf),
            "steered_conf": float(steered_conf),
            "edit_distance": float(edit_delta),
            "text_changed": bool(text_changed),
            "is_success": is_success,
            "baseline_dev_ratio": devanagari_ratio(baseline),
            "steered_dev_ratio": devanagari_ratio(steered),
            **({"ppl_baseline": float(ppl_b)} if ppl_b is not None else {}),
            **({"ppl_steered": float(ppl_s)} if ppl_s is not None else {}),
            **({"semantic_sim": float(sim_bs)} if sim_bs is not None else {}),
        })

    success_rate = success_count / max(1, len(eval_prompts))

    return {
        "success_rate": success_rate,
        "total": len(eval_prompts),
        "successes": success_count,
        "results": results
    }


class LinearProbeBaseline:
    """Main pipeline for linear probe baseline."""

    def __init__(self, config: LinearProbeConfig):
        self.config = config
        setup_environment()
        self.model, self.tokenizer = load_model_and_tokenizer(config)

    def run(self):
        """Run the complete baseline pipeline."""
        print("\n" + "="*60)
        print("LINEAR PROBE BASELINE FOR LANGUAGE STEERING")
        print("="*60)

        # Load data
        print("\nLoading datasets...")
        hindi_texts, english_texts = load_text_pairs(self.config)

        # Split data
        split_idx = int(0.9 * len(hindi_texts))
        hindi_train = hindi_texts[:split_idx]
        hindi_val = hindi_texts[split_idx:]
        english_train = english_texts[:split_idx]
        english_val = english_texts[split_idx:]

        # Build evaluation prompts
        eval_prompts = build_eval_prompts(hindi_val, english_val, self.config.eval_prompts)

        # Determine layers to test
        layers = list(range(*self.config.layer_range))
        print(f"\nTesting layers: {layers}")

        # Train probes
        print("\nTraining linear probes...")
        probes = train_linear_probes(
            self.model,
            self.tokenizer,
            self.config,
            hindi_train,
            english_train,
            layers
        )

        # Evaluate each layer individually
        print("\nEvaluating individual layers...")
        layer_results = {}

        for layer in layers:
            print(f"\nEvaluating layer {layer}...")
            result = evaluate_probe_steering(
                self.model,
                self.tokenizer,
                self.config,
                {layer: probes[layer]},
                eval_prompts,
                target_lang="english" if self.config.eval_mode == "shadow_hindi" else "hindi",
                strength=self.config.eval_strength
            )
            layer_results[layer] = result
            print(f"  Success rate: {result['success_rate']:.3f}")

        # Select best layers
        sorted_layers = sorted(
            layer_results.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        best_layers = [l for l, _ in sorted_layers[:self.config.top_k_layers]]
        print(f"\nBest layers by effectiveness: {best_layers}")

        # Evaluate best combination
        print(f"\nEvaluating best layer combination: {best_layers}")
        best_probes = {l: probes[l] for l in best_layers}

        # Weight layers by their individual success rates
        layer_weights = {
            l: layer_results[l]['success_rate']
            for l in best_layers
        }

        final_result = evaluate_probe_steering(
            self.model,
            self.tokenizer,
            self.config,
            best_probes,
            eval_prompts,
            target_lang="english" if self.config.eval_mode == "shadow_hindi" else "hindi",
            strength=self.config.eval_strength
        )

        print(f"\nFinal success rate: {final_result['success_rate']:.3f}")

        quality_summary = summarize_quality_metrics(final_result["results"])
        aggregate_summary = {
            "total": final_result["total"],
            "successes": final_result["successes"],
            "success_rate": final_result["success_rate"],
        }
        if quality_summary:
            final_result["quality_metrics"] = quality_summary

        # Save results
        output = {
            "method": "linear_probe_baseline",
            "config": asdict(self.config),
            "layer_results": {
                str(l): {
                    "success_rate": r["success_rate"],
                    "total": r["total"],
                    "successes": r["successes"]
                }
                for l, r in layer_results.items()
            },
            "best_layers": best_layers,
            "layer_weights": layer_weights,
            "final_result": {
                "success_rate": final_result["success_rate"],
                "total": final_result["total"],
                "successes": final_result["successes"]
            },
            "detailed_results": final_result["results"]
        }
        output["aggregate"] = aggregate_summary
        if quality_summary:
            output["quality_metrics"] = quality_summary

        # Save to file
        output_path = Path(self.config.out_dir) / "linear_probe_baseline_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

        # Print comparison summary
        print("\n" + "="*60)
        print("LINEAR PROBE BASELINE SUMMARY")
        print("="*60)
        print(f"Method: Linear Probe + Steering Vectors")
        print(f"Best layers: {best_layers}")
        print(f"Success rate: {final_result['success_rate']:.1%}")
        print(f"Total examples: {final_result['total']}")

        return output


def main():
    """Main entry point for linear probe baseline."""
    parser = argparse.ArgumentParser(description="Linear Probe Baseline for Language Steering")
    parser.add_argument("--layers", type=str, help="Comma-separated list of layers to test")
    parser.add_argument("--mode", type=str, default="shadow_hindi",
                       choices=["shadow_hindi", "shadow_english", "steer"])
    parser.add_argument("--strength", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Samples per language for probe training")
    parser.add_argument("--eval-prompts", type=int, default=24,
                       help="Number of evaluation prompts")
    parser.add_argument("--debug", action="store_true", help="Debug mode with fewer samples")

    args = parser.parse_args()

    # Create config
    config = LinearProbeConfig()
    config.eval_mode = args.mode
    config.eval_strength = args.strength
    config.probe_train_samples = args.samples
    config.eval_prompts = args.eval_prompts

    if args.debug:
        config.samples_per_language = 2000
        config.probe_train_samples = 1000
        config.eval_prompts = 12
        config.max_sequence_length = 256

    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
        config.layer_range = (min(layers), max(layers) + 1)

    # Run baseline
    baseline = LinearProbeBaseline(config)
    baseline.run()


if __name__ == "__main__":
    main()
