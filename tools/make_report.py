#!/usr/bin/env python3
# tools/make_report.py
# Generate figures and tables from he_pipeline_results after training
# Usage: python tools/make_report.py --results_dir he_pipeline_results

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def add_critical_plots(out_dir, results, eval_results):
    """Add critical visualizations for steering effectiveness and layer analysis"""
    fig_dir = out_dir / "figures"

    # 1. Steering Effectiveness by Prompt Type
    if eval_results and "records" in eval_results:
        records = eval_results.get("records", [])
        if records:
            prompt_types = []
            success = []

            target = (eval_results or {}).get("target", "english")
            for r in records:
                prompt = r.get("prompt", "")
                # Detect prompt type based on content
                # Devanagari block U+0900..U+097F
                if any("\u0900" <= ch <= "\u097f" for ch in prompt):
                    prompt_types.append("Hindi")
                elif any(
                    phrase in prompt.lower()
                    for phrase in ["kal party", "main ghar", "kya hai", "nahi"]
                ):
                    prompt_types.append("Hinglish")
                else:
                    prompt_types.append("English")
                success.append(1 if r.get("steered_lang") == target else 0)

            if prompt_types and success:
                df_prompt = pd.DataFrame({"type": prompt_types, "success": success})
                plt.figure(figsize=(6, 4))
                # seaborn >= 0.13 uses errorbar=None (ci is deprecated)
                sns.barplot(data=df_prompt, x="type", y="success", errorbar=None)
                plt.title("Steering Success by Prompt Type")
                plt.ylabel("Success Rate")
                plt.ylim(0, 1.1)
                plt.tight_layout()
                plt.savefig(fig_dir / "success_by_prompt_type.png", dpi=200)
                plt.close()

    # 2. Layer Contribution Pie Chart
    weights = results.get("layer_weights", {})
    if weights:
        plt.figure(figsize=(8, 6))
        layers = list(weights.keys())
        values = list(weights.values())
        # Convert to percentages
        total = sum(values)
        if total > 0:
            percentages = [v / total * 100 for v in values]
            plt.pie(
                values,
                labels=[f"L{l}" for l in layers],
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title("Layer Contribution to Steering")
            plt.tight_layout()
            plt.savefig(fig_dir / "layer_contribution_pie.png", dpi=200)
            plt.close()

    # 3. Feature Activation Heatmap (top features only)
    per_layer = results.get("per_layer", {})
    for layer_str, data in per_layer.items():
        hi_feats = data.get("top_hindi_features", [])[:10]
        en_feats = data.get("top_english_features", [])[:10]

        if not (hi_feats or en_feats):
            continue

        # Create feature type matrix
        feat_data = []
        feat_labels = []

        for f in hi_feats[:5]:
            feat_data.append([1, 0])  # [Hindi, English]
            feat_labels.append(f"H{f}")

        for f in en_feats[:5]:
            feat_data.append([0, 1])
            feat_labels.append(f"E{f}")

        if feat_data:
            plt.figure(figsize=(4, 6))
            sns.heatmap(
                feat_data,
                xticklabels=["Hindi", "English"],
                yticklabels=feat_labels,
                cmap="RdBu_r",
                center=0.5,
                cbar_kws={"label": "Feature Type"},
            )
            plt.title(f"Top Features Layer {layer_str}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"top_features_L{layer_str}.png", dpi=200)
            plt.close()

    # 4. Causal deltas (optional): visualize top positive/negative features per selected layer
    causal_scores = results.get("causal_scores", {})
    if causal_scores:
        for layer_key, side_scores in causal_scores.items():
            # side_scores: {"hi": {feat: delta, ...}, "en": {...}}
            try:
                L = int(layer_key.lstrip("L"))
            except Exception:
                L = layer_key
            for side in ("hi", "en"):
                score_map = side_scores.get(side, {})
                if not score_map:
                    continue
                items = [(int(k), float(v)) for k, v in score_map.items()]
                # Top 10 positive and top 10 negative deltas
                items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
                top_pos = items_sorted[:10]
                top_neg = sorted(items, key=lambda x: x[1])[:10]
                if top_pos:
                    plt.figure(figsize=(8, 4))
                    plt.bar([f"F{f}" for f, _ in top_pos], [v for _, v in top_pos], color="#2ca02c")
                    plt.title(f"Layer {L} {side.upper()} causal deltas (top +)")
                    plt.ylabel("Δ flip rate vs baseline")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"causal_top_pos_L{L}_{side}.png", dpi=200)
                    plt.close()
                if top_neg:
                    plt.figure(figsize=(8, 4))
                    plt.bar([f"F{f}" for f, _ in top_neg], [v for _, v in top_neg], color="#d62728")
                    plt.title(f"Layer {L} {side.upper()} causal deltas (top −)")
                    plt.ylabel("Δ flip rate vs baseline")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"causal_top_neg_L{L}_{side}.png", dpi=200)
                    plt.close()

    # 5. LLM judge summary (optional): visualize Gemini summary scores if available
    judge_path = out_dir / "llm_judge_gemini.json"
    if judge_path.exists():
        try:
            with open(judge_path, "r", encoding="utf-8") as f:
                judge = json.load(f)
            summary = judge.get("summary", {}) if isinstance(judge, dict) else {}
            if summary:
                labels = []
                values = []
                for key in [
                    "language_compliance",
                    "meaning_preservation",
                    "coherence_fluency",
                    "overall_success",
                ]:
                    if key in summary and summary[key] is not None:
                        labels.append(key.replace("_", " ").title())
                        values.append(float(summary[key]))
                if labels and values:
                    plt.figure(figsize=(6, 4))
                    sns.barplot(x=labels, y=values, color="#8c564b", errorbar=None)
                    plt.ylim(0.0, 1.0)
                    plt.ylabel("Score")
                    plt.title("LLM Judge (Gemini) Summary Scores")
                    plt.xticks(rotation=20, ha="right")
                    plt.tight_layout()
                    plt.savefig(fig_dir / "llm_judge_summary.png", dpi=200)
                    plt.close()

            # Optional: per-record overall histogram if available
            recs = judge.get("records", []) if isinstance(judge, dict) else []
            overall = []
            for r in recs:
                j = r.get("judge", {}) if isinstance(r, dict) else {}
                v = j.get("overall_success", None)
                if isinstance(v, (int, float)):
                    overall.append(float(v))
            if overall:
                plt.figure(figsize=(6, 4))
                sns.histplot(overall, bins=20, color="#9467bd")
                plt.xlim(0.0, 1.0)
                plt.xlabel("Overall Success")
                plt.title("LLM Judge Overall Success Distribution")
                plt.tight_layout()
                plt.savefig(fig_dir / "llm_judge_overall_hist.png", dpi=200)
                plt.close()
        except Exception as e:
            print(f"Warning: Could not plot LLM judge scores: {e}")


def create_summary_table(out_dir, results, eval_results):
    """Create comprehensive summary statistics table"""
    tab_dir = out_dir / "tables"

    summary = {"Metric": [], "Value": []}

    # Basic stats
    summary["Metric"].append("Selected Layers")
    summary["Value"].append(str(results.get("selected_layers", [])))

    summary["Metric"].append("Success Rate")
    if eval_results and "records" in eval_results:
        records = eval_results.get("records", [])
        if records:
            success = sum(1 for r in records if r.get("steered_lang") == "english")
            summary["Value"].append(
                f"{success}/{len(records)} ({100*success/len(records):.1f}%)"
            )
        else:
            summary["Value"].append("No evaluation records")
    else:
        summary["Value"].append("N/A")

    # Per-layer stats
    per_layer = results.get("per_layer", {})
    for layer, data in per_layer.items():
        hi_count = len(data.get("top_hindi_features", []))
        en_count = len(data.get("top_english_features", []))
        summary["Metric"].append(f"L{layer} Features (HI/EN)")
        summary["Value"].append(f"{hi_count}/{en_count}")

    # Layer weights summary
    weights = results.get("layer_weights", {})
    if weights:
        max_weight_layer = max(weights.keys(), key=lambda k: weights[k])
        summary["Metric"].append("Highest Weight Layer")
        summary["Value"].append(
            f"L{max_weight_layer} ({weights[max_weight_layer]:.3f})"
        )

    # Include aggregate stats (flip rate, changed rate) if available
    aggregate = results.get("aggregate", {})
    if aggregate:
        summary["Metric"].append("Aggregate Flip Rate")
        summary["Value"].append(
            f"{aggregate.get('successes', 0)}/{aggregate.get('total_prompts', 0)} "
            f"({aggregate.get('flip_rate', float('nan')):.3f})"
        )
        summary["Metric"].append("Aggregate Changed Rate")
        summary["Value"].append(
            f"{aggregate.get('changed_prompts', 0)}/{aggregate.get('total_prompts', 0)} "
            f"({aggregate.get('changed_rate', float('nan')):.3f})"
        )

    # Include quality metrics if the pipeline produced them
    quality_metrics = results.get("quality_metrics") or {}
    if quality_metrics:
        summary["Metric"].append("ΔPPL (mean/median)")
        summary["Value"].append(
            f"{quality_metrics.get('mean_delta_ppl', float('nan')):.3f} / "
            f"{quality_metrics.get('median_delta_ppl', float('nan')):.3f}"
        )
        if "mean_semantic_similarity" in quality_metrics:
            summary["Metric"].append("Semantic Similarity (mean/median)")
            summary["Value"].append(
                f"{quality_metrics.get('mean_semantic_similarity', float('nan')):.3f} / "
                f"{quality_metrics.get('median_semantic_similarity', float('nan')):.3f}"
            )

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(tab_dir / "summary_stats.csv", index=False)

    # Also create markdown table
    with open(tab_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Pipeline Summary Statistics\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n## Notes\n")
        f.write(
            "- HI/EN features: Number of top Hindi and English features identified\n"
        )
        f.write(
            "- Success rate: Percentage of prompts successfully steered to target language\n"
        )
        f.write("- Layer weights: Relative contribution of each layer to steering\n")


def process_features_chunked(features, chunk_size=10000):
    """Process features in chunks to avoid memory issues"""
    for i in range(0, len(features), chunk_size):
        chunk = features[i : i + chunk_size]
        yield chunk


def main(results_dir: str):
    out_dir = Path(results_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    rep_dir = out_dir / "report"
    for d in [fig_dir, tab_dir, rep_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 0) Optionally load LLM-judge results (Gemini)
    llm_judge_path = out_dir / "llm_judge_gemini.json"
    llm_judge = load_json(llm_judge_path) if llm_judge_path.exists() else {}

    # 0.1) Optionally load detector calibration summary and copy key stats into tables/README
    calib_path = out_dir / "detector_calibration.json"
    calib = load_json(calib_path) if calib_path.exists() else {}

    # 1) Layer scores
    ls_path = out_dir / "layer_scores.json"
    # Support either results.json (legacy) or results_sae_only.json (current)
    res_path_legacy = out_dir / "results.json"
    res_path_sae = out_dir / "results_sae_only.json"
    eval_path = out_dir / "evaluation_results.json"  # optional telemetry file
    layer_scores = load_json(ls_path) if ls_path.exists() else {}
    if res_path_legacy.exists():
        results = load_json(res_path_legacy)
    elif res_path_sae.exists():
        results = load_json(res_path_sae)
    else:
        results = {}

    # Tables: per-layer scores
    rows = []
    scores_obj = layer_scores.get("scores") if isinstance(layer_scores, dict) else None
    if scores_obj:
        for L, s in scores_obj.items():
            try:
                rows.append(
                    {
                        "layer": int(L),
                        "LAPE": s.get("lape", np.nan),
                        "Probing": s.get("probing", np.nan),
                        "MMD": s.get("geometric", np.nan),
                        "Consensus": s.get("consensus", np.nan),
                    }
                )
            except Exception:
                continue
    df_layers = pd.DataFrame(rows)
    if not df_layers.empty:
        df_layers = df_layers.sort_values("layer")
        df_layers.to_csv(tab_dir / "layer_scores.csv", index=False)

        # Figures: consensus bar; MMD vs probing; metric stack
        plt.figure(figsize=(10, 4))
        sns.barplot(data=df_layers, x="layer", y="Consensus", color="#1f77b4")
        plt.title("Consensus score by layer")
        plt.xlabel("Layer")
        plt.ylabel("Consensus")
        plt.tight_layout()
        plt.savefig(fig_dir / "consensus_by_layer.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df_layers, x="Probing", y="MMD", hue="layer", palette="viridis", s=60
        )
        plt.title("MMD vs Probing")
        plt.tight_layout()
        plt.savefig(fig_dir / "mmd_vs_probing.png", dpi=200)
        plt.close()

        # Stacked bar for metrics (normalized per layer)
        df_long = df_layers.melt(
            id_vars=["layer", "Consensus"],
            value_vars=["LAPE", "Probing", "MMD"],
            var_name="Metric",
            value_name="Score",
        )
        # Normalize metric columns per layer
        df_norm = df_long.copy()
        df_norm["norm"] = df_norm.groupby("layer")["Score"].transform(
            lambda x: x / (x.sum() + 1e-12)
        )
        dfp = df_norm.pivot(index="layer", columns="Metric", values="norm").fillna(0.0)
        dfp.loc[:, ["LAPE", "Probing", "MMD"]].plot(
            kind="bar", stacked=True, figsize=(10, 4), colormap="tab20c"
        )
        plt.title("Per-layer metric mix (normalized)")
        plt.xlabel("Layer")
        plt.ylabel("Normalized weight")
        plt.tight_layout()
        plt.savefig(fig_dir / "metric_mix_by_layer.png", dpi=200)
        plt.close()

        # Split metrics plot into chunks for readability
        chunk_size = 8
        n_layers = len(df_layers)
        for i in range(0, n_layers, chunk_size):
            chunk = df_layers.iloc[i : i + chunk_size]
            plt.figure(figsize=(10, 4))
            plt.plot(chunk["layer"], chunk["LAPE"], label="LAPE", marker="o", linewidth=2)
            plt.plot(
                chunk["layer"], chunk["Probing"], label="Probing", marker="s", linewidth=2
            )
            plt.plot(chunk["layer"], chunk["MMD"], label="MMD", marker="^", linewidth=2)
            plt.title(f"Metrics across layers {i}-{min(i+chunk_size-1, n_layers-1)}")
            plt.xlabel("Layer")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                fig_dir / f"metrics_layers_{i}_{min(i+chunk_size-1, n_layers-1)}.png",
                dpi=200,
            )
            plt.close()
    else:
        # No layer scores available; skip these plots gracefully
        print("ℹ️  No layer_scores.json found or empty 'scores'; skipping metric plots")

    # 2) Selected layers summary and weights
    sel_layers = results.get("selected_layers", [])
    weights = results.get("layer_weights", {})
    df_sel = pd.DataFrame(
        [
            {
                "layer": int(L),
                "weight": float(weights.get(str(L), weights.get(L, np.nan))),
            }
            for L in sel_layers
        ]
    ).sort_values("layer")
    df_sel.to_csv(tab_dir / "selected_layers.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_sel, x="layer", y="weight", color="#ff7f0e")
    plt.title("Weights of selected layers")
    plt.xlabel("Layer")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(fig_dir / "selected_layer_weights.png", dpi=200)
    plt.close()

    # 3) Gemini label table (top entries)
    labels_serialized = results.get("gemini_labels", {})
    if labels_serialized:
        # convert to layer, idx, label
        labrows = []
        for k, v in labels_serialized.items():
            # k like "L18:130821"
            try:
                Ls, fs = k[1:].split(":")
                L = int(Ls)
                f = int(fs)
                labrows.append({"layer": L, "feature_idx": f, "label": v})
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed label key '{k}': {e}")
                continue
        df_labels = pd.DataFrame(labrows).sort_values(["layer", "feature_idx"])
        df_labels.to_csv(tab_dir / "gemini_labels.csv", index=False)
    else:
        df_labels = pd.DataFrame(columns=["layer", "feature_idx", "label"])

    # 4) Feature sparsity diagnostics
    per_layer = results.get("per_layer", {})
    for Ls, rec in per_layer.items():
        try:
            L = int(Ls)
        except Exception:
            L = Ls

        # Cohen's d distribution
        coh = rec.get("cohens_d", [])
        if coh:
            plt.figure(figsize=(6, 4))
            sns.histplot(coh, bins=50, kde=False, color="#2ca02c")
            plt.title(f"Cohen's d distribution (Layer {L})")
            plt.xlabel("|d| positive → Hindi; negative → English")
            plt.tight_layout()
            plt.savefig(fig_dir / f"cohens_d_hist_L{L}.png", dpi=200)
            plt.close()

        # Activation frequency differences
        hi_f = rec.get("hindi_freq", [])
        en_f = rec.get("english_freq", [])
        if hi_f and en_f:
            diff = np.array(hi_f) - np.array(en_f)
            plt.figure(figsize=(6, 4))
            sns.histplot(diff, bins=50, color="#9467bd")
            plt.title(f"Activation freq (HI − EN) (Layer {L})")
            plt.xlabel("Frequency difference")
            plt.tight_layout()
            plt.savefig(fig_dir / f"freq_diff_hist_L{L}.png", dpi=200)
            plt.close()

        # Gradient importance distribution
        grad_imp = rec.get("gradient_importance", [])
        if grad_imp:
            plt.figure(figsize=(6, 4))
            sns.histplot(grad_imp, bins=50, kde=False, color="#ff7f0e")
            plt.title(f"Gradient importance distribution (Layer {L})")
            plt.xlabel("Gradient importance")
            plt.tight_layout()
            plt.savefig(fig_dir / f"gradient_importance_hist_L{L}.png", dpi=200)
            plt.close()

    # 4.5) SAE L0 Evolution Tracking (critical for debugging L0 collapse)
    sae_metrics_path = out_dir / "sae_training_metrics.json"
    cfg = results.get("config", {}) if isinstance(results, dict) else {}
    ckpt_dir = Path(cfg.get("ckpt_dir", str(out_dir)))
    layer_metric_files = sorted(ckpt_dir.glob("sae_layer*_metrics.jsonl"))
    sae_data = None
    if sae_metrics_path.exists():
        print("📊 Generating SAE L0 evolution plots...")
        sae_data = load_json(sae_metrics_path)
    elif layer_metric_files:
        print("📊 Aggregating SAE metrics from per-layer logs...")
        sae_data = {}
        for path in layer_metric_files:
            try:
                layer_key = path.stem.split("_")[1].lstrip("layer")
            except Exception:
                layer_key = path.stem
            sae_data[layer_key] = {"epochs": [], "l0_history": [], "target": None}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sae_data[layer_key]["epochs"].append(rec.get("epoch"))
                    sae_data[layer_key]["l0_history"].append(rec.get("train_last", {}).get("l0_soft"))
                    if sae_data[layer_key]["target"] is None:
                        sae_data[layer_key]["target"] = rec.get("l0_target", 0)
    if sae_data:
        for layer_str, metrics in sae_data.items():
            try:
                layer = int(layer_str)
                epochs = metrics.get("epochs", [])
                l0_values = metrics.get("l0_history", [])
                target = metrics.get("target", 64)  # Default target

                if epochs and l0_values and len(epochs) == len(l0_values):
                    plt.figure(figsize=(8, 4))
                    plt.plot(
                        epochs,
                        l0_values,
                        label=f"L{layer} L0",
                        linewidth=2,
                        marker="o",
                        markersize=3,
                    )
                    plt.axhline(
                        y=target,
                        color="r",
                        linestyle="--",
                        label=f"Target ({target})",
                        alpha=0.7,
                    )
                    plt.xlabel("Epoch")
                    plt.ylabel("L0 (active features)")
                    plt.title(f"SAE L0 Evolution - Layer {layer}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"sae_l0_layer{layer}.png", dpi=200)
                    plt.close()

                    # Also create a final L0 vs target comparison
                    final_l0 = l0_values[-1] if l0_values else 0
                    achievement_ratio = final_l0 / target if target > 0 else 0
                    print(
                        f"  Layer {layer}: Final L0={final_l0:.1f}, Target={target}, Achievement={achievement_ratio:.1%}"
                    )
            except (ValueError, KeyError) as e:
                print(
                    f"Warning: Could not process SAE metrics for layer {layer_str}: {e}"
                )
                continue
    else:
        print("ℹ️  No SAE training metrics found - L0 evolution plots skipped")

    # 5) Evaluation summary (requires you to save it; see below how)
    if eval_path.exists():
        eval_results = load_json(eval_path)
        records = eval_results.get("records", [])
        df_eval = pd.DataFrame(records)
        # success rate
        success = np.mean(
            df_eval["steered_lang"] == eval_results.get("target", "english")
        )
        with open(tab_dir / "evaluation_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Success rate ({eval_results.get('mode','shadow_hindi')} → {eval_results.get('target','english')}): {success:.3f}\n"
            )
        df_eval.to_csv(tab_dir / "evaluation_details.csv", index=False)

        # If telemetry exists per record, you can create heatmaps of deltaL2 per layer
        all_tel = []
        for r in records:
            tel = r.get("telemetry", {})
            for Ls, t in tel.items():
                all_tel.append(
                    {
                        "prompt": r["prompt"],
                        "layer": int(Ls),
                        "deltaL2": t.get("delta_l2", 0.0),
                    }
                )
        if all_tel:
            df_tel = pd.DataFrame(all_tel)
            df_tel_piv = df_tel.pivot_table(
                index="prompt", columns="layer", values="deltaL2", fill_value=0.0
            )
            plt.figure(
                figsize=(
                    max(6, 0.4 * len(df_tel_piv.columns)),
                    max(4, 0.2 * len(df_tel_piv.index)),
                )
            )
            sns.heatmap(df_tel_piv, cmap="mako", cbar_kws={"label": "deltaL2"})
            plt.title("Per-prompt deltaL2 by layer")
            plt.tight_layout()
            plt.savefig(fig_dir / "deltaL2_heatmap.png", dpi=200)
            plt.close()

    # 6) Add critical visualizations
    print("📊 Generating critical visualizations...")
    add_critical_plots(out_dir, results, eval_results if eval_path.exists() else None)

    # 7) Create comprehensive summary statistics
    print("📋 Creating summary statistics...")
    create_summary_table(out_dir, results, eval_results if eval_path.exists() else None)

    # 7.5) Quality-aware success summary (merge strict flips + LLM judge)
    qa_rows = []
    if llm_judge:
        # Overall judge summary
        summary = llm_judge.get("summary", {})
        qa_rows.append({
            "metric": "judge_language_compliance",
            "value": summary.get("language_compliance", float("nan"))
        })
        qa_rows.append({
            "metric": "judge_meaning_preservation",
            "value": summary.get("meaning_preservation", float("nan"))
        })
        qa_rows.append({
            "metric": "judge_coherence_fluency",
            "value": summary.get("coherence_fluency", float("nan"))
        })
        qa_rows.append({
            "metric": "judge_overall_success",
            "value": summary.get("overall_success", float("nan"))
        })

    # Strict flips via rescore (if evaluation_results.json exists, derive; else skip)
    strict_flip_rate = None
    if eval_path.exists():
        try:
            eval_results = load_json(eval_path)
            recs = eval_results.get("records", [])
            if recs:
                target = eval_results.get("target", "english")
                flips = sum(1 for r in recs if r.get("steered_lang") == target)
                strict_flip_rate = flips / max(1, len(recs))
        except Exception:
            pass
    # Fallback: compute strict flip rate from 'results' schema
    if strict_flip_rate is None and results:
        try:
            recs = results.get("results", [])
            mode = (results.get("config", {}).get("eval_mode") or "shadow_hindi").lower()
            target = "english" if mode == "shadow_hindi" else "hindi"
            eligible = [r for r in recs if (r.get("baseline_lang") == ("hindi" if target == "english" else "english"))]
            flips = sum(
                1
                for r in eligible
                if r.get("steered_lang") == target
                and (r.get("steered") or "").strip() != (r.get("baseline") or "").strip()
            )
            strict_flip_rate = flips / max(1, len(eligible)) if eligible else None
        except Exception:
            strict_flip_rate = None

    if strict_flip_rate is not None:
        qa_rows.append({"metric": "strict_flip_rate", "value": strict_flip_rate})

    qa_md = ""
    if qa_rows:
        df_qa = pd.DataFrame(qa_rows)
        df_qa.to_csv(tab_dir / "quality_aware_summary.csv", index=False)
        # Prepare a markdown snippet to embed later in the report
        qa_md_lines = ["\n## Quality-aware Success Summary\n"]
        for _, row in df_qa.iterrows():
            qa_md_lines.append(f"- {row['metric']}: {row['value']}")
        qa_md = "\n".join(qa_md_lines) + "\n"

    # 7.6) Quality metrics figures (optional): Perplexity deltas and semantic similarity
    recs_for_quality = results.get("results", []) if isinstance(results, dict) else []
    if recs_for_quality:
        ppl_deltas = []
        sims = []
        for r in recs_for_quality:
            pb = r.get("ppl_baseline")
            ps = r.get("ppl_steered")
            sim = r.get("semantic_sim")
            try:
                if pb is not None and ps is not None:
                    ppl_deltas.append(float(ps) - float(pb))
            except Exception:
                pass
            try:
                if sim is not None:
                    sims.append(float(sim))
            except Exception:
                pass
        if ppl_deltas:
            plt.figure(figsize=(6,4))
            sns.histplot(ppl_deltas, bins=30, color="#1f77b4")
            plt.title("Perplexity delta (steered − baseline)")
            plt.xlabel("Δ PPL")
            plt.tight_layout()
            plt.savefig(fig_dir / "ppl_delta_hist.png", dpi=200)
            plt.close()
        if sims:
            plt.figure(figsize=(6,4))
            sns.histplot(sims, bins=30, color="#2ca02c")
            plt.title("Semantic similarity: baseline vs steered")
            plt.xlabel("Cosine similarity")
            plt.xlim(-1.0, 1.0)
            plt.tight_layout()
            plt.savefig(fig_dir / "semantic_similarity_hist.png", dpi=200)
            plt.close()

    # 7.7) If calibration exists, write a small table and note for the report
    if calib:
        try:
            summary = calib.get("summary", {}) if isinstance(calib, dict) else {}
            acc = float(summary.get("accuracy", float("nan")))
            macro_f1 = float(summary.get("macro_f1", float("nan")))
            used_gem = summary.get("used_gemini", False)
            dataset_name = calib.get("dataset_name", "calibration")
            with open(tab_dir / "detector_calibration_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Macro-F1: {macro_f1:.4f}\n")
                f.write(f"Gemini used: {used_gem}\n")
        except Exception as e:
            print(f"Warning: Could not write calibration summary: {e}")

    # 8) Write a comprehensive README.md for the report folder
    with open(rep_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("# Language Steering Pipeline Report\n\n")
        f.write("## Overview\n")
        f.write(
            "This report contains comprehensive analysis of the Hindi-English language steering pipeline.\n\n"
        )

        # If available, include QA section near the top
        if qa_md:
            f.write(qa_md)

        f.write("## Tables\n")
        f.write("### Core Results\n")
        for p in sorted(tab_dir.glob("*.csv")):
            f.write(f"- `{p.name}` - {p.stem.replace('_', ' ').title()}\n")
        for p in sorted(tab_dir.glob("*.txt")):
            f.write(f"- `{p.name}` - {p.stem.replace('_', ' ').title()}\n")
        for p in sorted(tab_dir.glob("*.md")):
            f.write(f"- `{p.name}` - {p.stem.replace('_', ' ').title()}\n")

        f.write("\n### Key Files\n")
        f.write("- `summary_stats.csv` - Comprehensive pipeline statistics\n")
        f.write(
            "- `layer_scores.csv` - Per-layer discovery metrics (LAPE, Probing, MMD)\n"
        )
        f.write("- `selected_layers.csv` - Final layer selection and weights\n")
        f.write("- `gemini_labels.csv` - Feature interpretations (if available)\n")

        f.write("\n## Figures\n")
        f.write("### Discovery Analysis\n")
        for p in sorted(fig_dir.glob("consensus*.png")):
            f.write(f"- `{p.name}` - Layer consensus scores\n")
        for p in sorted(fig_dir.glob("mmd_vs_probing*.png")):
            f.write(f"- `{p.name}` - MMD vs Probing correlation\n")
        for p in sorted(fig_dir.glob("metrics_layers_*.png")):
            f.write(f"- `{p.name}` - Metrics across layer chunks\n")

        f.write("\n### SAE Training Analysis\n")
        for p in sorted(fig_dir.glob("sae_l0_layer*.png")):
            f.write(f"- `{p.name}` - SAE L0 evolution (critical for debugging)\n")
        for p in sorted(fig_dir.glob("*_hist_L*.png")):
            f.write(f"- `{p.name}` - Feature distribution histograms\n")

        f.write("\n### Steering Effectiveness\n")
        for p in sorted(fig_dir.glob("success_by_prompt_type*.png")):
            f.write(f"- `{p.name}` - Success rate by prompt type\n")
        for p in sorted(fig_dir.glob("layer_contribution*.png")):
            f.write(f"- `{p.name}` - Layer contribution pie chart\n")
        for p in sorted(fig_dir.glob("top_features_L*.png")):
            f.write(f"- `{p.name}` - Top feature activation patterns\n")
        for p in sorted(fig_dir.glob("deltaL2_heatmap*.png")):
            f.write(f"- `{p.name}` - Delta L2 heatmap by layer and prompt\n")

        # If calibration confusion heatmap exists, list it
        if (fig_dir / "detector_confusion.png").exists():
            f.write("\n### Detector Calibration\n")
            f.write("- `detector_confusion.png` - Language detector confusion heatmap (row-normalized)\n")

        f.write("\n## Usage Notes\n")
        f.write("- **SAE L0 plots**: Critical for debugging L0 collapse issues\n")
        f.write(
            "- **Success rate plots**: Show steering effectiveness by prompt type\n"
        )
        f.write(
            "- **Feature histograms**: Reveal feature sparsity and distribution patterns\n"
        )
        f.write(
            "- **Layer contribution**: Shows which layers are most important for steering\n"
        )

    print(f"\n✅ Report generation complete!")
    print(f"📁 Figures → {fig_dir}")
    print(f"📊 Tables → {tab_dir}")
    print(f"📋 Report → {rep_dir / 'report.md'}")
    print(f"\n🔍 Key files to check:")
    print(f"  - summary_stats.csv - Overall pipeline performance")
    print(f"  - sae_l0_layer*.png - SAE training progress (critical for debugging)")
    print(f"  - success_by_prompt_type.png - Steering effectiveness")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="he_pipeline_results")
    args = ap.parse_args()
    main(args.results_dir)
