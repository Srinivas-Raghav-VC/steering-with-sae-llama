#!/usr/bin/env python3
# tools/make_report.py
# Generate figures and tables from he_pipeline_results after training
# Usage: python tools/make_report.py --results_dir he_pipeline_results

import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def main(results_dir: str):
    out_dir = Path(results_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    rep_dir = out_dir / "report"
    for d in [fig_dir, tab_dir, rep_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Layer scores
    ls_path = out_dir / "layer_scores.json"
    res_path = out_dir / "results.json"
    eval_path = out_dir / "evaluation_results.json"  # optional telemetry file
    layer_scores = load_json(ls_path)
    results = load_json(res_path)

    # Tables: per-layer scores
    rows = []
    for L, s in layer_scores["scores"].items():
        rows.append({
            "layer": int(L),
            "LAPE": s["lape"],
            "Probing": s["probing"],
            "MMD": s["geometric"],
            "Consensus": s["consensus"] if "consensus" in s else np.nan
        })
    df_layers = pd.DataFrame(rows).sort_values("layer")
    df_layers.to_csv(tab_dir / "layer_scores.csv", index=False)

    # Figures: consensus bar; MMD vs probing; metric stack
    plt.figure(figsize=(10,4))
    sns.barplot(data=df_layers, x="layer", y="Consensus", color="#1f77b4")
    plt.title("Consensus score by layer")
    plt.xlabel("Layer"); plt.ylabel("Consensus")
    plt.tight_layout()
    plt.savefig(fig_dir / "consensus_by_layer.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df_layers, x="Probing", y="MMD", hue="layer", palette="viridis", s=60)
    plt.title("MMD vs Probing")
    plt.tight_layout()
    plt.savefig(fig_dir / "mmd_vs_probing.png", dpi=200)
    plt.close()

    # Stacked bar for metrics (normalized per layer)
    df_long = df_layers.melt(id_vars=["layer","Consensus"], value_vars=["LAPE","Probing","MMD"], var_name="Metric", value_name="Score")
    # Normalize metric columns per layer
    df_norm = df_long.copy()
    df_norm["norm"] = df_norm.groupby("layer")["Score"].transform(lambda x: x / (x.sum() + 1e-12))
    dfp = df_norm.pivot(index="layer", columns="Metric", values="norm").fillna(0.0)
    dfp.loc[:, ["LAPE","Probing","MMD"]].plot(kind="bar", stacked=True, figsize=(10,4), colormap="tab20c")
    plt.title("Per-layer metric mix (normalized)")
    plt.xlabel("Layer"); plt.ylabel("Normalized weight")
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_mix_by_layer.png", dpi=200)
    plt.close()

    # 2) Selected layers summary and weights
    sel_layers = results.get("selected_layers", [])
    weights = results.get("layer_weights", {})
    df_sel = pd.DataFrame([{"layer": int(L), "weight": float(weights.get(str(L), weights.get(L, np.nan)))} for L in sel_layers]).sort_values("layer")
    df_sel.to_csv(tab_dir / "selected_layers.csv", index=False)

    plt.figure(figsize=(6,4))
    sns.barplot(data=df_sel, x="layer", y="weight", color="#ff7f0e")
    plt.title("Weights of selected layers")
    plt.xlabel("Layer"); plt.ylabel("Weight")
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
                L = int(Ls); f = int(fs)
                labrows.append({"layer": L, "feature_idx": f, "label": v})
            except:
                pass
        df_labels = pd.DataFrame(labrows).sort_values(["layer","feature_idx"])
        df_labels.to_csv(tab_dir / "gemini_labels.csv", index=False)
    else:
        df_labels = pd.DataFrame(columns=["layer","feature_idx","label"])

    # 4) Evaluation summary (requires you to save it; see below how)
    if eval_path.exists():
        eval_results = load_json(eval_path)
        records = eval_results.get("records", [])
        df_eval = pd.DataFrame(records)
        # success rate
        success = np.mean(df_eval["steered_lang"] == eval_results.get("target","english"))
        with open(tab_dir / "evaluation_summary.txt", "w") as f:
            f.write(f"Success rate ({eval_results.get('mode','shadow_hindi')} → {eval_results.get('target','english')}): {success:.3f}\n")
        df_eval.to_csv(tab_dir / "evaluation_details.csv", index=False)

        # If telemetry exists per record, you can create heatmaps of deltaL2 per layer
        all_tel = []
        for r in records:
            tel = r.get("telemetry", {})
            for Ls, t in tel.items():
                all_tel.append({"prompt": r["prompt"], "layer": int(Ls), "deltaL2": t.get("delta_l2", 0.0)})
        if all_tel:
            df_tel = pd.DataFrame(all_tel)
            df_tel_piv = df_tel.pivot_table(index="prompt", columns="layer", values="deltaL2", fill_value=0.0)
            plt.figure(figsize=(max(6, 0.4*len(df_tel_piv.columns)), max(4, 0.2*len(df_tel_piv.index))))
            sns.heatmap(df_tel_piv, cmap="mako", cbar_kws={"label": "deltaL2"})
            plt.title("Per-prompt deltaL2 by layer")
            plt.tight_layout()
            plt.savefig(fig_dir / "deltaL2_heatmap.png", dpi=200)
            plt.close()

    # 5) Write a tiny README.md for the report folder
    with open(rep_dir / "report.md", "w") as f:
        f.write("# Report artifacts\n\n")
        f.write("## Tables\n")
        for p in sorted(tab_dir.glob("*.csv")):
            f.write(f"- {p.name}\n")
        for p in sorted(tab_dir.glob("*.txt")):
            f.write(f"- {p.name}\n")
        f.write("\n## Figures\n")
        for p in sorted(fig_dir.glob("*.png")):
            f.write(f"- {p.name}\n")

    print(f"Figures → {fig_dir}")
    print(f"Tables → {tab_dir}")
    print(f"Report → {rep_dir / 'report.md'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="he_pipeline_results")
    args = ap.parse_args()
    main(args.results_dir)
