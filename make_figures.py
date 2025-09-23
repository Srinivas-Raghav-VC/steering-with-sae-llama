#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate figures from SAE-only results JSON files.
# Usage:
#   python make_figures.py --inputs runs/*/results_sae_only.json --out figs

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (8, 4.5)
plt.rcParams["font.size"] = 11

def load_runs(patterns: List[str]) -> List[Dict]:
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    runs: List[Dict] = []
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                runs.append(json.load(fh))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return runs

def fig_layer_effectiveness(runs: List[Dict], outdir: Path):
    # Use first run with 'effectiveness'
    for r in runs:
        eff = r.get("effectiveness")
        if eff:
            layers = sorted(int(k) for k in eff.keys())
            succ = [eff[str(L)]["success_rate"] for L in layers]
            plt.figure()
            plt.bar([str(L) for L in layers], succ, color="#3b82f6")
            plt.ylim(0, 1.0)
            plt.ylabel("Flip rate (strict)")
            plt.xlabel("Layer (0-based)")
            plt.title("Per-layer steering effectiveness")
            out = outdir / "figure1_layer_effectiveness.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"Saved {out}")
            return

def bootstrap_mean_ci(values: List[float], iters: int = 2000, alpha: float = 0.05):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return (0.0, (0.0, 0.0))
    means = []
    n = len(arr)
    rng = np.random.default_rng(0)
    for _ in range(iters):
        samp = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(samp))
    means = np.sort(means)
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return float(np.mean(arr)), (float(lo), float(hi))

def fig_mid_vs_late(runs: List[Dict], outdir: Path):
    mid = []
    late = []
    for r in runs:
        eff = r.get("effectiveness", {})
        for k, v in eff.items():
            L = int(k)
            sr = float(v.get("success_rate", 0.0))
            if 18 <= L <= 22:
                mid.append(sr)
            if 29 <= L <= 31:
                late.append(sr)
    m_mean, m_ci = bootstrap_mean_ci(mid)
    l_mean, l_ci = bootstrap_mean_ci(late)
    labels = ["Mid (18–22)", "Late (29–31)"]
    means = [m_mean, l_mean]
    errlo = [m_mean - m_ci[0], l_mean - l_ci[0]]
    errhi = [m_ci[1] - m_mean, l_ci[1] - l_mean]
    plt.figure()
    plt.bar(labels, means, yerr=[errlo, errhi], capsize=6, color=["#10b981", "#ef4444"])
    plt.ylim(0, 1.0)
    plt.ylabel("Flip rate (strict)")
    plt.title("Mid vs Late Layer Steering Effectiveness")
    out = outdir / "figure2_mid_vs_late.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

def fig_scope_comparison(runs: List[Dict], outdir: Path):
    seq_vals = []
    last_vals = []
    for r in runs:
        cfg = r.get("config", {})
        scope = (cfg.get("intervention_scope") or "").lower()
        eff = r.get("effectiveness", {})
        # Aggregate selected layers only if present, else all
        chosen = r.get("selected_layers") or list(map(int, eff.keys()))
        vals = [float(eff[str(L)]["success_rate"]) for L in chosen if str(L) in eff]
        if len(vals) == 0:
            continue
        val = float(np.mean(vals))
        if scope == "sequence":
            seq_vals.append(val)
        elif scope == "last":
            last_vals.append(val)
    seq_m, seq_ci = bootstrap_mean_ci(seq_vals)
    last_m, last_ci = bootstrap_mean_ci(last_vals)
    lbl = ["Sequence-wide", "Last-token"]
    means = [seq_m, last_m]
    errlo = [seq_m - seq_ci[0], last_m - last_ci[0]]
    errhi = [seq_ci[1] - seq_m, last_ci[1] - last_m]
    plt.figure()
    plt.bar(lbl, means, yerr=[errlo, errhi], capsize=6, color=["#6366f1", "#f59e0b"])
    plt.ylim(0, 1.0)
    plt.ylabel("Flip rate (strict)")
    plt.title("Intervention Scope Comparison")
    out = outdir / "figure4_scope_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

def fig_strength_sweep(runs: List[Dict], outdir: Path):
    # Plot success vs steer-alpha across runs (averaged over chosen layers)
    points: List[Tuple[float, float]] = []
    for r in runs:
        cfg = r.get("config", {})
        alpha = float(cfg.get("eval_strength") or cfg.get("default_strength") or 0.0)
        eff = r.get("effectiveness", {})
        chosen = r.get("selected_layers") or list(map(int, eff.keys()))
        vals = [float(eff[str(L)]["success_rate"]) for L in chosen if str(L) in eff]
        if len(vals) == 0:
            continue
        points.append((alpha, float(np.mean(vals))))
    if not points:
        return
    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.figure()
    plt.plot(xs, ys, "-o", color="#22c55e")
    plt.ylim(0, 1.0)
    plt.xlabel("Steer alpha")
    plt.ylabel("Flip rate (strict)")
    plt.title("Strength Sweep")
    out = outdir / "figure5_strength_sweep.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

def fig_single_vs_multi(runs: List[Dict], outdir: Path):
    single = []
    multi = []
    for r in runs:
        eff = r.get("effectiveness", {})
        chosen = r.get("selected_layers") or list(map(int, eff.keys()))
        if len(chosen) == 1:
            # single-layer: use its success
            L = chosen[0]
            if str(L) in eff:
                single.append(float(eff[str(L)]["success_rate"]))
        elif len(chosen) > 1:
            vals = [float(eff[str(L)]["success_rate"]) for L in chosen if str(L) in eff]
            if len(vals) > 0:
                multi.append(float(np.mean(vals)))
    sm, sm_ci = bootstrap_mean_ci(single)
    mm, mm_ci = bootstrap_mean_ci(multi)
    if len(single) == 0 and len(multi) == 0:
        return
    labels = ["Single", "Multi"]
    means = [sm, mm]
    errlo = [sm - sm_ci[0], mm - mm_ci[0]]
    errhi = [sm_ci[1] - sm, mm_ci[1] - mm]
    plt.figure()
    plt.bar(labels, means, yerr=[errlo, errhi], capsize=6, color=["#0ea5e9", "#ef4444"])
    plt.ylim(0, 1.0)
    plt.ylabel("Flip rate (strict)")
    plt.title("Single vs Multi-layer (avg over chosen layers)")
    out = outdir / "figure3_single_vs_multi.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Glob(s) to results JSON files (e.g., runs/*/results_sae_only.json)",
    )
    ap.add_argument("--out", default="figs", help="Output directory")
    args = ap.parse_args()

    runs = load_runs(args.inputs)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not runs:
        print("No runs loaded.")
        return

    fig_layer_effectiveness(runs, outdir)
    fig_mid_vs_late(runs, outdir)
    fig_single_vs_multi(runs, outdir)
    fig_scope_comparison(runs, outdir)
    fig_strength_sweep(runs, outdir)

if __name__ == "__main__":
    main()
