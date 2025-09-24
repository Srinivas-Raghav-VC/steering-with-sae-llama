#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate and evaluate the shared language detector (tools/lang_detect.py).

Inputs: a labeled dataset with columns/text fields containing the input text and its gold label.
Supported formats: JSONL (fields: text, label|lang) or CSV (columns: text, label|lang).

Outputs:
- he_pipeline_results/detector_calibration.json: summary metrics and confusion matrix
- he_pipeline_results/figures/detector_confusion.png: confusion heatmap

Usage (PowerShell):
  python tools/calibrate_detector.py --input data/labeled_hi_en.jsonl --limit 5000

Labels expected: english, hindi, mixed, unknown (case-insensitive). Other labels will be mapped to unknown.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools.lang_detect import classify_language, gemini_enabled


LABELS = ["english", "hindi", "mixed", "unknown"]


def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    if s in LABELS:
        return s
    # Map variants
    if s in {"en", "eng"}:
        return "english"
    if s in {"hi", "hin", "hnd", "hindi-dev"}:
        return "hindi"
    if s in {"code-mixed", "hinglish", "code_mix", "code-mix"}:
        return "mixed"
    return "unknown"


def _iter_jsonl(path: Path, text_field: str = "", label_field: str = "") -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (
                rec.get(text_field)
                or rec.get("text")
                or rec.get("sentence")
                or rec.get("input")
                or ""
            ).strip()
            label = _normalize_label(
                rec.get(label_field)
                or rec.get("label")
                or rec.get("lang")
                or rec.get("gold")
                or ""
            )
            if not text:
                continue
            yield text, label


def _iter_csv(path: Path, text_field: str = "", label_field: str = "") -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                row.get(text_field)
                or row.get("text")
                or row.get("sentence")
                or row.get("input")
                or ""
            ).strip()
            label = _normalize_label(
                row.get(label_field)
                or row.get("label")
                or row.get("lang")
                or row.get("gold")
                or ""
            )
            if not text:
                continue
            yield text, label


def load_labeled(path: Path, limit: int = 0, text_field: str = "", label_field: str = "") -> List[Tuple[str, str]]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        it = _iter_jsonl(path, text_field=text_field, label_field=label_field)
    elif ext == ".csv":
        it = _iter_csv(path, text_field=text_field, label_field=label_field)
    else:
        raise SystemExit(f"Unsupported file format: {ext} (use .jsonl or .csv)")
    out: List[Tuple[str, str]] = []
    for i, (text, label) in enumerate(it):
        out.append((text, label))
        if limit and len(out) >= limit:
            break
    return out


def evaluate_detector(pairs: List[Tuple[str, str]]) -> Dict:
    # Confusion matrix: gold x pred
    idx = {lab: i for i, lab in enumerate(LABELS)}
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    records = []

    for text, gold in pairs:
        pred, conf = classify_language(text)
        if pred not in idx:
            pred = "unknown"
        cm[idx[gold], idx[pred]] += 1
        records.append({"text": text, "gold": gold, "pred": pred, "conf": float(conf)})

    total = int(cm.sum())
    acc = float(np.trace(cm) / max(1, total))
    per_label = {}
    for i, lab in enumerate(LABELS):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, (prec + rec)) if (prec + rec) > 0 else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}

    macro_f1 = float(np.mean([v["f1"] for v in per_label.values()]))

    return {
        "used_gemini": gemini_enabled(),
        "labels": LABELS,
        "confusion": cm.tolist(),
        "total": total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "records": records,
    }


def save_confusion_figure(fig_path: Path, summary: Dict):
    labels = summary.get("labels", LABELS)
    cm = np.array(summary.get("confusion", [[0]]), dtype=float)
    # Normalize rows
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title("Language Detector Confusion (row-normalized)")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Calibrate/evaluate the shared language detector")
    ap.add_argument("--input", required=True, help="Path to labeled data (.jsonl or .csv)")
    ap.add_argument("--out", default="he_pipeline_results/detector_calibration.json", help="Output JSON path")
    ap.add_argument("--limit", type=int, default=0, help="Max examples to evaluate (0 = all)")
    ap.add_argument("--dataset-name", default=None, help="Name of the evaluation dataset for metadata")
    ap.add_argument("--text-field", default="", help="Explicit text column name (optional)")
    ap.add_argument("--label-field", default="", help="Explicit label column name (optional)")
    ap.add_argument(
        "--compare-heuristic",
        action="store_true",
        help="Also evaluate with Gemini disabled to compare heuristic-only performance",
    )
    ap.add_argument(
        "--no-gemini",
        action="store_true",
        help="Force-disable Gemini fallback for the main evaluation run",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    pairs = load_labeled(
        in_path,
        limit=args.limit,
        text_field=args.text_field or "",
        label_field=args.label_field or "",
    )
    if not pairs:
        raise SystemExit("No labeled examples loaded. Check input format and columns.")

    print(f"Evaluating on {len(pairs)} examples...")
    # Optional disable Gemini for the main pass
    previous_flag = os.environ.get("DISABLE_GEMINI_DETECTOR")
    if args.no_gemini:
        os.environ["DISABLE_GEMINI_DETECTOR"] = "1"

    summary = evaluate_detector(pairs)

    # Optional comparison run with heuristic-only fallback
    comparison = None
    if args.compare_heuristic and os.environ.get("DISABLE_GEMINI_DETECTOR") != "1":
        try:
            os.environ["DISABLE_GEMINI_DETECTOR"] = "1"
            comparison = evaluate_detector(pairs)
        finally:
            if previous_flag is None:
                os.environ.pop("DISABLE_GEMINI_DETECTOR", None)
            else:
                os.environ["DISABLE_GEMINI_DETECTOR"] = previous_flag

    if previous_flag is None and not args.compare_heuristic and args.no_gemini is False:
        os.environ.pop("DISABLE_GEMINI_DETECTOR", None)
    elif previous_flag is not None:
        os.environ["DISABLE_GEMINI_DETECTOR"] = previous_flag

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": args.dataset_name or in_path.stem,
        "num_examples": summary["total"],
        "summary": summary,
    }
    if comparison:
        payload["heuristic_only"] = comparison

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved calibration summary → {out_path}")

    # Save a confusion heatmap
    fig_dir = out_path.parent / "figures"
    save_confusion_figure(fig_dir / "detector_confusion.png", summary)
    print(f"Saved confusion heatmap → {fig_dir / 'detector_confusion.png'}")


if __name__ == "__main__":
    main()
