#!/usr/bin/env python3
"""
Convert he_pipeline_results/results_sae_only.json into TSVs that
hi-en-bias-eval and MIPE can ingest.
"""

import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="he_pipeline_results/results_sae_only.json")
    ap.add_argument("--out-dir", default="he_pipeline_results/bias_eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for rec in data.get("results", []):
        rows.append({
            "prompt": (rec.get("prompt", "") or "").replace("\t", " "),
            "baseline": (rec.get("baseline", "") or "").replace("\t", " "),
            "steered": (rec.get("steered", "") or "").replace("\t", " "),
            "baseline_lang": rec.get("baseline_lang", "") or "",
            "steered_lang": rec.get("steered_lang", "") or ""
        })

    # 1) TSV for hi-en-bias-eval prompts
    with open(out_dir / "steered_outputs.tsv", "w", encoding="utf-8") as f:
        f.write("prompt\tbaseline\tsteered\tbaseline_lang\tsteered_lang\n")
        for r in rows:
            f.write("{prompt}\t{baseline}\t{steered}\t{baseline_lang}\t{steered_lang}\n".format(**r))

    # 2) JSONL for MIPE (prompt + continuation)
    with open(out_dir / "steered_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "prompt": r["prompt"],
                "continuation": r["steered"]
            }, ensure_ascii=False) + "\n")

    print(f"Wrote bias-eval exports to {out_dir}")

if __name__ == "__main__":
    main()
