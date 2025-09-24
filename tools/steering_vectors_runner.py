#!/usr/bin/env python3
"""
Steering-vector baseline hook for comparing dense vs sparse interventions.
Uses the steering-vectors toolkit to implement dense steering baselines.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def main():
    ap = argparse.ArgumentParser(description="Run steering-vector baseline comparison")
    ap.add_argument("--results", default="he_pipeline_results/results_sae_only.json",
                    help="Path to SAE results JSON file")
    ap.add_argument("--out", default="he_pipeline_results/steering_vectors.json",
                    help="Output path for steering vector results")
    ap.add_argument("--layers", default="18,19,20",
                    help="Comma-separated layer indices")
    ap.add_argument("--strength", type=float, default=2.0,
                    help="Steering strength")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Model name")
    args = ap.parse_args()

    try:
        from steering_vectors import api  # pip install steering-vectors
    except ImportError:
        print("Error: steering-vectors not installed. Run: pip install steering-vectors")
        print("See: https://steering-vectors.github.io/steering-vectors/")
        sys.exit(1)

    # Parse layers
    try:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    except ValueError as e:
        print(f"Error parsing layers '{args.layers}': {e}")
        sys.exit(1)

    if not layers:
        print("Error: No valid layers specified")
        sys.exit(1)

    print(f"🚀 Running steering-vector baseline on layers {layers}")

    try:
        # Derive steering vectors for language attribute
        print("📊 Deriving steering vectors...")
        vectors = api.derive_vectors(
            model_name=args.model,
            layers=layers,
            attribute="language:english_minus_hindi",
        )
        print(f"✅ Derived vectors for {len(vectors)} layers")

        # Evaluate vectors against SAE results
        print(f"🧪 Evaluating with strength {args.strength}...")
        report = api.evaluate_vectors(
            vectors=vectors,
            strength=args.strength,
            results_file=args.results,
        )
        print("✅ Evaluation complete")

        # Save results
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata to report
        report_with_meta = {
            "config": {
                "model": args.model,
                "layers": layers,
                "strength": args.strength,
                "source_results": str(args.results)
            },
            "results": report
        }

        output_path.write_text(
            json.dumps(report_with_meta, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"💾 Saved steering-vector baseline → {args.out}")

        # Print summary stats
        if isinstance(report, dict) and "summary" in report:
            summary = report["summary"]
            print("\n📈 Summary:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error running steering-vector baseline: {e}")
        print("Make sure steering-vectors is properly installed and the model is accessible")
        sys.exit(1)

if __name__ == "__main__":
    main()
