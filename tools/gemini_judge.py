#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge (Gemini) evaluation for SAE steering results.

Usage (PowerShell):
  $env:GEMINI_API_KEY = "<your_api_key>"
  python tools/gemini_judge.py --results he_pipeline_results/results_sae_only.json --out he_pipeline_results/llm_judge_gemini.json

This script reads the pipeline's results JSON and calls Gemini to score each
record on:
  - language_compliance (is steered output in English for shadow_hindi?)
  - meaning_preservation (relative to baseline)
  - coherence_fluency (English quality)
  - overall_success (holistic)

Notes:
  - Requires google-generativeai (see requirements.txt) and GEMINI_API_KEY env.
    - Runs post-hoc; does not make API calls inside training/extraction loops.
    - Safe to sub-sample with --limit for cost control.
    - Treat these scores as supplemental; include human review for publication-level results.
"""

import argparse
import json
import os
import sys
import time
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)


class JudgeScores(BaseModel):
    """Structured response model for Gemini judge scores."""
    language_compliance: float = Field(ge=0.0, le=1.0, description="How well does the steered output comply with target language?")
    meaning_preservation: float = Field(ge=0.0, le=1.0, description="How well is the original meaning preserved?")
    coherence_fluency: float = Field(ge=0.0, le=1.0, description="How coherent and fluent is the steered output?")
    overall_success: float = Field(ge=0.0, le=1.0, description="Overall success of the steering intervention")
    explanation: str = Field(description="Brief explanation of the scores")


GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.0,
    "response_mime_type": "application/json",
}


def _call_gemini_structured(model, prompt: str) -> Dict[str, Any]:
    """Call Gemini with structured JSON response validation."""
    # Try to use gemini-groundcite if installed
    try:
        from gemini_groundcite import ensure_json  # type: ignore
        return ensure_json(model=model, prompt=prompt, response_model=JudgeScores)
    except Exception:
        ensure_json = None

    # Fallback to manual structured response
    full_prompt = f"""{prompt}

Please respond with a valid JSON object matching this exact schema:
{{
    "language_compliance": <float between 0.0 and 1.0>,
    "meaning_preservation": <float between 0.0 and 1.0>,
    "coherence_fluency": <float between 0.0 and 1.0>,
    "overall_success": <float between 0.0 and 1.0>,
    "explanation": "<brief explanation string>"
}}"""

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=GENERATION_CONFIG,
        )

        raw_text = getattr(response, "text", "") or ""

        if not raw_text:
            return {"error": f"No text response (finish_reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'})"}

        parsed = _safe_json_parse(raw_text) or {}

        # Validate with Pydantic
        validated = JudgeScores(**parsed)
        return validated.model_dump()
    except ValidationError as e:
        raise ValueError(f"Judge response failed validation: {e}") from e
    except Exception as e:
        raise ValueError(f"Gemini API call failed: {e}") from e


def _build_judge_prompt(prompt: str, baseline: str, steered: str) -> str:
    """Construct a sanitized JSON-based prompt for Gemini judging.
    Encapsulates inputs in a JSON block to reduce injection risk and keeps instructions concise.
    """
    payload = {
        "task": "Evaluate language translation quality between baseline and modified outputs.",
        "goal": "Transform Hindi input to English output while preserving meaning.",
        "inputs": {
            "prompt": prompt,
            "baseline": baseline,
            "steered": steered,
        },
        "instructions": [
            "Rate each aspect from 0.0 to 1.0",
            "Respond ONLY with a JSON object matching the requested schema",
        ],
        "schema": {
            "language_compliance": "float 0..1",
            "meaning_preservation": "float 0..1",
            "coherence_fluency": "float 0..1",
            "overall_success": "float 0..1",
            "explanation": "short string",
        },
    }
    return (
        "You are a careful evaluator. Read the JSON payload and return a JSON object with the requested numeric scores.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)[:6000]}\n\n"
        "Return a JSON object with keys: language_compliance, meaning_preservation, coherence_fluency, overall_success, explanation."
    )


def _safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from model output.
    Gemini may wrap JSON in code fences; we attempt to locate braces and parse.
    """
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results_sae_only.json")
    ap.add_argument("--out", required=True, help="Where to write LLM-judge scores JSON")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    ap.add_argument("--limit", type=int, default=0, help="Max records to score (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Delay between calls (seconds)")
    ap.add_argument("--dry-run", action="store_true", help="Print prompts only, no API calls")
    ap.add_argument(
        "--update-results",
        action="store_true",
        help="Also write a 'quality_aware' summary back into the results JSON",
    )
    ap.add_argument("--cache", type=str, default=None, help="Path to a JSON cache file (reused across runs)")
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[Dict[str, Any]] = data.get("results", [])
    mode = (data.get("config", {}).get("eval_mode") or "shadow_hindi").lower()

    if not records:
        print("No records found in results; exiting.")
        sys.exit(1)

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    out_items: List[Dict[str, Any]] = []

    # Load cache if requested
    cache: Dict[str, Any] = {}
    cache_path: Optional[Path] = None
    if args.cache:
        cache_path = Path(args.cache)
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

    if args.dry_run:
        for r in records[:3]:  # show first few
            prompt = _build_judge_prompt(
                r.get("prompt", ""), r.get("baseline", ""), r.get("steered", "")
            )
            print("\n----- EVAL PROMPT SAMPLE -----\n")
            print(prompt[:1200])
        print("\n[Dry run complete]")
        return

    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set; set it in your environment.")
        sys.exit(2)

    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai not installed. Add it to requirements and pip install.")
        sys.exit(3)

    genai.configure(api_key=api_key)

    # Configure safety settings to avoid over-blocking evaluation content
    safety_settings = {
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
    }

    try:
        model = genai.GenerativeModel(args.model, safety_settings=safety_settings)
    except Exception:
        model = genai.GenerativeModel(args.model)

    for i, r in enumerate(records):
        prompt = _build_judge_prompt(
            r.get("prompt", ""), r.get("baseline", ""), r.get("steered", "")
        )
        # Hash key by prompt/baseline/steered
        h = hashlib.sha256()
        for s in (r.get("prompt", ""), r.get("baseline", ""), r.get("steered", "")):
            h.update(s.encode("utf-8", errors="ignore"))
            h.update(b"\x00")
        key = h.hexdigest()

        if key in cache:
            parsed = cache[key]
        else:
            try:
                parsed = _call_gemini_structured(model, prompt)
            except Exception as e:
                parsed = {"error": str(e)}
            # Save to cache
            cache[key] = parsed
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

        item = {
            "prompt": r.get("prompt", ""),
            "baseline": r.get("baseline", ""),
            "steered": r.get("steered", ""),
            "judge": parsed,
        }
        out_items.append(item)
        if args.sleep > 0:
            time.sleep(args.sleep)

        if (i + 1) % 10 == 0:
            print(f"Scored {i+1}/{len(records)}")

    # Aggregate simple means over numeric keys
    agg: Dict[str, float] = {}
    cnt: Dict[str, int] = {}
    keys = [
        "language_compliance",
        "meaning_preservation",
        "coherence_fluency",
        "overall_success",
    ]
    for x in out_items:
        j = x.get("judge") or {}
        for k in keys:
            v = j.get(k)
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0.0) + float(v)
                cnt[k] = cnt.get(k, 0) + 1
    summary = {k: (agg.get(k, 0.0) / max(1, cnt.get(k, 0))) for k in keys}

    summary_with_count = dict(summary)
    summary_with_count["count"] = len(out_items)

    out = {
        "mode": mode,
        "model": args.model,
        "scored": len(out_items),
        "summary": summary_with_count,
        "records": out_items,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved LLM-judge results → {out_path}")

    # Optionally, compute strict flip rate from results and write back an aggregated
    # 'quality_aware' block into the main results JSON so downstream tools don't need to merge.
    if args.update_results:
        try:
            # Load main results
            res_path = Path(args.results)
            with open(res_path, "r", encoding="utf-8") as rf:
                res = json.load(rf)

            # Compute strict flip rate from res['results']
            recs = res.get("results", [])
            mode_res = (res.get("config", {}).get("eval_mode") or mode or "shadow_hindi").lower()
            target = "english" if mode_res == "shadow_hindi" else "hindi"
            opposite = "hindi" if target == "english" else "english"
            eligible = [r for r in recs if r.get("baseline_lang") == opposite]
            flips = 0
            for r in eligible:
                if r.get("steered_lang") == target:
                    # guard against trivial copies
                    if (r.get("steered") or "").strip() != (r.get("baseline") or "").strip():
                        flips += 1
            strict_flip_rate = (flips / len(eligible)) if eligible else None

            # Build quality-aware combined score
            judge_overall = summary.get("overall_success")
            if strict_flip_rate is not None and isinstance(judge_overall, (int, float)):
                combined = 0.5 * float(strict_flip_rate) + 0.5 * float(judge_overall)
            else:
                combined = float(judge_overall) if isinstance(judge_overall, (int, float)) else None

            quality_aware = {
                "strict_flip_rate": strict_flip_rate,
                "judge_summary": summary_with_count,
                "combined_success": combined,
                "source": {
                    "llm_judge_file": str(out_path),
                    "model": args.model,
                },
                "scored_examples": len(out_items),
            }

            # Attach to results JSON
            res.setdefault("quality_aware_history", [])
            res["quality_aware"] = quality_aware
            res["quality_aware_history"].append(
                {
                    "timestamp": time.time(),
                    "quality_aware": quality_aware,
                }
            )
            if combined is not None:
                res["quality_aware_success"] = combined

            with open(res_path, "w", encoding="utf-8") as wf:
                json.dump(res, wf, indent=2, ensure_ascii=False)
            print(
                f"Updated results with quality-aware success → {res_path} (combined={combined})"
            )
        except Exception as e:
            print(f"Warning: failed to update results JSON with quality-aware scores: {e}")


if __name__ == "__main__":
    main()
