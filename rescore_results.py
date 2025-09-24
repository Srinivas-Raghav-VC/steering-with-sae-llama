# rescore_results.py
import json
import sys

from tools.lang_detect import classify_language, normalized_edit_distance, TEXT_CHANGE_THRESHOLD


def main(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    recs = data.get("results", [])
    total = 0
    flips = 0
    changed = 0

    for r in recs:
        baseline = (r.get("baseline") or "").strip()
        steered = (r.get("steered") or "").strip()
        if not baseline or not steered:
            continue

        baseline_lang, _ = classify_language(baseline)
        if baseline_lang == "hindi":
            total += 1
            if normalized_edit_distance(baseline, steered) >= TEXT_CHANGE_THRESHOLD:
                changed += 1
            steered_lang, _ = classify_language(steered)
            if steered_lang == "english" and normalized_edit_distance(baseline, steered) >= TEXT_CHANGE_THRESHOLD:
                flips += 1

    rate = 100.0 * flips / max(1, total)
    print(
        f"Strict flips (Hindi baseline -> English steered): {flips}/{total} "
        f"({rate:.1f}%). Changed-text among those baselines: {changed}/{total}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "he_pipeline_results/results.json")
