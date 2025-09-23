# rescore_results.py
import json, re, sys

def dev_ratio(s: str) -> float:
    dev = 0; alpha = 0
    for ch in s:
        if ch.isalpha():
            alpha += 1
            if '\u0900' <= ch <= '\u097f':
                dev += 1
    return dev / float(alpha or 1)

ROMAN_HI = set("""
hai hain hoon raha rahe rahi mera meri mere kya nahi ka ki ke mein hum aap tum bhai yaar ghar pyar
""".split())
WORD_RE = re.compile(r"[A-Za-z']+")

def roman_hi_ratio(text: str) -> float:
    toks = WORD_RE.findall(text.lower())
    if not toks: return 0.0
    hits = sum(1 for t in toks if t in ROMAN_HI)
    return hits / len(toks)

def is_hindi_like(s: str) -> bool:
    return dev_ratio(s) >= 0.25 or roman_hi_ratio(s) >= 0.08

def is_english_like(s: str) -> bool:
    return dev_ratio(s) <= 0.05 and roman_hi_ratio(s) < 0.05

def main(path):
    with open(path, "r") as f:
        data = json.load(f)
    recs = data.get("results", [])
    total = 0
    flips = 0
    changed = 0
    for r in recs:
        b = (r.get("baseline") or "").strip()
        s = (r.get("steered") or "").strip()
        if not b or not s:
            continue
        if is_hindi_like(b):  # only count Hindi baselines
            total += 1
            if s != b:
                changed += 1
            if is_english_like(s):  # strict Hindi -> English flip
                flips += 1
    print(f"Strict flips (Hindi baseline -> English steered): {flips}/{total} "
          f"({100.0*flips/max(1,total):.1f}%). Changed-text among those baselines: {changed}/{total}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "he_pipeline_results/results.json"
    main(path)
