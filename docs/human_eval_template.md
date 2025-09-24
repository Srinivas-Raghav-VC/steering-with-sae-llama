# Human Evaluation Protocol (Template)

This template guides manual evaluation of steered outputs for:
- Language compliance (target language achieved?)
- Meaning preservation (same meaning as baseline?)
- Fluency/coherence (natural and grammatical?)

Instructions:
1. Prepare a CSV with columns: prompt, baseline, steered.
2. Randomize row order; blind raters to the model setting when possible.
3. For each row, assign scores 0..1 (or 1..5 Likert) for:
   - language_compliance
   - meaning_preservation
   - coherence_fluency
   - overall_success
4. Add optional comments for edge cases or failures.

Suggested CSV Header:
```
prompt,baseline,steered,language_compliance,meaning_preservation,coherence_fluency,overall_success,comments
```

Quality control:
- Include sentinel items (obvious good/bad) to check annotator reliability.
- Compute per-rater agreement (Spearman/Pearson, or Cohen’s kappa if categorical).

Deliverables to include in report:
- Mean ± CI for each metric
- Inter-rater agreement statistics
- 3–5 qualitative examples illustrating success/failure patterns
