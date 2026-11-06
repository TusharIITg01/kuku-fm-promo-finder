# Kuku FM — Promo Finder (Problem 1)

Screenplay-based promo candidate ranking for *I Married a Broken Soldier. He was a Duke*.

## Status

The ranking prototype is committed here. The **complete 46-file reproducible package** (including 17 evaluation batches, derived candidate data, full output tables, and the two-page PDF) has been generated and **tested locally**, but has **not yet been uploaded to this repository**. Do not use this GitHub link as the final submission until the complete package is uploaded.

## Ranking

The script `rank_candidates.py` accepts the Phase 2 `promo_candidates.json` and 17 `batch_XX_response.json` evaluation files:

```bash
python rank_candidates.py --candidates data/promo_candidates.json --evaluations evaluations --output outputs
```

It validates IDs and score ranges, applies four editorial quality gates, computes a weighted score (hook 25%, cliffhanger 25%, clarity 25%, emotion 15%, coherence 10%), and selects a shortlist with overlap suppression.

## Verified full-pipeline results (local)

- 50 episodes parsed
- 162 candidates evaluated: 80 PASS, 82 REJECT
- Top 20 shortlisted
- Zero Phase 4 validation errors
- Rebuilt successfully with Python 3 standard library
- Video verification pending

## Data and copyright

The original screenplay is third-party assessment material. Keep this repository **private** until Kuku FM permits redistribution. The complete local package contains derived screenplay text and should also be treated as confidential.

## Known editorial issue

Candidate C015-01 has evidence that falls outside its proposed clip window; review manually before use. Scores are editorial screenplay assessments, not observed ad-performance metrics.
