# Kuku FM Promo Finder — Problem 1

Screenplay-based promo discovery for *I Married a Broken Soldier. He was a Duke*.

## Quick start

Requires Python 3.9+; no external dependencies.

```bash
python build_phase4.py
```

The command reads six input files in `inputs/`, validates 162 candidates and 17 batches of evaluations, and recreates the full ranking under `output/`. It does **not** call Claude live: the human/Claude-assisted evaluations are frozen in the supplied JSON.

## Pipeline

1. `src/parse_screenplays.py` extracts screenplay events and timestamps from original screenplay input.
2. `src/generate_candidates.py` creates 20–60-second contiguous clip candidates with 15-second lookahead.
3. `evaluations/` holds the calibrated Claude-assisted evaluations for all 162 candidates.
4. `build_phase4.py` validates IDs/gates/scores, ranks candidates, and selects non-overlapping Top 20.
5. `outputs/` contains raw results, eligible clips, Top 20 editorial review, and Top 10 showcase.

## Reproducibility

The `inputs/` folder is the authoritative, self-contained input for the ranking script. `data/` provides extracted candidate records for inspection. Run `python build_phase4.py` from the repository root. The script writes to `output/` (generated).

## Results

- 50 episodes, 162 generated/evaluated clips
- 80 PASS and 82 REJECT
- 20 non-overlapping shortlisted clips, plus editorial Top 10

## Limitations and data permissions

All timings and quality assessments are based on screenplay text; final clips **have not been checked against video**. `C015-01` has a timestamp/evidence inconsistency and needs manual review. The source screenplay is not included in this repo; only derived records are present. Do not make the repository public or redistribute screenplay content without Kuku FM permission.

## Submission

See `docs/Kuku_FM_Solution_2page.pdf` for the two-page solution summary. Repository: https://github.com/TusharIITg01/kuku-fm-promo-finder
