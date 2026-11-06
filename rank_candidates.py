#!/usr/bin/env python3
"""Reproducible scoring and overlap-aware promo selection (Python 3 standard library)."""
import argparse
import csv
import json
from pathlib import Path

WEIGHTS = {"hook_score": .25, "cliffhanger_score": .25,
           "clarity_score": .25, "emotion_score": .15,
           "coherence_score": .10}
GATES = ("hook_pass", "cliffhanger_pass", "clarity_pass", "coherence_pass")

def overlap(a, b):
    if a["episode_number"] != b["episode_number"]:
        return 0.0
    intersection = max(0, min(a["end_ms"], b["end_ms"]) -
                       max(a["start_ms"], b["start_ms"]))
    shorter = min(a["end_ms"] - a["start_ms"], b["end_ms"] - b["start_ms"])
    return intersection / shorter if shorter > 0 else 0.0

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--evaluations", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--overlap-threshold", type=float, default=.5)
    args = ap.parse_args()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = payload["candidates"] if isinstance(payload, dict) else payload
    evaluations = {}
    for path in sorted(args.evaluations.glob("batch_*_response.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        for evaluation in batch["evaluations"]:
            cid = evaluation["candidate_id"]
            if cid in evaluations:
                raise ValueError(f"Duplicate evaluation: {cid}")
            evaluations[cid] = evaluation

    ids = {c["candidate_id"] for c in candidates}
    if len(ids) != len(candidates) or ids != set(evaluations):
        raise ValueError("Candidate/evaluation IDs do not match exactly")

    rows = []
    for c in candidates:
        e = evaluations[c["candidate_id"]]
        for metric in WEIGHTS:
            score = e[metric]
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 5:
                raise ValueError(f"Invalid score for {c['candidate_id']}: {metric}")
        eligible = (e["decision"] == "PASS" and
                    all(e[g] is True for g in GATES) and
                    e["boundary_cut_issue"] is False)
        if e["decision"] == "PASS" and not eligible:
            raise ValueError(f"Inconsistent PASS gates: {c['candidate_id']}")
        rows.append({
            "candidate_id": c["candidate_id"],
            "episode_number": c["episode_number"],
            "start": c["start_raw"], "end": c["end_raw"],
            "start_ms": c["start_ms"], "end_ms": c["end_ms"],
            "duration_s": c["duration_s"],
            "weighted_score": round(sum(e[k] * w for k, w in WEIGHTS.items()), 4),
            "decision": e["decision"], "eligible": eligible,
            "hook_evidence": e["hook_evidence"],
            "cliffhanger_evidence": e["cliffhanger_evidence"],
            "brief_reasoning": e["brief_reasoning"],
        })
    rows.sort(key=lambda r: (not r["eligible"], -r["weighted_score"], r["candidate_id"]))
    selected = []
    for r in rows:
        if not r["eligible"]:
            continue
        if all(overlap(r, s) < args.overlap_threshold for s in selected):
            selected.append(r)
        if len(selected) >= args.top:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "all_ranked_candidates.csv", rows)
    write_csv(args.output / "top20_diversified.csv", selected)
    summary = {"total": len(rows), "eligible": sum(r["eligible"] for r in rows),
               "shortlisted": len(selected), "overlap_threshold": args.overlap_threshold}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
