#!/usr/bin/env python3
"""
Kuku FM microdrama promo MVP - Phase 2: deterministic promo candidate generation.

Input : parsed_episodes.json (and, optionally, parsed_events.csv) from Phase 1.
Output: promo_candidates.json, promo_candidates.csv, candidate_generation_summary.json

This script makes NO semantic judgement and calls NO LLM or external API.  It
only builds a small, diverse, fully traceable set of candidate windows for a
later evaluation phase.  Every structural signal below is a cheap heuristic
used for pruning; none of them is a hook, cliffhanger or quality score, and
the output is ordered chronologically, not ranked.

Usage
-----
    python3 generate_candidates.py
    python3 generate_candidates.py --parsed-json parser_output/parsed_episodes.json \
        --events-csv parser_output/parsed_events.csv --out-dir candidate_output \
        --aliases aliases.json --target-per-episode 4

Algorithm (details in the README block at the bottom of this docstring)
-----------------------------------------------------------------------
1. Per scene, take valid source events only (both timestamps present, end >= start).
2. Candidate boundaries are existing event times: start = some event's start,
   end = the end of some DIALOGUE / ACTION / SOUND_FX event (a CAMERA end alone
   is a shot change, not a story beat).  Nothing is interpolated.
3. Enumerate every (start, end) pair inside ONE scene and apply hard rules in a
   fixed order; every rejection is counted (first failure and any failure).
4. Compute structural features for survivors and a heuristic prune score.
5. Per episode, greedily pick up to N windows by prune score with a hard
   overlap limit (IoU) and a small same-scene penalty, so picks are diverse.
   If nothing passes, the episode simply gets zero candidates.
6. Attach up to 15 s of post-clip screenplay ("lookahead"), kept separate and
   never included in the duration.
7. Re-validate every candidate against the Phase 1 data, independently.

Duration is always end - start of the window.  CAMERA / ACTION / DIALOGUE /
SOUND_FX run on parallel tracks and are never summed.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VERSION = "1.0.1"
NON_CAMERA = ("DIALOGUE", "ACTION", "SOUND_FX")
BEAT_END_CATEGORIES = {"DIALOGUE", "ACTION", "SOUND_FX"}

# Hard rules, in evaluation order (this order defines "first failure").
HARD_RULES = [
    "outside_scene_bounds",         # window not inside one scene (covers cross-scene)
    "duration_below_min",
    "duration_above_max",
    "end_on_camera_only",           # end boundary is only a CAMERA shot change
    "dialogue_straddles_boundary",  # a spoken line would be cut in half
    "fragmented_sentence_start",    # first line continues an ellipsis-split sentence
    "fragmented_sentence_end",      # last line is continued by an ellipsis-split sentence
    "no_early_event",               # no dialogue/action/sfx begins within the first N seconds
    "too_few_dialogue_events",
    "dead_air_too_long",
]
CHEAP_RULES = set(HARD_RULES[:4])

# Generic English conflict vocabulary: a lexical count only, not a judgement.
CONFLICT_RE = re.compile(
    r"\b(?:arrest\w*|guards?|hang\w*|gallows|poison\w*|kill\w*|murder\w*|die[sd]?|dying|dead|death|"
    r"lie[sd]?|liar|traitor\w*|treason\w*|betray\w*|witch\w*|chain\w*|prison\w*|dungeon\w*|"
    r"execut\w*|blood\w*|wound\w*|knife|sword|fire|burn\w*|threat\w*|danger\w*|enem(?:y|ies)|"
    r"reveng\w*|vengeance|punish\w*|never|swear|sworn|accus\w*|guilty|innocent|frame[sd]?|"
    r"steal|stole\w*|thie(?:f|ves)|attack\w*|fight\w*|escape\w*|explo\w+|crime\w*|forbid\w*|"
    r"disown\w*|banish\w*|exile\w*|curse\w*|beg(?:ging)?|dare)\b", re.I)
TS_RE = re.compile(r"^(\d{1,3}):(\d{2})(?::(\d{2}))?(?:[.,](\d+))?$")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    min_s: float = 20.0
    max_s: float = 60.0
    lookahead_s: float = 15.0
    target_per_episode: int = 4
    max_iou: float = 0.20              # selected candidates may overlap at most this much
    early_event_s: float = 3.0         # a non-camera event must begin this soon
    min_dialogue_events: int = 2
    max_dead_air_s: float = 8.0        # longest stretch without dialogue/action/sfx
    fragment_gap_s: float = 1.5        # ellipsis-split sentence detection window
    same_scene_penalty: float = 0.05   # diversity nudge per earlier pick in the same scene
    min_prune_score: float = 0.35      # floor so weak windows are not forced
    cluster_iou: float = 0.5           # only for near-duplicate statistics


# --------------------------------------------------------------------------- #
# Lightweight views over the Phase 1 data
# --------------------------------------------------------------------------- #
class Ev:
    """One source event (read-only view of a Phase 1 event dict)."""
    __slots__ = ("id", "cat", "speaker_raw", "speaker", "delivery", "mods", "s", "e", "sraw", "eraw",
                 "text", "clean", "scene_idx", "scene_id", "order", "flags", "src")

    def __init__(self, d: Dict[str, Any], canon: Dict[str, str]):
        self.id = d["event_id"]
        self.cat = d["category"]
        self.speaker_raw = d.get("speaker")
        self.speaker = canon.get(self.speaker_raw, self.speaker_raw) if self.speaker_raw else None
        self.delivery = d.get("delivery")
        self.mods = d.get("voice_modifiers") or []
        self.s = d.get("start_ms")
        self.e = d.get("end_ms")
        self.sraw = d.get("start_raw")
        self.eraw = d.get("end_raw")
        self.text = d.get("text") or ""
        self.clean = d.get("text_clean") or self.text
        self.scene_idx = d.get("scene_index")
        self.scene_id = d.get("scene_id")
        self.order = d.get("file_order", 0)
        self.flags = d.get("flags") or []
        self.src = d


def load_aliases(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[Tuple[str, int], bool]]:
    """alias -> canonical, plus {(alias, episode): True} for low-confidence links."""
    canon: Dict[str, str] = {}
    low: Dict[Tuple[str, int], bool] = {}
    if path is None:
        return canon, low
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, spec in data.items():
        if name.startswith("_"):
            continue
        for alias in spec.get("aliases", []):
            canon[alias.upper()] = name.upper()
        for alias, eps in (spec.get("low_confidence_episodes") or {}).items():
            for ep in eps:
                low[(alias.upper(), int(ep))] = True
    return canon, low


def mini_parse_ts(raw: str) -> Optional[int]:
    """Independent re-parse of 'MM:SS.ss' used only by the validator."""
    m = TS_RE.match(raw.strip()) if raw else None
    if not m:
        return None
    a, b, c, frac = m.groups()
    if c is not None:
        h, mi, se = int(a), int(b), int(c)
    else:
        h, mi, se = 0, int(a), int(b)
    ms = ((h * 60 + mi) * 60 + se) * 1000
    if frac:
        ms += int(frac.ljust(3, "0")[:3])
    return ms


# --------------------------------------------------------------------------- #
# Small interval helpers (never sum parallel tracks)
# --------------------------------------------------------------------------- #
def union_length(intervals: List[Tuple[int, int]], lo: int, hi: int) -> int:
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_end = 0, lo
    for a, b in clipped:
        a = max(a, cur_end)
        if b > a:
            total += b - a
            cur_end = b
    return total


def longest_gap(intervals: List[Tuple[int, int]], lo: int, hi: int) -> int:
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b >= lo and a <= hi)
    cur, best = lo, 0
    for a, b in clipped:
        if a > cur:
            best = max(best, a - cur)
        cur = max(cur, b)
    return max(best, hi - cur)


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def intersects(ev: Ev, s: int, e: int) -> bool:
    """Positive-length overlap with [s, e]; zero-length events count if inside."""
    if ev.s == ev.e:
        return s <= ev.s <= e
    return ev.s < e and ev.e > s


def ends_with_ellipsis(t: str) -> bool:
    return t.rstrip().rstrip("\"”").rstrip().endswith(("...", "…"))


def starts_with_ellipsis(t: str) -> bool:
    return t.lstrip().lstrip("\"“").lstrip().startswith(("...", "…"))


# --------------------------------------------------------------------------- #
# Per-episode preparation
# --------------------------------------------------------------------------- #
class EpisodeData:
    def __init__(self, ep: Dict[str, Any], canon: Dict[str, str]):
        self.raw = ep
        self.number = ep["episode_number"]
        self.events_all = [Ev(d, canon) for d in ep["events"]]
        self.excluded: List[Dict[str, str]] = []
        self.valid: List[Ev] = []
        for ev in self.events_all:
            if ev.s is None or ev.e is None:
                self.excluded.append({"event_id": ev.id, "reason": "invalid_or_missing_timestamp"})
            elif ev.e < ev.s:
                self.excluded.append({"event_id": ev.id, "reason": "negative_duration"})
            elif ev.scene_idx is None:
                self.excluded.append({"event_id": ev.id, "reason": "event_without_scene"})
            else:
                self.valid.append(ev)
        self.valid.sort(key=lambda x: (x.s, x.e, x.order))
        self.scenes: Dict[int, Dict[str, Any]] = {}
        self.excluded_scenes: List[Dict[str, Any]] = []
        for sc in ep["scenes"]:
            if sc["start_ms"] is None or sc["end_ms"] is None or sc["end_ms"] <= sc["start_ms"]:
                self.excluded_scenes.append({"scene_id": sc["scene_id"], "reason": "invalid_scene_bounds"})
                continue
            self.scenes[sc["scene_index"]] = sc
        self.by_scene: Dict[int, List[Ev]] = defaultdict(list)
        for ev in self.valid:
            self.by_scene[ev.scene_idx].append(ev)
        self.dialogue_by_scene: Dict[int, List[Ev]] = {
            k: [x for x in v if x.cat == "DIALOGUE"] for k, v in self.by_scene.items()}
        scene_ends = [sc["end_ms"] for sc in self.scenes.values()]
        self.end_ms = max(scene_ends) if scene_ends else 0

    def out_of_scene(self, ev: Ev) -> bool:
        sc = self.scenes.get(ev.scene_idx)
        return bool(sc) and (ev.s < sc["start_ms"] or ev.e > sc["end_ms"] or "out_of_scene" in ev.flags)


# --------------------------------------------------------------------------- #
# Hard rules and features
# --------------------------------------------------------------------------- #
def window_members(ed: EpisodeData, s: int, e: int) -> List[Ev]:
    return [ev for ev in ed.valid if intersects(ev, s, e)]


def structural_failures(ed: EpisodeData, sc_idx: int, s: int, e: int, cfg: Config) -> List[str]:
    """The rules that need the window's content (evaluated only for in-range pairs)."""
    fails: List[str] = []
    members = window_members(ed, s, e)
    dialogues = [x for x in members if x.cat == "DIALOGUE"]
    if any(x.s < s < x.e or x.s < e < x.e for x in dialogues):
        fails.append("dialogue_straddles_boundary")
    scene_dlg = ed.dialogue_by_scene.get(sc_idx, [])
    inside_dlg = sorted((x for x in dialogues if x.s >= s and x.e <= e), key=lambda x: (x.s, x.e))
    if inside_dlg:
        first, last = inside_dlg[0], inside_dlg[-1]
        gap_ms = int(cfg.fragment_gap_s * 1000)
        i = next((k for k, x in enumerate(scene_dlg) if x.id == first.id), None)
        if i is not None and i > 0 and starts_with_ellipsis(first.clean):
            prev = scene_dlg[i - 1]
            if ends_with_ellipsis(prev.clean) and 0 <= first.s - prev.e <= gap_ms:
                fails.append("fragmented_sentence_start")
        j = next((k for k, x in enumerate(scene_dlg) if x.id == last.id), None)
        if j is not None and j + 1 < len(scene_dlg) and ends_with_ellipsis(last.clean):
            nxt = scene_dlg[j + 1]
            if starts_with_ellipsis(nxt.clean) and 0 <= nxt.s - last.e <= gap_ms:
                fails.append("fragmented_sentence_end")
    early_ms = int(cfg.early_event_s * 1000)
    if not any(x.cat in NON_CAMERA and s <= x.s <= s + early_ms for x in members):
        fails.append("no_early_event")
    if len(inside_dlg) < cfg.min_dialogue_events:
        fails.append("too_few_dialogue_events")
    live = [(x.s, x.e) for x in members if x.cat in NON_CAMERA]
    if longest_gap(live, s, e) > int(cfg.max_dead_air_s * 1000):
        fails.append("dead_air_too_long")
    return fails


def compute_features(ed: EpisodeData, sc_idx: int, s: int, e: int, cfg: Config,
                     low_conf: Dict[Tuple[str, int], bool]) -> Dict[str, Any]:
    members = window_members(ed, s, e)
    inside = [x for x in members if x.s >= s and x.e <= e and x.scene_idx == sc_idx]
    dur_ms = e - s
    dlg = sorted((x for x in inside if x.cat == "DIALOGUE"), key=lambda x: (x.s, x.e, x.order))
    non_cam = [x for x in members if x.cat in NON_CAMERA]
    speakers_raw = {x.speaker_raw for x in dlg if x.speaker_raw}
    speakers = [x.speaker for x in dlg if x.speaker]
    distinct = len(set(speakers))
    turns = sum(1 for a, b in zip(speakers, speakers[1:]) if a != b)
    dlg_text = " ".join(x.clean for x in dlg)
    all_text = " ".join([dlg_text] + [x.text for x in inside if x.cat in ("ACTION", "SOUND_FX")])
    conflict_terms = [m.group(0).lower() for m in CONFLICT_RE.finditer(all_text)]
    first_event = min((x.s for x in non_cam if x.s >= s), default=None)
    first_dlg = dlg[0].s if dlg else None
    last_beat = max((x for x in inside if x.cat in BEAT_END_CATEGORIES), key=lambda x: (x.e, x.order), default=None)
    last_dlg = dlg[-1] if dlg else None
    scene_dlg = ed.dialogue_by_scene.get(sc_idx, [])
    prior_gap = None
    if dlg:
        prior = [x for x in scene_dlg if x.e <= dlg[0].s and x.id != dlg[0].id]
        if prior:
            prior_gap = (dlg[0].s - max(prior, key=lambda x: x.e).e) / 1000.0
    cat_counts = Counter(x.cat for x in inside)
    sc = ed.scenes[sc_idx]
    ends_q = bool(last_dlg and re.search(r"\?\W*$", last_dlg.clean))
    ends_x = bool(last_dlg and re.search(r"!\W*$", last_dlg.clean))
    f = {
        "duration_ms": dur_ms,
        "event_count_inside": len(inside),
        "events_by_category": dict(cat_counts),
        "boundary_event_count": len(members) - len(inside),
        "dialogue_event_count": len(dlg),
        "dialogue_coverage_ratio": round(union_length([(x.s, x.e) for x in dlg], s, e) / dur_ms, 3),
        "dialogue_events_per_10s": round(len(dlg) / (dur_ms / 10000.0), 3),
        "distinct_speakers_raw": len(speakers_raw),
        "distinct_speakers_canonical": distinct,
        "speaker_turns": turns,
        "question_count": dlg_text.count("?"),
        "exclamation_count": dlg_text.count("!"),
        "action_event_count": cat_counts.get("ACTION", 0),
        "sound_fx_event_count": cat_counts.get("SOUND_FX", 0),
        "has_action_or_sfx": bool(cat_counts.get("ACTION") or cat_counts.get("SOUND_FX")),
        "conflict_term_count": len(conflict_terms),
        "conflict_terms": sorted(set(conflict_terms)),
        "first_nonCamera_event_offset_s": round((first_event - s) / 1000.0, 2) if first_event is not None else None,
        "first_dialogue_offset_s": round((first_dlg - s) / 1000.0, 2) if first_dlg is not None else None,
        "max_dead_air_s": round(longest_gap([(x.s, x.e) for x in non_cam], s, e) / 1000.0, 2),
        "ends_on_category": last_beat.cat if last_beat else None,
        "end_gap_after_last_beat_s": round((e - last_beat.e) / 1000.0, 2) if last_beat else None,
        "ends_with_question": ends_q,
        "ends_with_exclamation": ends_x,
        "ends_with_ellipsis": bool(last_dlg and ends_with_ellipsis(last_dlg.clean)),
        "opens_with_ellipsis": bool(dlg and starts_with_ellipsis(dlg[0].clean)),
        "gap_from_prior_dialogue_s": prior_gap,
        "opens_mid_exchange": prior_gap is not None and prior_gap <= 1.0,
        "starts_at_scene_start": s == sc["start_ms"],
        "ends_at_scene_end": e == sc["end_ms"],
        "scene_duration_s": round((sc["end_ms"] - sc["start_ms"]) / 1000.0, 3),
        "low_confidence_alias_hits": sorted({x.speaker_raw for x in dlg
                                            if x.speaker_raw and (x.speaker_raw, ed.number) in low_conf}),
    }
    return f


def prune_score(f: Dict[str, Any], cfg: Config) -> Tuple[float, Dict[str, float]]:
    """Heuristic 0-1 structural viability.  NOT a hook/cliffhanger/quality score."""
    d = f["distinct_speakers_canonical"]
    speaker_c = {0: 0.0, 1: 0.4, 2: 1.0, 3: 1.0, 4: 0.6}.get(d, 0.2)
    off = f["first_nonCamera_event_offset_s"]
    lead_c = 0.0 if off is None else (1.0 if off <= 1.0 else 0.7 if off <= 2.0 else 0.4 if off <= 3.0 else 0.0)
    gap = f["end_gap_after_last_beat_s"]
    end_c = 0.2 if gap is None else (1.0 if gap <= 1.0 else 0.6 if gap <= 2.0 else 0.2)
    if f["ends_on_category"] not in BEAT_END_CATEGORIES:
        end_c = 0.0
    dur_s = f["duration_ms"] / 1000.0
    band_c = 1.0 if 30.0 <= dur_s <= 45.0 else 0.6
    comp = {
        "dialogue_density": 0.20 * min(1.0, f["dialogue_events_per_10s"] / 3.0),
        "speaker_count": 0.15 * speaker_c,
        "speaker_turns": 0.15 * min(1.0, f["speaker_turns"] / 6.0),
        "questions_exclamations": 0.10 * min(1.0, (f["question_count"] + f["exclamation_count"]) / 6.0),
        "conflict_vocabulary": 0.10 * min(1.0, f["conflict_term_count"] / 4.0),
        "action_or_sfx_present": 0.05 * (1.0 if f["has_action_or_sfx"] else 0.0),
        "ends_on_event_boundary": 0.10 * end_c,
        "early_event": 0.10 * lead_c,
        "duration_band": 0.05 * band_c,
        "penalty_opens_mid_exchange": -0.05 if f["opens_mid_exchange"] else 0.0,
    }
    return round(max(0.0, sum(comp.values())), 4), {k: round(v, 4) for k, v in comp.items()}


# --------------------------------------------------------------------------- #
# Enumeration and selection
# --------------------------------------------------------------------------- #
def enumerate_pool(ed: EpisodeData, cfg: Config, low_conf) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """All windows that satisfy every hard rule, plus rejection counters."""
    min_ms, max_ms = int(round(cfg.min_s * 1000)), int(round(cfg.max_s * 1000))
    first_fail: Counter = Counter()
    any_fail: Counter = Counter()
    stats = {"pairs_considered": 0, "cross_scene_pairs": 0, "pool_size": 0}
    pool: List[Dict[str, Any]] = []
    idxs = sorted(ed.scenes)
    starts = {i: sorted({x.s for x in ed.by_scene.get(i, [])}) for i in idxs}
    ends = {i: sorted({x.e for x in ed.by_scene.get(i, [])}) for i in idxs}
    end_cats: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    for i in idxs:
        for x in ed.by_scene.get(i, []):
            end_cats[(i, x.e)].add(x.cat)

    # pairs whose start and end sit in different scenes cross a scene boundary
    for a_pos, a in enumerate(idxs):
        for b in idxs[a_pos + 1:]:
            n = sum(len(ends[b]) - bisect.bisect_right(ends[b], s) for s in starts[a])
            stats["cross_scene_pairs"] += n
            stats["pairs_considered"] += n
            first_fail["outside_scene_bounds"] += n
            any_fail["outside_scene_bounds"] += n

    for i in idxs:
        sc = ed.scenes[i]
        for s in starts[i]:
            for e in ends[i]:
                if e <= s:
                    continue
                stats["pairs_considered"] += 1
                dur = e - s
                fails: List[str] = []
                if s < sc["start_ms"] or e > sc["end_ms"]:
                    fails.append("outside_scene_bounds")
                if dur < min_ms:
                    fails.append("duration_below_min")
                if dur > max_ms:
                    fails.append("duration_above_max")
                if not (end_cats[(i, e)] & BEAT_END_CATEGORIES):
                    fails.append("end_on_camera_only")
                if not fails:
                    fails = structural_failures(ed, i, s, e, cfg)
                if fails:
                    first_fail[min(fails, key=HARD_RULES.index)] += 1
                    for r in fails:
                        any_fail[r] += 1
                    continue
                feats = compute_features(ed, i, s, e, cfg, low_conf)
                score, comps = prune_score(feats, cfg)
                pool.append({"scene_idx": i, "s": s, "e": e, "features": feats,
                             "prune_score": score, "prune_components": comps})
    stats["pool_size"] = len(pool)
    return pool, {"first_failure": dict(first_fail), "any_failure": dict(any_fail), **stats}


def select_diverse(pool: List[Dict[str, Any]], cfg: Config) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Greedy diverse pick: hard IoU limit + small same-scene nudge. Deterministic."""
    picked: List[Dict[str, Any]] = []
    remaining = [c for c in pool if c["prune_score"] >= cfg.min_prune_score]
    below_floor = len(pool) - len(remaining)
    rejected_overlap = 0
    while remaining and len(picked) < cfg.target_per_episode:
        def adj(c):
            same = sum(1 for p in picked if p["scene_idx"] == c["scene_idx"])
            return c["prune_score"] - cfg.same_scene_penalty * same
        best = max(remaining, key=lambda c: (adj(c), -c["s"], -c["e"]))
        picked.append(best)
        keep = []
        for c in remaining:
            if c is best:
                continue
            if iou((c["s"], c["e"]), (best["s"], best["e"])) > cfg.max_iou:
                rejected_overlap += 1
            else:
                keep.append(c)
        remaining = keep
    return picked, {"below_prune_floor": below_floor, "removed_by_overlap_limit": rejected_overlap}


# --------------------------------------------------------------------------- #
# Rendering and candidate assembly
# --------------------------------------------------------------------------- #
def label_of(ev: Ev) -> str:
    if ev.cat == "DIALOGUE":
        extra = [m for m in ev.mods] + ([ev.delivery] if ev.delivery else [])
        # Source label on purpose: alias merging is for counting only; showing a merged
        # name would leak identity knowledge that the clip itself does not establish.
        return f"{ev.speaker_raw}" + (f" ({'; '.join(extra)})" if extra else "")
    return {"CAMERA": "[CAMERA]", "ACTION": "[ACTION]", "SOUND_FX": "[SOUND FX]"}.get(ev.cat, f"[{ev.cat}]")


def render_lines(events: List[Tuple[Ev, str]], ref_ms: int, relative: bool, tag_in_progress: str) -> str:
    out = []
    for ev, rel in events:
        if ev.cat == "DIALOGUE":
            body = f"{label_of(ev)}: {ev.text}"
        else:
            body = f"{label_of(ev)} {ev.text}"
        if relative:
            off = max(0, ev.s - ref_ms) / 1000.0
            note = f" {tag_in_progress}" if rel in ("began_before", "both", "began_inside_clip") else ""
            out.append(f"+{off:05.2f}s {body}{note}")
        else:
            out.append(f"[{ev.sraw}-{ev.eraw}] {body}")
    return "\n".join(out)


def event_record(ev: Ev, relation: str) -> Dict[str, Any]:
    return {"event_id": ev.id, "file_order": ev.order, "category": ev.cat, "speaker_raw": ev.speaker_raw,
            "speaker": ev.speaker_raw, "speaker_canonical_for_counting": ev.speaker,
            "delivery": ev.delivery, "voice_modifiers": ev.mods,
            "start_raw": ev.sraw, "end_raw": ev.eraw, "relation_to_clip": relation,
            "scene_id": ev.scene_id, "text": ev.text}


def build_candidate(ed: EpisodeData, pick: Dict[str, Any], seq: int, cfg: Config,
                    low_conf) -> Dict[str, Any]:
    ep = ed.raw
    sc = ed.scenes[pick["scene_idx"]]
    s, e = pick["s"], pick["e"]
    members = window_members(ed, s, e)
    inside, boundary, began_before, continues_after, foreign = [], [], [], [], []
    for ev in sorted(members, key=lambda x: (x.s, x.e, x.order)):
        if ev.scene_idx != pick["scene_idx"]:
            foreign.append(ev)
            boundary.append((ev, "overlaps_from_other_scene"))
            continue
        b, a = ev.s < s, ev.e > e
        if b and a:
            began_before.append(ev); continues_after.append(ev); rel = "both"
        elif b:
            began_before.append(ev); rel = "began_before"
        elif a:
            continues_after.append(ev); rel = "continues_after"
        else:
            rel = "inside"
        if rel == "inside":
            inside.append((ev, rel))
        else:
            boundary.append((ev, rel))
    clip_events = [ev for ev, _ in inside]
    start_ev = next(ev for ev in ed.by_scene[pick["scene_idx"]] if ev.s == s)
    end_ev = next(ev for ev in ed.by_scene[pick["scene_idx"]] if ev.e == e and ev.cat in BEAT_END_CATEGORIES)

    # lookahead: strictly after the clip end, kept apart from the clip
    la_limit = e + int(round(cfg.lookahead_s * 1000))
    la_events = []
    for ev in sorted(ed.valid, key=lambda x: (x.s, x.e, x.order)):
        # Lookahead is strictly post-clip: events must START at/after clip end.
        # A boundary-overlapping event is recorded only in boundary_events.
        if ev.s >= e and ev.s < la_limit:
            la_events.append((ev, "after_clip"))
    available_ms = max(0, min(la_limit, ed.end_ms) - e)
    la_scenes = {ev.scene_idx for ev, _ in la_events}
    crosses = bool(la_scenes - {pick["scene_idx"]}) or (e + available_ms > sc["end_ms"])

    warnings: List[str] = []
    for ev in clip_events + foreign:
        if ed.out_of_scene(ev):
            warnings.append(f"source_event_outside_scene_boundary:{ev.id}")
    for ev in foreign:
        warnings.append(f"foreign_scene_event_overlaps_window:{ev.id}")
    for ev in clip_events:
        for fl in ("same_track_overlap", "out_of_order"):
            if fl in ev.flags:
                warnings.append(f"source_event_flag_{fl}:{ev.id}")
    for a in pick["features"]["low_confidence_alias_hits"]:
        warnings.append(f"low_confidence_alias:{a}")
    if pick["features"]["opens_mid_exchange"]:
        warnings.append("opens_mid_exchange")
    if pick["features"]["opens_with_ellipsis"]:
        warnings.append("opens_with_ellipsis")
    if available_ms == 0:
        warnings.append("no_lookahead_available")
    elif available_ms < int(round(cfg.lookahead_s * 1000)):
        warnings.append(f"lookahead_shorter_than_requested:{available_ms / 1000.0:.2f}s")
    if crosses:
        warnings.append("lookahead_crosses_scene_boundary")
    warnings = sorted(set(warnings))

    cand = {
        "candidate_id": f"C{ed.number:03d}-{seq:02d}",
        "episode_number": ed.number,
        "source_file": ep["filename"],
        "asset_id": ep.get("asset_id"),
        "scene_id": sc["scene_id"],
        "scene_index": pick["scene_idx"],
        "scene_heading": sc.get("heading_raw"),
        "start_raw": start_ev.sraw,
        "end_raw": end_ev.eraw,
        "start_ms": s,
        "end_ms": e,
        "duration_ms": e - s,
        "duration_s": round((e - s) / 1000.0, 3),
        "start_event_id": start_ev.id,
        "end_event_id": end_ev.id,
        "event_ids": [ev.id for ev, rel in inside if rel == "inside"],
        "boundary_event_ids": {
            "began_before_clip": [ev.id for ev in began_before],
            "continues_after_clip": [ev.id for ev in continues_after],
            "other_scene_overlap": [ev.id for ev in foreign]},
        "events": [event_record(ev, "inside") for ev, _ in inside],
        "boundary_events": [event_record(ev, rel) for ev, rel in boundary],
        "candidate_text": render_lines(inside, s, False, ""),
        "candidate_text_relative": render_lines(inside, s, True, "[in progress at clip start]"),
        "lookahead": {
            "note": "Screenplay content AFTER the clip end; excluded from duration_s.",
            "starts_at_clip_end_raw": end_ev.eraw,
            "requested_s": cfg.lookahead_s,
            "available_s": round(available_ms / 1000.0, 3),
            "crosses_scene_boundary": crosses,
            "event_ids": [ev.id for ev, _ in la_events],
            "events": [event_record(ev, rel) for ev, rel in la_events],
            "text": render_lines(la_events, e, False, ""),
            "text_relative": render_lines(la_events, e, True, "[in progress at clip end]"),
        },
        "structural_features": pick["features"],
        "prune_score": pick["prune_score"],
        "prune_score_components": pick["prune_components"],
        "prune_score_note": "Heuristic structural viability for pruning only; not a quality score or rank.",
        "warnings": warnings,
    }
    return cand


# --------------------------------------------------------------------------- #
# Validation (independent of the generation path)
# --------------------------------------------------------------------------- #
def load_csv_index(path: Optional[Path]) -> Optional[Dict[str, Dict[str, str]]]:
    if path is None or not path.is_file():
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["event_id"]: r for r in csv.DictReader(f)}


def validate_candidates(cands: List[Dict[str, Any]], episodes: List[Dict[str, Any]],
                        csv_idx: Optional[Dict[str, Dict[str, str]]], cfg: Config) -> Dict[str, Any]:
    ep_by_num = {e["episode_number"]: e for e in episodes}
    checks = Counter()
    failures: List[Dict[str, str]] = []
    min_ms, max_ms = int(round(cfg.min_s * 1000)), int(round(cfg.max_s * 1000))

    def check(name: str, ok: bool, cid: str, detail: str = "") -> None:
        checks[name + (":pass" if ok else ":FAIL")] += 1
        if not ok:
            failures.append({"candidate_id": cid, "check": name, "detail": detail})

    seen_ids: Set[str] = set()
    by_ep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in cands:
        cid = c["candidate_id"]
        by_ep[c["episode_number"]].append(c)
        check("unique_candidate_id", cid not in seen_ids, cid)
        seen_ids.add(cid)
        ep = ep_by_num.get(c["episode_number"])
        check("episode_exists", ep is not None, cid)
        if ep is None:
            continue
        ev_src = {x["event_id"]: x for x in ep["events"]}
        scene = next((sc for sc in ep["scenes"] if sc["scene_id"] == c["scene_id"]), None)
        check("scene_exists", scene is not None, cid)
        s, e = c["start_ms"], c["end_ms"]
        check("duration_20_60_inclusive", min_ms <= e - s <= max_ms, cid, f"{e - s} ms")
        check("duration_equals_end_minus_start", c["duration_ms"] == e - s and
              abs(c["duration_s"] - (e - s) / 1000.0) < 1e-9, cid)
        check("start_raw_reparses_to_start_ms", mini_parse_ts(c["start_raw"]) == s, cid, c["start_raw"])
        check("end_raw_reparses_to_end_ms", mini_parse_ts(c["end_raw"]) == e, cid, c["end_raw"])
        if scene is not None:
            check("inside_single_scene", scene["start_ms"] <= s and e <= scene["end_ms"], cid,
                  f"{c['start_raw']}-{c['end_raw']} vs scene {scene['start_raw']}-{scene['end_raw']}")
            scene_events = [x for x in ep["events"] if x["scene_id"] == c["scene_id"]]
            check("start_is_source_event_start",
                  any(x["start_ms"] == s and x["start_raw"] == c["start_raw"] for x in scene_events), cid)
            check("end_is_source_event_end",
                  any(x["end_ms"] == e and x["end_raw"] == c["end_raw"] for x in scene_events), cid)
        ep_end = max((sc["end_ms"] for sc in ep["scenes"] if sc["end_ms"] is not None), default=0)
        check("inside_episode", 0 <= s < e <= ep_end, cid)
        all_ids = c["event_ids"] + sum(c["boundary_event_ids"].values(), [])
        check("event_ids_exist_in_parsed_json", all(i in ev_src for i in all_ids), cid)
        ok_txt = all(ev_src[x["event_id"]]["text"] == x["text"] and
                     ev_src[x["event_id"]]["start_raw"] == x["start_raw"] and
                     ev_src[x["event_id"]]["end_raw"] == x["end_raw"]
                     for x in c["events"] + c["lookahead"]["events"] if x["event_id"] in ev_src)
        check("event_text_and_timestamps_match_source", ok_txt, cid)
        if csv_idx is not None:
            ok_csv = all(x["event_id"] in csv_idx and csv_idx[x["event_id"]]["text"] == x["text"]
                         and csv_idx[x["event_id"]]["start_raw"] == x["start_raw"]
                         and csv_idx[x["event_id"]]["end_raw"] == x["end_raw"]
                         for x in c["events"] + c["lookahead"]["events"])
            check("event_ids_match_parsed_events_csv", ok_csv, cid)
        expected_inside = sorted(
            [x["event_id"] for x in ep["events"]
             if x["scene_id"] == c["scene_id"] and x["start_ms"] >= s and x["end_ms"] <= e],
            key=lambda event_id: next((x["start_ms"], x["end_ms"], x.get("file_order", 0))
                                      for x in ep["events"] if x["event_id"] == event_id))
        actual_inside = [x["event_id"] for x in c["events"]]
        check("events_list_contains_only_fully_contained_events",
              actual_inside == expected_inside, cid,
              f"expected {len(expected_inside)} inside events, got {len(actual_inside)}")
        check("event_ids_match_fully_contained_events",
              sorted(c["event_ids"]) == sorted(expected_inside), cid)
        check("boundary_events_separate_from_inside",
              not (set(actual_inside) & {x["event_id"] for x in c.get("boundary_events", [])}), cid)
        boundary_ids = [x["event_id"] for x in c.get("boundary_events", [])]
        check("boundary_event_ids_match_boundary_records",
              set(boundary_ids) == set(sum(c["boundary_event_ids"].values(), [])), cid)
        check("boundary_events_match_source",
              all(x["event_id"] in ev_src and
                  ev_src[x["event_id"]]["start_raw"] == x["start_raw"] and
                  ev_src[x["event_id"]]["end_raw"] == x["end_raw"] and
                  ev_src[x["event_id"]]["text"] == x["text"] and
                  ev_src[x["event_id"]]["category"] == x["category"]
                  for x in c.get("boundary_events", [])), cid)
        # Candidate text must be rendered exclusively from the fully-contained event records.
        expected_text_lines = []
        for x in c["events"]:
            if x["category"] == "DIALOGUE":
                extra = [m for m in x.get("voice_modifiers", [])] + ([x["delivery"]] if x.get("delivery") else [])
                label = x.get("speaker_raw") or ""
                if extra: label += " (" + "; ".join(extra) + ")"
                body = f"{label}: {x['text']}"
            else:
                body = {"CAMERA": "[CAMERA]", "ACTION": "[ACTION]", "SOUND_FX": "[SOUND FX]"}.get(
                    x["category"], f"[{x['category']}]") + " " + x["text"]
            expected_text_lines.append(f"[{x['start_raw']}-{x['end_raw']}] {body}")
        check("candidate_text_contains_only_inside_events",
              c["candidate_text"] == "\n".join(expected_text_lines), cid)
        la = c["lookahead"]
        la_ev = [ev_src[i] for i in la["event_ids"] if i in ev_src]
        check("lookahead_events_start_at_or_after_clip_end",
              all(x["start_ms"] >= e for x in la_ev), cid)
        check("lookahead_within_15_second_start_window",
              all(x["start_ms"] < e + int(round(cfg.lookahead_s * 1000)) for x in la_ev), cid)
        check("lookahead_not_duplicated_from_boundary",
              not (set(la["event_ids"]) & set(boundary_ids)), cid)
        check("lookahead_not_duplicated_from_clip",
              not (set(la["event_ids"]) & set(c["event_ids"])), cid)
        check("lookahead_not_in_duration",
              c["duration_ms"] == e - s and la["available_s"] <= cfg.lookahead_s + 1e-9, cid)
        check("lookahead_text_events_match_ids",
              set(la["event_ids"]) == {x["event_id"] for x in la["events"]}, cid)
        check("lookahead_event_source_fields_match",
              all(x["event_id"] in ev_src and
                  ev_src[x["event_id"]]["start_raw"] == x["start_raw"] and
                  ev_src[x["event_id"]]["end_raw"] == x["end_raw"] and
                  ev_src[x["event_id"]]["text"] == x["text"]
                  for x in la["events"]), cid)
    for num, lst in by_ep.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                v = iou((lst[i]["start_ms"], lst[i]["end_ms"]), (lst[j]["start_ms"], lst[j]["end_ms"]))
                check("pairwise_iou_within_limit", v <= cfg.max_iou + 1e-9,
                      f"{lst[i]['candidate_id']}/{lst[j]['candidate_id']}", f"iou={v:.3f}")
    n_fail = sum(v for k, v in checks.items() if k.endswith(":FAIL"))
    return {"status": "PASS" if n_fail == 0 else "FAIL", "checks_failed": n_fail,
            "check_counts": dict(sorted(checks.items())), "failures": failures[:50]}


# --------------------------------------------------------------------------- #
# Summary and outputs
# --------------------------------------------------------------------------- #
def describe(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    return {"count": len(s), "min": round(s[0], 3), "max": round(s[-1], 3),
            "mean": round(statistics.fmean(s), 3), "median": round(statistics.median(s), 3)}


def leader_clusters(pool: List[Dict[str, Any]], thr: float) -> int:
    leaders: List[Tuple[int, int]] = []
    for c in sorted(pool, key=lambda c: (c["s"], c["e"])):
        if not any(iou((c["s"], c["e"]), l) > thr for l in leaders):
            leaders.append((c["s"], c["e"]))
    return len(leaders)


CSV_COLUMNS = [
    "candidate_id", "episode_number", "source_file", "scene_id", "start_raw", "end_raw", "duration_s",
    "start_event_id", "end_event_id", "event_count_inside", "dialogue_event_count",
    "distinct_speakers_canonical", "distinct_speakers_raw", "speaker_turns", "dialogue_events_per_10s",
    "dialogue_coverage_ratio", "question_count", "exclamation_count", "action_event_count",
    "sound_fx_event_count", "conflict_term_count", "first_nonCamera_event_offset_s",
    "first_dialogue_offset_s", "max_dead_air_s", "ends_on_category", "end_gap_after_last_beat_s",
    "ends_with_question", "ends_with_exclamation", "ends_with_ellipsis", "opens_mid_exchange",
    "starts_at_scene_start", "ends_at_scene_end", "prune_score", "lookahead_available_s",
    "lookahead_crosses_scene", "lookahead_event_count", "boundary_event_count", "lookahead_status", "warnings"]


def write_outputs(out_dir: Path, cands: List[Dict[str, Any]], summary: Dict[str, Any], cfg: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "promo_candidates.json", "w", encoding="utf-8") as f:
        json.dump({"phase": 2, "generator_version": VERSION, "config": asdict(cfg),
                   "note": "Ordered by episode then start time. Not ranked.",
                   "candidates": cands}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "promo_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for c in cands:
            row = {**c["structural_features"], **{k: c[k] for k in (
                "candidate_id", "episode_number", "source_file", "scene_id", "start_raw", "end_raw",
                "duration_s", "start_event_id", "end_event_id", "prune_score")}}
            row["lookahead_available_s"] = c["lookahead"]["available_s"]
            row["lookahead_crosses_scene"] = c["lookahead"]["crosses_scene_boundary"]
            row["lookahead_event_count"] = len(c["lookahead"]["event_ids"])
            row["boundary_event_count"] = len(c.get("boundary_events", []))
            avail = c["lookahead"]["available_s"]
            row["lookahead_status"] = "full" if avail >= cfg.lookahead_s else ("partial" if avail > 0 else "none")
            row["warnings"] = ";".join(c["warnings"])
            w.writerow({k: row.get(k) for k in CSV_COLUMNS})
    with open(out_dir / "candidate_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    validation_report = {
        "status": summary["validation"]["status"],
        "candidate_count": summary["total_candidates"],
        "generator_validation": summary["validation"],
        "lookahead_statistics": summary["lookahead_statistics"],
        "notes": "This file is written automatically by generate_candidates.py. "
                 "It contains the generator's independent candidate checks; the delivered "
                 "validation file may also include the separate raw-screenplay integrity audit."
    }
    with open(out_dir / "candidate_generation_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)


def print_summary(summary: Dict[str, Any], out_dir: Path) -> None:
    v = summary["validation"]
    print("=" * 68)
    print(f"Candidate generation v{VERSION}")
    print("=" * 68)
    print(f"Episodes processed       : {summary['episodes_processed']}")
    print(f"Candidates generated     : {summary['total_candidates']}  "
          f"(per episode: {summary['candidates_per_episode_distribution']})")
    print(f"Episodes with zero       : {summary['episodes_with_zero_candidates'] or 'none'}")
    d = summary["duration_distribution_s"]
    print(f"Duration (s)             : min {d.get('min')}  median {d.get('median')}  max {d.get('max')}")
    print(f"Valid pool before pruning: {summary['pool_size_total']} windows "
          f"(~{summary['pool_distinct_clusters_total']} distinct at IoU>{summary['config']['cluster_iou']})")
    print(f"Pairs considered         : {summary['pairs_considered_total']}")
    print("Rejected (first failing hard rule):")
    for k, n in summary["rejected_by_rule_first_failure"].items():
        print(f"    {k:<28}{n}")
    print(f"Max IoU among selected   : {summary['selected_overlap_statistics']['max_iou']}")
    print(f"Lookahead                : full {summary['lookahead_statistics']['full']}, "
          f"partial {summary['lookahead_statistics']['partial']}, "
          f"none {summary['lookahead_statistics']['none']} (total {summary['lookahead_statistics']['total']})")
    print(f"Warnings                 : {summary['warnings_by_type'] or 'none'}")
    print(f"Excluded source events   : {summary['excluded_source_events_total']}")
    print(f"VALIDATION               : {v['status']} ({v['checks_failed']} failed of "
          f"{sum(v['check_counts'].values())} checks)")
    print(f"Outputs written to       : {out_dir.resolve()}")
    print("=" * 68)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2: deterministic promo candidate generation.")
    ap.add_argument("--parsed-json", type=Path, default=Path("parser_output/parsed_episodes.json"))
    ap.add_argument("--events-csv", type=Path, default=Path("parser_output/parsed_events.csv"),
                    help="optional second source used to cross-check event IDs and text")
    ap.add_argument("--aliases", type=Path, default=None, help="optional alias table (see aliases.json)")
    ap.add_argument("--out-dir", type=Path, default=Path("candidate_output"))
    d = Config()
    for name, typ in (("min_s", float), ("max_s", float), ("lookahead_s", float), ("target_per_episode", int),
                      ("max_iou", float), ("early_event_s", float), ("min_dialogue_events", int),
                      ("max_dead_air_s", float), ("min_prune_score", float)):
        ap.add_argument("--" + name.replace("_", "-"), type=typ, default=getattr(d, name))
    args = ap.parse_args(argv)
    cfg = Config(**{k: getattr(args, k) for k in asdict(d) if hasattr(args, k)})

    if not args.parsed_json.is_file():
        print(f"error: {args.parsed_json} not found (run Phase 1 first)", file=sys.stderr)
        return 2
    episodes = json.loads(args.parsed_json.read_text(encoding="utf-8"))["episodes"]
    canon, low_conf = load_aliases(args.aliases)

    all_cands: List[Dict[str, Any]] = []
    per_episode: Dict[int, Dict[str, Any]] = {}
    first_total: Counter = Counter()
    any_total: Counter = Counter()
    pairs_total = pool_total = clusters_total = cross_total = 0
    excluded_events: List[Dict[str, Any]] = []
    excluded_scenes: List[Dict[str, Any]] = []
    pool_removed = Counter()

    for ep in episodes:
        ed = EpisodeData(ep, canon)
        excluded_events += [{"episode_number": ed.number, **x} for x in ed.excluded]
        excluded_scenes += [{"episode_number": ed.number, **x} for x in ed.excluded_scenes]
        pool, st = enumerate_pool(ed, cfg, low_conf)
        picks, sel_stats = select_diverse(pool, cfg)
        picks.sort(key=lambda c: (c["s"], c["e"]))
        cands = [build_candidate(ed, p, i + 1, cfg, low_conf) for i, p in enumerate(picks)]
        all_cands += cands
        first_total.update(st["first_failure"])
        any_total.update(st["any_failure"])
        pairs_total += st["pairs_considered"]
        cross_total += st["cross_scene_pairs"]
        pool_total += st["pool_size"]
        clusters = leader_clusters(pool, cfg.cluster_iou)
        clusters_total += clusters
        pool_removed.update(sel_stats)
        per_episode[ed.number] = {
            "candidates": len(cands), "pool_size": st["pool_size"], "pool_distinct_clusters": clusters,
            "pairs_considered": st["pairs_considered"], "rejected_first_failure": st["first_failure"],
            **sel_stats, "excluded_events": len(ed.excluded), "scenes": len(ed.scenes)}

    all_cands.sort(key=lambda c: (c["episode_number"], c["start_ms"], c["end_ms"]))
    csv_idx = load_csv_index(args.events_csv)
    validation = validate_candidates(all_cands, episodes, csv_idx, cfg)
    validation["parsed_events_csv_cross_check"] = "used" if csv_idx is not None else "not available"

    # overlap statistics among selected candidates
    by_ep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in all_cands:
        by_ep[c["episode_number"]].append(c)
    ious, identical_sets, shared_event_pairs = [], 0, 0
    for lst in by_ep.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                ious.append(iou((lst[i]["start_ms"], lst[i]["end_ms"]), (lst[j]["start_ms"], lst[j]["end_ms"])))
                a, b = set(lst[i]["event_ids"]), set(lst[j]["event_ids"])
                identical_sets += a == b
                shared_event_pairs += bool(a & b)
    n_per = Counter(len(by_ep.get(ep["episode_number"], [])) for ep in episodes)
    la_full = sum(1 for c in all_cands if c["lookahead"]["available_s"] >= cfg.lookahead_s)
    warn_counts = Counter(w.split(":")[0] for c in all_cands for w in c["warnings"])
    dur_buckets = Counter()
    for c in all_cands:
        b = int(c["duration_s"] // 10) * 10
        dur_buckets[f"{b}-{b + 10}s" if c["duration_s"] < 60 else "60s"] += 1

    summary = {
        "generator_version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "input": {"parsed_json": str(args.parsed_json), "events_csv": str(args.events_csv) if csv_idx else None,
                  "aliases": str(args.aliases) if args.aliases else None},
        "episodes_processed": len(episodes),
        "total_candidates": len(all_cands),
        "candidates_per_episode": {str(k): v["candidates"] for k, v in sorted(per_episode.items())},
        "candidates_per_episode_distribution": {str(k): v for k, v in sorted(n_per.items())},
        "episodes_with_zero_candidates": [k for k, v in sorted(per_episode.items()) if v["candidates"] == 0],
        "episodes_below_target": [k for k, v in sorted(per_episode.items())
                                  if v["candidates"] < cfg.target_per_episode],
        "duration_distribution_s": describe([c["duration_s"] for c in all_cands]),
        "duration_buckets": dict(sorted(dur_buckets.items())),
        "pairs_considered_total": pairs_total,
        "cross_scene_pairs_total": cross_total,
        "rejected_by_rule_first_failure": {r: first_total.get(r, 0) for r in HARD_RULES},
        "rejected_by_rule_any_failure": {r: any_total.get(r, 0) for r in HARD_RULES},
        "rejection_counting_note": ("first_failure counts each rejected window once, by the first rule in "
                                    "evaluation order; any_failure counts every rule a window breaks. "
                                    "Content rules (straddle onwards) are only evaluated for windows that "
                                    "already satisfy scene and duration rules."),
        "pool_size_total": pool_total,
        "pool_distinct_clusters_total": clusters_total,
        "pool_pruned_after_hard_rules": dict(pool_removed),
        "duplicate_overlap_statistics_pool": {
            "valid_windows_before_diversification": pool_total,
            "approx_distinct_windows_at_iou": cfg.cluster_iou,
            "approx_distinct_count": clusters_total,
            "collapse_factor": round(pool_total / clusters_total, 2) if clusters_total else None},
        "selected_overlap_statistics": {
            "pairs_compared": len(ious),
            "max_iou": round(max(ious), 3) if ious else 0.0,
            "mean_iou": round(statistics.fmean(ious), 3) if ious else 0.0,
            "pairs_above_iou_limit": sum(1 for v in ious if v > cfg.max_iou + 1e-9),
            "pairs_with_identical_event_sets": identical_sets,
            "pairs_sharing_any_inside_event": shared_event_pairs},
        "lookahead_statistics": {
            "full": sum(1 for c in all_cands if c["lookahead"]["available_s"] >= cfg.lookahead_s),
            "partial": sum(1 for c in all_cands if 0 < c["lookahead"]["available_s"] < cfg.lookahead_s),
            "none": sum(1 for c in all_cands if c["lookahead"]["available_s"] == 0),
            "total": len(all_cands),
            "categories_sum_to_total": (
                sum(1 for c in all_cands if c["lookahead"]["available_s"] >= cfg.lookahead_s) +
                sum(1 for c in all_cands if 0 < c["lookahead"]["available_s"] < cfg.lookahead_s) +
                sum(1 for c in all_cands if c["lookahead"]["available_s"] == 0)
            ) == len(all_cands),
            "crosses_scene_boundary": sum(1 for c in all_cands if c["lookahead"]["crosses_scene_boundary"])},
        "warnings_by_type": dict(warn_counts.most_common()),
        "candidates_with_warnings": [{"candidate_id": c["candidate_id"], "warnings": c["warnings"]}
                                     for c in all_cands if c["warnings"]],
        "excluded_source_events_total": len(excluded_events),
        "excluded_source_events": excluded_events,
        "excluded_scenes": excluded_scenes,
        "per_episode": {str(k): v for k, v in sorted(per_episode.items())},
        "validation": validation,
    }
    write_outputs(args.out_dir, all_cands, summary, cfg)
    print_summary(summary, args.out_dir)
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
