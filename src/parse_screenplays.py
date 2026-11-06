#!/usr/bin/env python3
"""
Kuku FM microdrama promo MVP - Phase 1: deterministic screenplay parser.

Reads a ZIP of timestamped screenplay .txt files (never extracts to disk) and
writes four reports:

    parsed_episodes.json   full structured data, one record per episode
    parsed_events.csv      one row per event block
    parser_summary.json    counts, duration statistics, validation results
    parser_exceptions.csv  every anomaly and every line that could not be parsed

Usage
-----
    python3 parse_screenplays.py SCRIPTS.zip
    python3 parse_screenplays.py SCRIPTS.zip --out-dir out --expected-episodes 50
    python3 parse_screenplays.py SCRIPTS.zip --tolerance-ms 50

Design rules
------------
* Standard library only (Python 3.8+).
* Timestamps are preserved as the exact source token (``*_raw``).  Milliseconds
  (``*_ms``) are derived from that token only; nothing is invented or
  interpolated.  A missing or malformed token stays ``None``.
* Duration is always ``end - start`` of one event.  Events on different tracks
  (CAMERA / ACTION / DIALOGUE / SOUND_FX) run in parallel by design, so
  durations are never summed.  Overlap is only reported *within* one track.
* Nothing is dropped silently.  Any non-blank line that is not recognised is
  written to the exceptions report with its line number and raw content.
* No constants about the data (episode count, blocks per file, scene count)
  are assumed.  ``--expected-episodes`` is an optional check supplied by the
  caller.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

PARSER_VERSION = "1.0.0"
MAX_MEMBER_BYTES = 20 * 1024 * 1024  # refuse absurdly large ZIP members

# --------------------------------------------------------------------------- #
# Regular expressions (all case-insensitive where the format allows it)
# --------------------------------------------------------------------------- #
RE_FILENAME = re.compile(r"^episode[\s_-]*(\d+)(?:[\s_-]+episode[\s_-]*(\d+))?", re.I)
RE_HEADER_FIELD = re.compile(
    r"^\s*(EPISODE\s+NUMBER|EPISODE\s+LOGLINE|CONTINUITY\s+NOTES)\s*:\s*(.*?)\s*$", re.I)
RE_SCRIPT_START = re.compile(r"^\s*-{2,}\s*SCRIPT\s+START\s*-{2,}\s*$", re.I)
RE_SCRIPT_END = re.compile(r"^\s*-{2,}\s*SCRIPT\s+END\s*-{2,}\s*$", re.I)
RE_SCENE_FIELD = re.compile(r"^\s*SCENE\s+(HEADING|START|END)\s*:\s*(.*?)\s*$", re.I)
RE_SCENE_MARK = re.compile(r"^\s*SCENE\s+([^\s:]+)\s*$", re.I)
RE_ACTIONS_LABEL = re.compile(r"^\s*actions?\s+and\s+dialogue\s*:?\s*$", re.I)
RE_BLOCK_HEADER = re.compile(r"^\s*\*\*\s*(.+?)\s*\*\*\s*:?\s*$")
RE_TS_LINE_START = re.compile(r"^\s*(?:start|end)\s*:", re.I)
RE_TS_PAIR = re.compile(
    r"\b(start|end)\s*:\s*\(?\s*([^|()]*?)\s*\)?\s*(?=\||\b(?:start|end)\s*:|$)", re.I)
# MM:SS(.fff)  or  HH:MM:SS(.fff)
RE_TS_TOKEN = re.compile(r"^(\d{1,3}):(\d{2})(?::(\d{2}))?(?:[.,](\d+))?$")
RE_PAREN = re.compile(r"\([^()]*\)")
RE_INT_EXT = re.compile(r"^\s*((?:INT|EXT|I)\s*[./]\s*(?:EXT|INT|E)?\.?|INT\.?|EXT\.?)\s+", re.I)

NON_DIALOGUE_LABELS = {
    "CAMERA": "CAMERA",
    "ACTION": "ACTION",
    "SOUND FX": "SOUND_FX",
    "SOUNDFX": "SOUND_FX",
    "SOUND EFFECTS": "SOUND_FX",
    "SOUND EFFECT": "SOUND_FX",
    "SFX": "SOUND_FX",
    "SOUND": "SOUND_FX",
}
VOICE_MODIFIERS = {"O.S", "V.O", "O.C", "CONT'D", "CONT’D", "CONT"}
TIME_OF_DAY_WORDS = {"DAY", "NIGHT", "DUSK", "DAWN", "MORNING", "EVENING", "AFTERNOON",
                     "CONTINUOUS", "LATER", "SUNSET", "SUNRISE"}
TRACKS = ("CAMERA", "ACTION", "DIALOGUE", "SOUND_FX")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    event_id: str = ""
    episode_number: Optional[int] = None
    file_order: int = 0
    sorted_index: Optional[int] = None
    scene_id: Optional[str] = None
    scene_index: Optional[int] = None
    category: str = ""
    label_raw: str = ""
    speaker_raw: Optional[str] = None
    speaker: Optional[str] = None
    delivery: Optional[str] = None
    voice_modifiers: List[str] = field(default_factory=list)
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    duration_s: Optional[float] = None
    text: str = ""
    text_clean: str = ""
    line_start: int = 0
    line_end: int = 0
    flags: List[str] = field(default_factory=list)


@dataclass
class Scene:
    scene_id: Optional[str] = None
    scene_index: int = 0
    line_no: int = 0
    heading_raw: Optional[str] = None
    int_ext: Optional[str] = None
    time_of_day: Optional[str] = None
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    duration_s: Optional[float] = None
    event_ids: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    _event_orders: List[int] = field(default_factory=list, repr=False)


@dataclass
class Episode:
    episode_number: Optional[int] = None
    episode_number_source: Optional[str] = None
    header_episode_number: Optional[int] = None
    filename: str = ""
    zip_member: str = ""
    asset_id: Optional[str] = None
    logline: Optional[str] = None
    continuity_notes: Optional[str] = None
    line_count: int = 0
    script_start_line: Optional[int] = None
    script_end_line: Optional[int] = None
    scenes: List[Scene] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)          # original file order
    events_chronological_ids: List[str] = field(default_factory=list)
    events_unsortable_ids: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def norm_line(line: str) -> str:
    """Normalise whitespace oddities used only for *structure detection*."""
    return line.replace(" ", " ").replace("﻿", "").replace("\t", " ")


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_timestamp(raw: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return (milliseconds, issue).  The raw token is never altered."""
    if raw is None or raw.strip() == "":
        return None, "missing"
    m = RE_TS_TOKEN.match(raw.strip())
    if not m:
        return None, "malformed"
    a, b, c, frac = m.groups()
    if c is not None:                     # HH:MM:SS
        hours, minutes, seconds = int(a), int(b), int(c)
        if minutes >= 60:
            return None, "malformed"
    else:                                 # MM:SS
        hours, minutes, seconds = 0, int(a), int(b)
    if seconds >= 60:
        return None, "malformed"
    ms = ((hours * 60 + minutes) * 60 + seconds) * 1000
    issue = None
    if frac:
        if len(frac) > 3:
            issue = "precision_truncated"
        ms += int(frac.ljust(3, "0")[:3])
    return ms, issue


def strip_parentheticals(label: str) -> Tuple[str, List[str], bool]:
    """Remove (...) groups from a label. Returns (base, groups, unbalanced)."""
    groups: List[str] = []
    base = label
    while True:
        found = RE_PAREN.findall(base)
        if not found:
            break
        groups.extend(g[1:-1].strip() for g in found)
        base = RE_PAREN.sub(" ", base)
    unbalanced = "(" in base or ")" in base
    return collapse_ws(base).rstrip(":").strip(), groups, unbalanced


def classify_label(label_raw: str) -> Dict[str, Any]:
    """Category, speaker, delivery and voice modifiers from a block label."""
    base, groups, unbalanced = strip_parentheticals(label_raw)
    base_upper = base.upper()
    modifiers: List[str] = []
    delivery_parts: List[str] = []
    for g in groups:
        rest: List[str] = []
        for part in g.split("/"):
            token = part.strip()
            if token.upper().rstrip(".,") in VOICE_MODIFIERS or token.upper().rstrip(".") in VOICE_MODIFIERS:
                modifiers.append(token.upper())
            elif token:
                rest.append(token)
        if rest:
            delivery_parts.append(", ".join(rest) if len(rest) > 1 else rest[0])
    category = NON_DIALOGUE_LABELS.get(base_upper)
    out = {
        "category": category or "DIALOGUE",
        "speaker_raw": None if category else base,
        "speaker": None if category else base_upper,
        "delivery": "; ".join(delivery_parts) if delivery_parts else None,
        "voice_modifiers": modifiers,
        "unbalanced": unbalanced,
        "empty_base": base == "",
    }
    return out


def parse_heading(heading: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort INT/EXT and time-of-day from a scene heading (never invented)."""
    if not heading:
        return None, None
    int_ext = None
    m = RE_INT_EXT.match(heading)
    if m:
        int_ext = re.sub(r"\s+", "", m.group(1).upper())
    time_of_day = None
    parts = re.split(r"\s+[–—-]\s+", heading.strip())
    if len(parts) > 1:
        last = parts[-1].strip().upper()
        if last in TIME_OF_DAY_WORDS:
            time_of_day = last
    return int_ext, time_of_day


def describe(values: List[float]) -> Dict[str, Any]:
    """Count/min/max/mean/median/p90 for a list of numbers (no summing of tracks)."""
    if not values:
        return {"count": 0}
    s = sorted(values)
    p90 = s[min(len(s) - 1, max(0, int(round(0.9 * len(s) + 0.5)) - 1))]
    return {
        "count": len(s),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "mean": round(statistics.fmean(s), 3),
        "median": round(statistics.median(s), 3),
        "p90": round(p90, 3),
    }


# --------------------------------------------------------------------------- #
# Exception logging
# --------------------------------------------------------------------------- #
def add_exc(ep: Episode, severity: str, etype: str, detail: str, *,
            line_no: Optional[int] = None, raw: Optional[str] = None,
            scene_id: Optional[str] = None, event_order: Optional[int] = None,
            ts_raw: Optional[str] = None) -> None:
    ep.exceptions.append({
        "severity": severity, "type": etype, "scene_id": scene_id,
        "event_order": event_order, "line_no": line_no, "timestamp_raw": ts_raw,
        "detail": detail, "raw_content": raw,
    })


# --------------------------------------------------------------------------- #
# Single-file parsing
# --------------------------------------------------------------------------- #
class _Builder:
    """Accumulates one event block while reading lines."""
    def __init__(self, label_raw: str, line_no: int, scene: Optional[Scene]):
        self.label_raw = label_raw
        self.line_start = line_no
        self.line_end = line_no
        self.scene = scene
        self.ts_tokens: Dict[str, str] = {}
        self.ts_lines: List[Tuple[int, str]] = []
        self.text_lines: List[str] = []
        self.flags: List[str] = []


def parse_filename(filename: str) -> Tuple[Optional[int], Optional[str]]:
    stem = PurePosixPath(filename).stem
    m = RE_FILENAME.match(stem)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def parse_episode(text: str, zip_member: str) -> Episode:
    filename = PurePosixPath(zip_member.replace("\\", "/")).name
    ep = Episode(filename=filename, zip_member=zip_member)
    lines = text.splitlines()
    ep.line_count = len(lines)

    # -- episode number: filename first, header as cross-check ---------------
    fn_num, asset = parse_filename(filename)
    ep.asset_id = asset
    for raw in lines:
        m = RE_HEADER_FIELD.match(norm_line(raw))
        if m and re.sub(r"\s+", " ", m.group(1).upper()) == "EPISODE NUMBER":
            d = re.search(r"(\d+)", m.group(2))
            ep.header_episode_number = int(d.group(1)) if d else None
            break
    if fn_num is not None:
        ep.episode_number, ep.episode_number_source = fn_num, "filename"
    elif ep.header_episode_number is not None:
        ep.episode_number, ep.episode_number_source = ep.header_episode_number, "header"
        add_exc(ep, "warning", "filename_unparsed",
                "Episode number not found in filename; taken from header.", raw=filename)
    else:
        add_exc(ep, "error", "episode_number_missing",
                "No episode number in filename or header.", raw=filename)
    if (fn_num is not None and ep.header_episode_number is not None
            and fn_num != ep.header_episode_number):
        add_exc(ep, "error", "episode_number_mismatch",
                f"Filename says {fn_num}, header says {ep.header_episode_number}.", raw=filename)
    if fn_num is not None and ep.header_episode_number is None:
        add_exc(ep, "warning", "header_episode_number_missing",
                "No parsable 'EPISODE NUMBER' header field.")

    # -- line-by-line state machine ------------------------------------------
    header_seen: Dict[str, str] = {}
    script_started = False
    script_ended = False
    cur_scene: Optional[Scene] = None
    cur: Optional[_Builder] = None
    builders: List[_Builder] = []
    scene_field_seen: Dict[int, set] = {}

    def finalize() -> None:
        nonlocal cur
        if cur is not None:
            builders.append(cur)
            cur = None

    def new_scene(sid: Optional[str], line_no: int) -> Scene:
        sc = Scene(scene_id=sid, scene_index=len(ep.scenes), line_no=line_no)
        ep.scenes.append(sc)
        scene_field_seen[sc.scene_index] = set()
        return sc

    for idx, raw_line in enumerate(lines, start=1):
        line = norm_line(raw_line)
        stripped = line.strip()
        if stripped == "":
            continue

        if RE_SCRIPT_START.match(line):
            finalize()
            if script_started:
                add_exc(ep, "warning", "duplicate_script_start", "Second SCRIPT START marker.",
                        line_no=idx, raw=raw_line)
            else:
                ep.script_start_line = idx
            script_started, script_ended = True, False
            continue
        if RE_SCRIPT_END.match(line):
            finalize()
            if ep.script_end_line is not None:
                add_exc(ep, "warning", "duplicate_script_end", "Second SCRIPT END marker.",
                        line_no=idx, raw=raw_line)
            ep.script_end_line = idx
            script_ended = True
            continue

        # ---- header region (before SCRIPT START) ---------------------------
        if not script_started:
            m = RE_HEADER_FIELD.match(line)
            if m:
                key = re.sub(r"\s+", " ", m.group(1).upper())
                if key in header_seen:
                    add_exc(ep, "warning", "duplicate_header_field", f"Repeated {key}.",
                            line_no=idx, raw=raw_line)
                else:
                    header_seen[key] = m.group(2)
                continue
            structural = (RE_SCENE_FIELD.match(line) or RE_SCENE_MARK.match(line)
                          or RE_BLOCK_HEADER.match(line))
            if structural:   # content without a SCRIPT START marker: parse it anyway
                add_exc(ep, "warning", "missing_script_start",
                        "Screenplay content found before any SCRIPT START marker.",
                        line_no=idx, raw=raw_line)
                script_started = True
            else:
                add_exc(ep, "warning", "unparsed_line", "Unrecognised line before SCRIPT START.",
                        line_no=idx, raw=raw_line)
                continue

        # ---- after SCRIPT END ----------------------------------------------
        if script_ended:
            add_exc(ep, "warning", "unparsed_line", "Content after SCRIPT END.",
                    line_no=idx, raw=raw_line)
            continue

        # ---- inside the script ---------------------------------------------
        m = RE_SCENE_FIELD.match(line)
        if m:
            finalize()
            if cur_scene is None:
                cur_scene = new_scene(None, idx)
                add_exc(ep, "warning", "scene_field_without_scene",
                        "Scene field appeared before any 'SCENE NN' marker.",
                        line_no=idx, raw=raw_line)
            key = m.group(1).upper()
            seen = scene_field_seen[cur_scene.scene_index]
            if key in seen:
                add_exc(ep, "warning", "duplicate_scene_field",
                        f"Repeated scene {key.lower()}; first value kept.",
                        line_no=idx, raw=raw_line, scene_id=cur_scene.scene_id)
            else:
                seen.add(key)
                if key == "HEADING":
                    cur_scene.heading_raw = m.group(2)
                elif key == "START":
                    cur_scene.start_raw = m.group(2).strip().strip("()").strip()
                else:
                    cur_scene.end_raw = m.group(2).strip().strip("()").strip()
            continue
        m = RE_SCENE_MARK.match(line)
        if m:
            finalize()
            cur_scene = new_scene(m.group(1), idx)
            continue
        if RE_ACTIONS_LABEL.match(line):
            finalize()
            continue
        m = RE_BLOCK_HEADER.match(line)
        if m:
            finalize()
            label = m.group(1).strip().rstrip(":").strip()
            cur = _Builder(label, idx, cur_scene)
            continue

        if cur is not None:
            cur.line_end = idx
            if RE_TS_LINE_START.match(line):
                pairs = RE_TS_PAIR.findall(line)
                if cur.ts_lines or cur.text_lines:
                    # a second timestamp line, or one after the text: keep, report
                    add_exc(ep, "warning", "extra_timestamp_line",
                            "Timestamp line is not the first line after the block header.",
                            line_no=idx, raw=raw_line)
                    cur.flags.append("extra_timestamp_line")
                    if cur.ts_lines:
                        continue
                cur.ts_lines.append((idx, raw_line))
                for k, v in pairs:
                    cur.ts_tokens.setdefault(k.lower(), v.strip())
                if not pairs:
                    cur.flags.append("unparsable_timestamp_line")
                    add_exc(ep, "error", "malformed_timestamp",
                            "Timestamp line has no recognisable Start:/End: token.",
                            line_no=idx, raw=raw_line)
            else:
                cur.text_lines.append(raw_line.strip())
            continue

        add_exc(ep, "warning", "unparsed_line",
                "Line is outside any event block and not a recognised structure.",
                line_no=idx, raw=raw_line,
                scene_id=cur_scene.scene_id if cur_scene else None)

    finalize()

    if not script_started:
        add_exc(ep, "error", "missing_script_markers", "No SCRIPT START marker and no screenplay content.")
    elif ep.script_end_line is None:
        add_exc(ep, "warning", "missing_script_end", "No SCRIPT END marker found.")
    for key in ("EPISODE NUMBER", "EPISODE LOGLINE", "CONTINUITY NOTES"):
        if key not in header_seen:
            add_exc(ep, "warning", "missing_header_field", f"Header field '{key}' not found.")
    ep.logline = header_seen.get("EPISODE LOGLINE")
    ep.continuity_notes = header_seen.get("CONTINUITY NOTES")

    build_events(ep, builders)
    finish_scenes(ep)
    return ep


def build_events(ep: Episode, builders: List[_Builder]) -> None:
    prefix = f"E{ep.episode_number:03d}" if ep.episode_number is not None else "EXXX"
    for order, b in enumerate(builders):
        info = classify_label(b.label_raw)
        ev = Event(
            event_id=f"{prefix}-{order + 1:04d}",
            episode_number=ep.episode_number,
            file_order=order,
            scene_id=b.scene.scene_id if b.scene else None,
            scene_index=b.scene.scene_index if b.scene else None,
            category=info["category"],
            label_raw=b.label_raw,
            speaker_raw=info["speaker_raw"],
            speaker=info["speaker"],
            delivery=info["delivery"],
            voice_modifiers=info["voice_modifiers"],
            line_start=b.line_start,
            line_end=b.line_end,
            flags=list(b.flags),
        )
        ev.text = "\n".join(b.text_lines).strip()
        ev.text_clean = ev.text
        if ev.category == "DIALOGUE":
            t = ev.text
            if len(t) >= 2 and t[0] in "\"“" and t[-1] in "\"”":
                ev.text_clean = t[1:-1].strip()
            elif t:
                ev.flags.append("unquoted_dialogue")
                add_exc(ep, "info", "unquoted_dialogue",
                        "Speaker block whose text is not wrapped in quotes.",
                        line_no=ev.line_start, raw=t[:200], event_order=order)
        if info["unbalanced"]:
            ev.flags.append("unbalanced_label_parentheses")
            add_exc(ep, "warning", "unbalanced_label_parentheses", "Label has unbalanced parentheses.",
                    line_no=ev.line_start, raw=b.label_raw, event_order=order)
        if info["empty_base"]:
            ev.flags.append("empty_label")
            add_exc(ep, "error", "empty_label", "Block label has no name.",
                    line_no=ev.line_start, raw=b.label_raw, event_order=order)
        if not ev.text:
            ev.flags.append("empty_text")
            add_exc(ep, "warning", "empty_text", "Event block has no text.",
                    line_no=ev.line_start, raw=b.label_raw, event_order=order)
        if b.scene is None:
            ev.flags.append("no_scene")
            add_exc(ep, "error", "event_without_scene",
                    "Event appears before any SCENE marker.",
                    line_no=ev.line_start, raw=b.label_raw, event_order=order)

        # timestamps
        for key, attr_raw, attr_ms in (("start", "start_raw", "start_ms"), ("end", "end_raw", "end_ms")):
            token = b.ts_tokens.get(key)
            if token is not None and token != "":
                setattr(ev, attr_raw, token)
            ms, issue = parse_timestamp(token)
            if issue == "missing":
                ev.flags.append(f"missing_{key}")
                add_exc(ep, "error", "missing_timestamp", f"No {key} timestamp.",
                        line_no=ev.line_start, raw=b.label_raw, event_order=order,
                        scene_id=ev.scene_id)
            elif issue == "malformed":
                ev.flags.append(f"malformed_{key}")
                add_exc(ep, "error", "malformed_timestamp",
                        f"Cannot parse {key} timestamp.", line_no=ev.line_start,
                        raw=b.label_raw, event_order=order, scene_id=ev.scene_id, ts_raw=token)
            else:
                setattr(ev, attr_ms, ms)
                if issue == "precision_truncated":
                    ev.flags.append(f"precision_truncated_{key}")
                    add_exc(ep, "warning", "precision_truncated",
                            f"{key} timestamp has more than 3 decimals; milliseconds truncated.",
                            line_no=ev.line_start, event_order=order, ts_raw=token)
        if ev.start_ms is not None and ev.end_ms is not None:
            ev.duration_ms = ev.end_ms - ev.start_ms
            ev.duration_s = round(ev.duration_ms / 1000.0, 3)
            if ev.duration_ms < 0:
                ev.flags.append("end_before_start")
                add_exc(ep, "error", "end_before_start", "End timestamp is before start.",
                        line_no=ev.line_start, event_order=order, scene_id=ev.scene_id,
                        ts_raw=f"{ev.start_raw} -> {ev.end_raw}")
            elif ev.duration_ms == 0:
                ev.flags.append("zero_duration")
                add_exc(ep, "info", "zero_duration", "Start equals end.",
                        line_no=ev.line_start, event_order=order, scene_id=ev.scene_id,
                        ts_raw=f"{ev.start_raw} -> {ev.end_raw}")
        ep.events.append(ev)
        if b.scene is not None:
            b.scene._event_orders.append(order)


def finish_scenes(ep: Episode) -> None:
    """Parse scene timestamps and fill derived fields."""
    for sc in ep.scenes:
        sc.int_ext, sc.time_of_day = parse_heading(sc.heading_raw)
        sc.event_ids = [ep.events[o].event_id for o in sc._event_orders]
        if sc.heading_raw is None:
            sc.flags.append("missing_heading")
            add_exc(ep, "warning", "scene_missing_heading", "Scene has no heading.",
                    line_no=sc.line_no, scene_id=sc.scene_id)
        for key in ("start", "end"):
            tok = getattr(sc, f"{key}_raw")
            ms, issue = parse_timestamp(tok)
            if issue == "missing":
                sc.flags.append(f"missing_{key}")
                add_exc(ep, "error", "scene_missing_timestamp", f"Scene {key} missing.",
                        line_no=sc.line_no, scene_id=sc.scene_id)
            elif issue == "malformed":
                sc.flags.append(f"malformed_{key}")
                add_exc(ep, "error", "scene_malformed_timestamp",
                        f"Scene {key} cannot be parsed.", line_no=sc.line_no,
                        scene_id=sc.scene_id, ts_raw=tok)
            else:
                setattr(sc, f"{key}_ms", ms)
                if issue == "precision_truncated":
                    sc.flags.append(f"precision_truncated_{key}")
        if sc.start_ms is not None and sc.end_ms is not None:
            sc.duration_ms = sc.end_ms - sc.start_ms
            sc.duration_s = round(sc.duration_ms / 1000.0, 3)
            if sc.duration_ms <= 0:
                sc.flags.append("scene_end_not_after_start")
                add_exc(ep, "error", "scene_end_not_after_start",
                        "Scene end is not after scene start.", line_no=sc.line_no,
                        scene_id=sc.scene_id, ts_raw=f"{sc.start_raw} -> {sc.end_raw}")
        if not sc._event_orders:
            sc.flags.append("scene_without_events")
            add_exc(ep, "warning", "scene_without_events", "Scene contains no event blocks.",
                    line_no=sc.line_no, scene_id=sc.scene_id)


# --------------------------------------------------------------------------- #
# Episode-level validation (needs all events parsed)
# --------------------------------------------------------------------------- #
def validate_episode(ep: Episode, tolerance_ms: int) -> None:
    ev_by_order = {e.file_order: e for e in ep.events}

    # -- chronological order (original order is preserved in ep.events) ------
    timed = [e for e in ep.events if e.start_ms is not None]
    timed_sorted = sorted(timed, key=lambda e: (e.start_ms, e.end_ms if e.end_ms is not None else e.start_ms,
                                                e.file_order))
    for i, e in enumerate(timed_sorted):
        e.sorted_index = i
    ep.events_chronological_ids = [e.event_id for e in timed_sorted]
    ep.events_unsortable_ids = [e.event_id for e in ep.events if e.start_ms is None]

    # -- out-of-order blocks (file order vs start time) ----------------------
    prev: Optional[Event] = None
    for e in ep.events:
        if e.start_ms is None:
            continue
        if prev is not None and e.start_ms < prev.start_ms:
            e.flags.append("out_of_order")
            add_exc(ep, "warning", "out_of_order",
                    f"Starts {(prev.start_ms - e.start_ms) / 1000:.2f}s before the previous block "
                    f"({prev.event_id}, {prev.category}) in file order.",
                    line_no=e.line_start, scene_id=e.scene_id, event_order=e.file_order,
                    ts_raw=f"{e.start_raw} (prev {prev.start_raw})")
        prev = e

    # -- overlap WITHIN one track only (cross-track overlap is by design) ----
    by_track: Dict[str, List[Event]] = defaultdict(list)
    for e in ep.events:
        if e.start_ms is not None and e.end_ms is not None and e.end_ms > e.start_ms:
            by_track[e.category].append(e)
    for track, evs in by_track.items():
        evs.sort(key=lambda e: (e.start_ms, e.end_ms, e.file_order))
        active: List[Event] = []
        for e in evs:
            active = [a for a in active if a.end_ms > e.start_ms]
            for a in active:
                overlap = min(a.end_ms, e.end_ms) - e.start_ms
                if overlap > 0:
                    for x in (a, e):
                        if "same_track_overlap" not in x.flags:
                            x.flags.append("same_track_overlap")
                    who = ""
                    if track == "DIALOGUE":
                        who = f" ({a.speaker} / {e.speaker})"
                    add_exc(ep, "warning", "same_track_overlap",
                            f"{track} events {a.event_id} and {e.event_id} overlap by "
                            f"{overlap / 1000:.2f}s{who}.",
                            line_no=e.line_start, scene_id=e.scene_id, event_order=e.file_order,
                            ts_raw=f"{a.start_raw}-{a.end_raw} vs {e.start_raw}-{e.end_raw}")
            active.append(e)

    # -- events outside their own scene's [start, end] -----------------------
    for sc in ep.scenes:
        if sc.start_ms is None or sc.end_ms is None:
            continue
        for o in sc._event_orders:
            e = ev_by_order[o]
            if e.start_ms is None and e.end_ms is None:
                continue
            before = (sc.start_ms - e.start_ms) if e.start_ms is not None else 0
            after = (e.end_ms - sc.end_ms) if e.end_ms is not None else 0
            if before > tolerance_ms or after > tolerance_ms:
                e.flags.append("out_of_scene")
                parts = []
                if before > tolerance_ms:
                    parts.append(f"starts {before / 1000:.2f}s before scene start")
                if after > tolerance_ms:
                    parts.append(f"ends {after / 1000:.2f}s after scene end")
                add_exc(ep, "warning", "out_of_scene", "; ".join(parts) + ".",
                        line_no=e.line_start, scene_id=sc.scene_id, event_order=e.file_order,
                        ts_raw=f"{e.start_raw}-{e.end_raw} vs scene {sc.start_raw}-{sc.end_raw}")

    # -- scene continuity ----------------------------------------------------
    ids_seen = Counter(s.scene_id for s in ep.scenes if s.scene_id is not None)
    for sid, n in ids_seen.items():
        if n > 1:
            add_exc(ep, "error", "duplicate_scene_id", f"Scene id {sid} appears {n} times.", scene_id=sid)
    numeric = [(s, int(s.scene_id)) for s in ep.scenes if s.scene_id and s.scene_id.isdigit()]
    for (s_prev, n_prev), (s_cur, n_cur) in zip(numeric, numeric[1:]):
        if n_cur != n_prev + 1:
            add_exc(ep, "warning", "scene_id_sequence",
                    f"Scene id jumps from {s_prev.scene_id} to {s_cur.scene_id}.",
                    scene_id=s_cur.scene_id, line_no=s_cur.line_no)
    first_timed = next((s for s in ep.scenes if s.start_ms is not None), None)
    if first_timed is not None and first_timed.start_ms != 0:
        add_exc(ep, "warning", "first_scene_not_at_zero",
                f"First scene starts at {first_timed.start_raw}, not 00:00.", scene_id=first_timed.scene_id,
                line_no=first_timed.line_no, ts_raw=first_timed.start_raw)
    for s_prev, s_cur in zip(ep.scenes, ep.scenes[1:]):
        if s_prev.end_ms is None or s_cur.start_ms is None:
            continue
        delta = s_cur.start_ms - s_prev.end_ms
        if abs(delta) > tolerance_ms:
            kind = "scene_gap" if delta > 0 else "scene_overlap"
            s_cur.flags.append(kind)
            add_exc(ep, "warning", kind,
                    f"Scene {s_cur.scene_id} starts {abs(delta) / 1000:.2f}s "
                    f"{'after' if delta > 0 else 'before'} scene {s_prev.scene_id} ends.",
                    scene_id=s_cur.scene_id, line_no=s_cur.line_no,
                    ts_raw=f"prev end {s_prev.end_raw}, start {s_cur.start_raw}")

    # -- per-episode statistics (spans, never sums of parallel tracks) -------
    starts = [e.start_ms for e in ep.events if e.start_ms is not None]
    ends = [e.end_ms for e in ep.events if e.end_ms is not None]
    sc_starts = [s.start_ms for s in ep.scenes if s.start_ms is not None]
    sc_ends = [s.end_ms for s in ep.scenes if s.end_ms is not None]
    declared = (max(sc_ends) - min(sc_starts)) if sc_starts and sc_ends else None
    span = (max(ends) - min(starts)) if starts and ends else None
    ep.stats = {
        "scene_count": len(ep.scenes),
        "event_count": len(ep.events),
        "events_by_category": dict(Counter(e.category for e in ep.events)),
        "declared_duration_ms": declared,
        "declared_duration_s": round(declared / 1000, 3) if declared is not None else None,
        "event_span_ms": span,
        "event_span_s": round(span / 1000, 3) if span is not None else None,
        "duration_note": "Durations are end-start of the scene/event span; parallel tracks are never summed.",
    }


# --------------------------------------------------------------------------- #
# Zip reading
# --------------------------------------------------------------------------- #
def decode_bytes(data: bytes) -> Tuple[str, bool]:
    try:
        return data.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), True


def read_zip(zip_path: Path, global_exc: List[Dict[str, Any]]) -> List[Tuple[str, bytes]]:
    members: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            name = info.filename
            if info.is_dir():
                continue
            parts = PurePosixPath(name.replace("\\", "/")).parts

            def skip(severity: str, why: str) -> None:
                global_exc.append({"episode_number": None, "filename": name, "severity": severity,
                                   "type": "skipped_zip_member", "scene_id": None, "event_id": None,
                                   "line_no": None, "timestamp_raw": None, "detail": why,
                                   "raw_content": name})
            if any(p.startswith("__MACOSX") or p.startswith(".") for p in parts):
                skip("info", "Hidden/metadata member skipped.")
                continue
            if not name.lower().endswith(".txt"):
                skip("warning", "Non-.txt member skipped.")
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                skip("error", f"Member larger than {MAX_MEMBER_BYTES} bytes; not read.")
                continue
            members.append((name, zf.read(info)))
    return members


# --------------------------------------------------------------------------- #
# Aggregation and reporting
# --------------------------------------------------------------------------- #
def collect_exceptions(episodes: List[Episode], global_exc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in global_exc:
        rows.append(g)
    for ep in episodes:
        by_order = {e.file_order: e for e in ep.events}
        for x in ep.exceptions:
            order = x.get("event_order")
            rows.append({
                "episode_number": ep.episode_number, "filename": ep.filename,
                "severity": x["severity"], "type": x["type"], "scene_id": x["scene_id"],
                "event_id": by_order[order].event_id if order in by_order else None,
                "line_no": x["line_no"], "timestamp_raw": x["timestamp_raw"],
                "detail": x["detail"], "raw_content": x["raw_content"],
            })
    return rows


def build_summary(episodes: List[Episode], exceptions: List[Dict[str, Any]],
                  parse_errors: List[Dict[str, str]], expected: Optional[int],
                  zip_path: Path, tolerance_ms: int) -> Dict[str, Any]:
    nums = [e.episode_number for e in episodes if e.episode_number is not None]
    counts = Counter(nums)
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    distinct = sorted(counts)
    if expected is not None:
        universe = range(1, expected + 1)
    elif distinct:
        universe = range(distinct[0], distinct[-1] + 1)
    else:
        universe = range(0)
    missing = [n for n in universe if n not in counts]
    unexpected = [n for n in distinct if expected is not None and not (1 <= n <= expected)]

    events = [ev for ep in episodes for ev in ep.events]
    scenes = [sc for ep in episodes for sc in ep.scenes]

    # timestamp tokens, counted from the parsed structure
    tok = {"valid": 0, "malformed": 0, "missing": 0}
    by_source = {"scene": {"valid": 0, "malformed": 0, "missing": 0},
                 "event": {"valid": 0, "malformed": 0, "missing": 0}}
    shapes: Counter = Counter()
    for source, items in (("scene", scenes), ("event", events)):
        for it in items:
            for key in ("start", "end"):
                raw, ms = getattr(it, f"{key}_raw"), getattr(it, f"{key}_ms")
                state = "missing" if raw is None else ("valid" if ms is not None else "malformed")
                tok[state] += 1
                by_source[source][state] += 1
                if raw is not None:
                    shapes[re.sub(r"\d", "9", raw)] += 1
    found = tok["valid"] + tok["malformed"]

    # durations (valid, non-negative only; each event on its own, never summed)
    def durs(items):
        return [i.duration_s for i in items if i.duration_ms is not None and i.duration_ms >= 0]
    cat_stats = {c: describe(durs([e for e in events if e.category == c]))
                 for c in sorted({e.category for e in events})}
    ep_dur = [ep.stats["declared_duration_s"] for ep in episodes
              if ep.stats.get("declared_duration_s") is not None]
    ep_span = [ep.stats["event_span_s"] for ep in episodes if ep.stats.get("event_span_s") is not None]
    negative = sum(1 for e in events if e.duration_ms is not None and e.duration_ms < 0)

    exc_by_type = Counter(x["type"] for x in exceptions)
    exc_by_sev = Counter(x["severity"] for x in exceptions)
    scene_cont = sum(exc_by_type.get(k, 0) for k in
                     ("scene_gap", "scene_overlap", "first_scene_not_at_zero",
                      "duplicate_scene_id", "scene_id_sequence", "scene_end_not_after_start"))
    by_track_overlap = Counter()
    for x in exceptions:
        if x["type"] == "same_track_overlap":
            by_track_overlap[x["detail"].split(" ")[0]] += 1

    speakers = Counter(e.speaker for e in events if e.speaker)
    eps_with_issue = sorted({x["episode_number"] for x in exceptions
                             if x["severity"] in ("error", "warning") and x["episode_number"] is not None})
    validation = {
        "episodes_detected": len(episodes),
        "expected_episodes": expected,
        "expected_check": None if expected is None else ("PASS" if len(episodes) == expected
                                                         and not missing and not duplicates
                                                         and not unexpected else "FAIL"),
        "episode_numbers_detected": distinct,
        "missing_episode_numbers": missing,
        "duplicate_episode_numbers": duplicates,
        "unexpected_episode_numbers": unexpected,
        "filename_header_mismatches": [x["episode_number"] for x in exceptions
                                       if x["type"] == "episode_number_mismatch"],
        "scene_continuity_issues": scene_cont,
        "events_outside_scene_bounds": exc_by_type.get("out_of_scene", 0),
        "out_of_order_events": exc_by_type.get("out_of_order", 0),
        "same_track_overlap_pairs": exc_by_type.get("same_track_overlap", 0),
        "same_track_overlap_by_track": dict(by_track_overlap),
        "unparsed_lines": exc_by_type.get("unparsed_line", 0),
    }
    return {
        "parser_version": PARSER_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_zip": zip_path.name,
        "tolerance_ms": tolerance_ms,
        "total_episodes": len(episodes),
        "total_scenes": len(scenes),
        "scenes_per_episode": describe([len(ep.scenes) for ep in episodes]),
        "total_events": len(events),
        "events_by_category": dict(Counter(e.category for e in events)),
        "events_per_episode": describe([len(ep.events) for ep in episodes]),
        "timestamp_tokens": {
            "found_total": found, "valid": tok["valid"], "malformed": tok["malformed"],
            "missing": tok["missing"], "by_source": by_source,
            "format_shapes": dict(shapes.most_common()),
            "note": "Counted from the parsed structure: every scene and event has a start and end slot.",
        },
        "duration_statistics_seconds": {
            "event_by_category": cat_stats,
            "event_all": describe(durs(events)),
            "scene": describe(durs(scenes)),
            "episode_declared_duration": describe(ep_dur),
            "episode_event_span": describe(ep_span),
            "events_with_negative_duration_excluded": negative,
            "note": "Each duration is end minus start of one item; parallel tracks are never summed.",
        },
        "distinct_speakers_raw": len(speakers),
        "top_speakers": speakers.most_common(10),
        "validation": validation,
        "exceptions_by_type": dict(exc_by_type.most_common()),
        "exceptions_by_severity": dict(exc_by_sev),
        "episodes_with_warnings_or_errors": eps_with_issue,
        "parsing_errors": parse_errors,
    }


def uniquify_event_ids(episodes: List[Episode]) -> None:
    """If two files share an episode number, suffix the later files' event ids
    (``~2``, ``~3``...) so every event_id in the outputs stays unique."""
    seen: Counter = Counter()
    for ep in episodes:
        seen[ep.episode_number] += 1
        k = seen[ep.episode_number]
        if ep.episode_number is None or k == 1:
            continue
        rename = {e.event_id: f"{e.event_id}~{k}" for e in ep.events}
        for e in ep.events:
            e.event_id = rename[e.event_id]
        for sc in ep.scenes:
            sc.event_ids = [rename[i] for i in sc.event_ids]
        ep.events_chronological_ids = [rename[i] for i in ep.events_chronological_ids]
        ep.events_unsortable_ids = [rename[i] for i in ep.events_unsortable_ids]


def episode_to_dict(ep: Episode) -> Dict[str, Any]:
    d = asdict(ep)
    for sc in d["scenes"]:
        sc.pop("_event_orders", None)
    d.pop("exceptions", None)
    d["anomaly_counts"] = dict(Counter(x["type"] for x in ep.exceptions))
    return d


EVENT_COLUMNS = ["episode_number", "filename", "event_id", "file_order", "sorted_index", "scene_id",
                 "category", "label_raw", "speaker_raw", "speaker", "delivery", "voice_modifiers",
                 "start_raw", "end_raw", "start_ms", "end_ms", "duration_ms", "duration_s",
                 "text", "flags", "line_start", "line_end"]
EXC_COLUMNS = ["episode_number", "filename", "severity", "type", "scene_id", "event_id",
               "line_no", "timestamp_raw", "detail", "raw_content"]


def write_outputs(out_dir: Path, episodes: List[Episode], summary: Dict[str, Any],
                  exceptions: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "parsed_episodes.json", "w", encoding="utf-8") as f:
        json.dump({"parser_version": PARSER_VERSION,
                   "episodes": [episode_to_dict(e) for e in episodes]}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "parsed_events.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        w.writeheader()
        for ep in episodes:
            for e in ep.events:
                row = asdict(e)
                row["filename"] = ep.filename
                row["voice_modifiers"] = ";".join(e.voice_modifiers)
                row["flags"] = ";".join(e.flags)
                w.writerow({k: row.get(k) for k in EVENT_COLUMNS})
    with open(out_dir / "parser_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "parser_exceptions.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXC_COLUMNS)
        w.writeheader()
        for x in exceptions:
            w.writerow({k: x.get(k) for k in EXC_COLUMNS})


def print_summary(summary: Dict[str, Any], out_dir: Path) -> None:
    v, t = summary["validation"], summary["timestamp_tokens"]
    d = summary["duration_statistics_seconds"]
    print("=" * 66)
    print(f"Screenplay parser v{PARSER_VERSION}  |  {summary['source_zip']}")
    print("=" * 66)
    chk = v["expected_check"]
    print(f"Episodes detected      : {v['episodes_detected']}"
          + (f"  (expected {v['expected_episodes']}: {chk})" if chk else "  (no expected count given)"))
    nums = v["episode_numbers_detected"]
    if nums:
        print(f"Episode numbering      : {nums[0]}..{nums[-1]}, "
              f"missing={v['missing_episode_numbers'] or 'none'}, "
              f"duplicates={v['duplicate_episode_numbers'] or 'none'}, "
              f"filename/header mismatches={v['filename_header_mismatches'] or 'none'}")
    print(f"Scenes / events        : {summary['total_scenes']} / {summary['total_events']}")
    print(f"Events by category     : {summary['events_by_category']}")
    print(f"Timestamp tokens       : {t['found_total']} found "
          f"({t['valid']} valid, {t['malformed']} malformed), {t['missing']} missing")
    ea = d["event_all"]
    if ea.get("count"):
        print(f"Event duration (s)     : min {ea['min']}  median {ea['median']}  "
              f"mean {ea['mean']}  p90 {ea['p90']}  max {ea['max']}")
    ed = d["episode_declared_duration"]
    if ed.get("count"):
        print(f"Episode duration (s)   : min {ed['min']}  median {ed['median']}  max {ed['max']}")
    print(f"Scene continuity issues: {v['scene_continuity_issues']}")
    print(f"Events outside scene   : {v['events_outside_scene_bounds']}")
    print(f"Out-of-order blocks    : {v['out_of_order_events']}")
    print(f"Same-track overlaps    : {v['same_track_overlap_pairs']} {v['same_track_overlap_by_track'] or ''}")
    print(f"Unparsed lines         : {v['unparsed_lines']}")
    print(f"Exceptions by severity : {summary['exceptions_by_severity']}")
    if summary["parsing_errors"]:
        print(f"PARSING ERRORS         : {summary['parsing_errors']}")
    print(f"Outputs written to     : {out_dir.resolve()}")
    print("=" * 66)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse screenplay episodes from a ZIP (Phase 1).")
    ap.add_argument("zip_path", type=Path, help="ZIP containing episode .txt files")
    ap.add_argument("--out-dir", type=Path, default=Path("parser_output"),
                    help="output directory (default: ./parser_output)")
    ap.add_argument("--expected-episodes", type=int, default=None,
                    help="optional: assert this many episodes (numbered 1..N)")
    ap.add_argument("--tolerance-ms", type=int, default=0,
                    help="slack for out-of-scene and scene-continuity checks (default 0)")
    args = ap.parse_args(argv)

    if not args.zip_path.is_file():
        print(f"error: {args.zip_path} not found", file=sys.stderr)
        return 2
    global_exc: List[Dict[str, Any]] = []
    try:
        members = read_zip(args.zip_path, global_exc)
    except zipfile.BadZipFile as exc:
        print(f"error: not a valid ZIP: {exc}", file=sys.stderr)
        return 2

    episodes: List[Episode] = []
    parse_errors: List[Dict[str, str]] = []
    for name, data in members:
        try:
            text, replaced = decode_bytes(data)
            ep = parse_episode(text, name)
            if replaced:
                add_exc(ep, "warning", "decode_replacements",
                        "File is not valid UTF-8; undecodable bytes were replaced.")
            validate_episode(ep, args.tolerance_ms)
            episodes.append(ep)
        except Exception as exc:  # keep going; record the failure
            parse_errors.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            global_exc.append({"episode_number": None, "filename": name, "severity": "error",
                               "type": "parse_failure", "scene_id": None, "event_id": None,
                               "line_no": None, "timestamp_raw": None,
                               "detail": f"{type(exc).__name__}: {exc}", "raw_content": name})

    episodes.sort(key=lambda e: (e.episode_number is None, e.episode_number or 0, e.filename))

    # cross-episode numbering checks
    seen = Counter(e.episode_number for e in episodes if e.episode_number is not None)
    for ep in episodes:
        if ep.episode_number is not None and seen[ep.episode_number] > 1:
            add_exc(ep, "error", "duplicate_episode_number",
                    f"Episode number {ep.episode_number} appears in {seen[ep.episode_number]} files.")

    uniquify_event_ids(episodes)
    exceptions = collect_exceptions(episodes, global_exc)
    summary = build_summary(episodes, exceptions, parse_errors, args.expected_episodes,
                            args.zip_path, args.tolerance_ms)
    write_outputs(args.out_dir, episodes, summary, exceptions)
    print_summary(summary, args.out_dir)

    failed = bool(parse_errors) or summary["validation"]["expected_check"] == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
