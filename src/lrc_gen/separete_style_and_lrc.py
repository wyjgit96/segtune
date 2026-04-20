#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse a structured .lrc (with labels like [Intro][...]) and produce:
  1) JSON style description with global_prompt + local_prompts (time ranges + prompts)
  2) Cleaned .lrc with ALL structural lines removed and NO blank lines.

Usage:
  python build_styles_and_clean_lrc.py \
      --lrc input_with_sections.lrc \
      --global-prompt "A sentimental Chinese pop ballad sung by a young adult male vocalist, with emotional and nostalgic mood." \
      --out-json style.json \
      --out-lrc lyrics_clean.lrc
"""

import re
import json
import argparse
from typing import List, Tuple, Optional

ALLOWED_LABELS = {"Start", "End", "Intro", "Outro", "Break", "Bridge", "Inst", "Solo", "Verse", "Chorus"}

# [mm:ss.xx] at line start
TS_RE = re.compile(r"^\[(\d{2}):(\d{2}\.\d{2})\]")
# Structural line: [mm:ss.xx][Label][Optional description...]
STRUCT_RE = re.compile(r"^\[(\d{2}):(\d{2}\.\d{2})\]\[([A-Za-z]+)\](?:\[(.*?)\])?\s*$")


def ts_to_seconds(ts: str) -> float:
    mm, sscc = ts.split(":")
    ss, cc = sscc.split(".")
    return int(mm) * 60 + int(ss) + int(cc) / 100.0


def find_line_timestamp(line: str) -> Optional[float]:
    m = TS_RE.match(line.strip())
    if not m:
        return None
    mm, sscc = m.groups()
    return ts_to_seconds(f"{mm}:{sscc}")


def parse_structural_line(line: str):
    m = STRUCT_RE.match(line.strip())
    if not m:
        return None
    mm, sscc, label, desc = m.groups()
    if label not in ALLOWED_LABELS:
        return None
    t = ts_to_seconds(f"{mm}:{sscc}")
    return (t, label, (desc.strip() if desc is not None else None))


def build_sections(struct_lines, last_timestamp):
    """
    Fixed:
    - For each structural line WITH description, find the next structural timestamp t2 where t2 > t.
    - If not found, use last_timestamp.
    - Prevent end < start.
    """
    sections = []
    n = len(struct_lines)
    for i, (t, lbl, desc) in enumerate(struct_lines):
        if not desc or not desc.strip():
            continue  # only labeled+described sections go to JSON
        # find next strictly greater timestamp
        end_t = last_timestamp if last_timestamp is not None else t
        j = i + 1
        while j < n:
            t2 = struct_lines[j][0]
            if t2 > t:
                end_t = t2
                break
            j += 1
        if end_t < t:
            end_t = t
        sections.append((t, end_t, desc.strip()))
    return sections


def round2(x: float) -> float:
    return float(f"{x:.2f}")


def process(lrc_text: str, global_prompt: str):
    lines = [ln.rstrip("\n") for ln in lrc_text.splitlines()]

    all_ts: List[float] = []
    struct_lines: List[Tuple[float, str, Optional[str]]] = []
    cleaned_lines: List[str] = []

    for ln in lines:
        t = find_line_timestamp(ln)
        if t is not None:
            all_ts.append(t)

        s = parse_structural_line(ln)
        if s is not None:
            struct_lines.append(s)
            continue  # drop all structural rows from output LRC

        # keep non-struct row as-is (we'll remove blanks later)
        cleaned_lines.append(ln)

    # sort structural lines by time
    struct_lines.sort(key=lambda x: x[0])
    last_timestamp = max(all_ts) if all_ts else None

    # Build JSON local prompts
    sections_triplets = build_sections(struct_lines, last_timestamp if last_timestamp is not None else 0.0)
    local_prompts = [{"section": [round2(s), round2(e)], "prompt": desc} for (s, e, desc) in sections_triplets]
    style_json = {"global_prompt": global_prompt.strip(), "local_prompts": local_prompts}

    # ---- remove ALL blank lines in cleaned LRC ----
    cleaned_lines_no_blank = [ln for ln in cleaned_lines if ln.strip() != ""]

    # (optional) also drop lines that are just a timestamp with no lyric text:
    # cleaned_lines_no_blank = [
    #     ln for ln in cleaned_lines_no_blank
    #     if not (TS_RE.match(ln) and ln.strip() == TS_RE.match(ln).group(0))
    # ]

    cleaned_lrc = "\n".join(cleaned_lines_no_blank)
    if cleaned_lrc and not cleaned_lrc.endswith("\n"):
        cleaned_lrc += "\n"

    return style_json, cleaned_lrc


def main():
    ap = argparse.ArgumentParser(
        description="Build JSON style description and cleaned LRC from a structured .lrc file."
    )
    ap.add_argument("--lrc", required=True, help="Path to input .lrc file containing structural labels + descriptions.")
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument("--global-prompt", help="Global song description as a string.")
    gp.add_argument("--global-prompt-file", help="Path to a text file containing the global song description.")
    ap.add_argument("--out-json", default="style.json", help="Output JSON path (default: style.json)")
    ap.add_argument("--out-lrc", default="lyrics_clean.lrc", help="Output cleaned LRC path (default: lyrics_clean.lrc)")
    args = ap.parse_args()

    with open(args.lrc, "r", encoding="utf-8") as f:
        lrc_text = f.read()

    if args.global_prompt is not None:
        global_prompt = args.global_prompt
    else:
        with open(args.global_prompt_file, "r", encoding="utf-8") as f:
            global_prompt = f.read().strip()

    style_json, cleaned_lrc = process(lrc_text, global_prompt)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(style_json, f, ensure_ascii=False, indent=4)

    with open(args.out_lrc, "w", encoding="utf-8") as f:
        f.write(cleaned_lrc)

    print(f"[OK] Wrote JSON: {args.out_json}")
    print(f"[OK] Wrote LRC:  {args.out_lrc}")


if __name__ == "__main__":
    main()
