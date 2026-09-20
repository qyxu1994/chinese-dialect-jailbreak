#!/usr/bin/env python3
"""Normalize a CLAS-style adversarial-prompt dataset to the AdvBench schema
used by gen.py.

CLAS (Competition for LLM Attacks and Safeguards, NeurIPS 2024) does not
ship a single canonical public URL like StrongREJECT. This loader expects
the user to download or export the dataset to a local CSV/JSONL, then
normalizes its columns into (goal, target, intention[, category]).

Column-name auto-detection covers common variants:
  goal-like:      goal | prompt | instruction | behavior | request | adversarial_prompt
  category-like:  category | harm_category | type | label

Usage:
    # Auto-detect columns from a local CSV
    python scripts/load_clas.py --raw_csv /path/to/clas.csv

    # Explicit column mapping
    python scripts/load_clas.py --raw_csv /path/to/clas.jsonl --goal_col Instruction --cat_col Category

    # Sample down to N prompts
    python scripts/load_clas.py --raw_csv /path/to/clas.csv --n 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

GOAL_CANDIDATES = (
    "goal", "prompt", "instruction", "behavior", "request",
    "adversarial_prompt", "forbidden_prompt", "harmful_query",
)
CATEGORY_CANDIDATES = (
    "category", "harm_category", "type", "label", "topic",
)


def _read_any(path: str) -> list[dict]:
    """Read a CSV or JSONL file into a list of dicts."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=delim))
    if ext in (".jsonl", ".ndjson"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError(f"Unsupported JSON shape in {path}; expected top-level list or {{data: [...]}}")
    raise ValueError(f"Unsupported file extension: {ext}")


def _auto_detect(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_orig = {h.lower(): h for h in headers}
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    return None


def _normalize(rows: list[dict], goal_col: str, cat_col: str | None) -> list[dict]:
    out = []
    for r in rows:
        goal = (r.get(goal_col) or "").strip()
        if not goal:
            continue
        out.append({
            "goal": goal,
            "target": "Sure, here is …",
            "intention": goal,
            "category": (r.get(cat_col) or "uncategorized") if cat_col else "uncategorized",
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Normalize CLAS dataset to AdvBench schema.")
    p.add_argument("--raw_csv", required=True,
                   help="Path to a local CLAS dataset file (.csv / .tsv / .jsonl / .json).")
    p.add_argument("--goal_col", default=None,
                   help="Column name carrying the harmful goal. Auto-detected from common variants if omitted.")
    p.add_argument("--cat_col", default=None,
                   help="Column name carrying a category label. Auto-detected if omitted; "
                        "uses 'uncategorized' if no candidate column is present.")
    p.add_argument("--n", type=int, default=100,
                   help="Sample size cap (default: 100). Stratified across categories when present.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=os.path.join(_ROOT, "data", "clas.csv"))
    args = p.parse_args()

    rows = _read_any(args.raw_csv)
    if not rows:
        print(f"No rows in {args.raw_csv}", file=sys.stderr)
        return 1

    headers = list(rows[0].keys())
    goal_col = args.goal_col or _auto_detect(headers, GOAL_CANDIDATES)
    if goal_col is None:
        print(
            f"Could not auto-detect a goal column in {args.raw_csv}.\n"
            f"  headers: {headers}\n"
            f"  candidates tried: {GOAL_CANDIDATES}\n"
            f"Pass --goal_col explicitly.",
            file=sys.stderr,
        )
        return 2
    cat_col = args.cat_col or _auto_detect(headers, CATEGORY_CANDIDATES)
    print(f"Using goal_col={goal_col!r}, cat_col={cat_col!r}")

    normalized = _normalize(rows, goal_col, cat_col)
    print(f"Normalized {len(normalized)} rows")

    # Stratified down-sample by category.
    if len(normalized) > args.n:
        rng = random.Random(args.seed)
        by_cat: dict[str, list[dict]] = {}
        for r in normalized:
            by_cat.setdefault(r["category"], []).append(r)
        per_cat = max(1, args.n // max(1, len(by_cat)))
        sampled: list[dict] = []
        for cat, bucket in by_cat.items():
            rng.shuffle(bucket)
            sampled.extend(bucket[:per_cat])
        if len(sampled) < args.n:
            remaining = [r for r in normalized if r not in sampled]
            rng.shuffle(remaining)
            sampled.extend(remaining[: args.n - len(sampled)])
        rng.shuffle(sampled)
        normalized = sampled[: args.n]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["goal", "target", "intention", "category"])
        writer.writeheader()
        writer.writerows(normalized)
    print(f"Wrote {len(normalized)} rows → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
