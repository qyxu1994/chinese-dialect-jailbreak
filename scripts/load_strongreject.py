#!/usr/bin/env python3
"""Fetch the StrongREJECT dataset, sample a small subset, and normalize it
to the AdvBench schema used by gen.py.

Source: https://github.com/alexandrasouly/strongreject (CC-BY-4.0).
The raw CSV has columns (category, source, forbidden_prompt). We sample
`--n` prompts stratified by category and write data/strongreject_small.csv
with columns (goal, target, intention, category) — matching AdvBench's
(goal, target, intention) plus a `category` column for downstream slicing.

`intention` is a copy of `goal` because StrongREJECT does not ship a
separate refined intention column (AdvBench does). `target` is a fixed
"Sure, here is …" placeholder, matching AdvBench's style — it's used
only as a metadata column by gen.py and does not influence the attack.

Usage:
    python scripts/load_strongreject.py
    python scripts/load_strongreject.py --n 60 --seed 42
    python scripts/load_strongreject.py --raw_csv /path/to/local/strongreject.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

STRONGREJECT_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)


def _fetch_to_cache(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "strongreject_dataset.csv")
    if os.path.exists(path):
        return path
    print(f"Downloading {url} → {path}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    return path


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample roughly n prompts evenly across the `category` column."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "uncategorized"), []).append(r)
    cats = sorted(by_cat.keys())
    per_cat = max(1, n // len(cats))
    out: list[dict] = []
    for cat in cats:
        bucket = by_cat[cat]
        rng.shuffle(bucket)
        out.extend(bucket[:per_cat])
    # Top up or trim to exactly n.
    if len(out) < n:
        remaining = [r for r in rows if r not in out]
        rng.shuffle(remaining)
        out.extend(remaining[: n - len(out)])
    rng.shuffle(out)
    return out[:n]


def _normalize(rows: list[dict]) -> list[dict]:
    """Map StrongREJECT columns → AdvBench-compatible (goal, target, intention, category)."""
    out = []
    for r in rows:
        goal = (r.get("forbidden_prompt") or "").strip()
        if not goal:
            continue
        out.append({
            "goal": goal,
            "target": "Sure, here is …",  # AdvBench-style placeholder, not used by gen.py
            "intention": goal,
            "category": r.get("category", "uncategorized"),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Normalize StrongREJECT to AdvBench schema.")
    p.add_argument("--n", type=int, default=60, help="Number of prompts to sample (default: 60).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw_csv", default=None,
                   help="Path to a local StrongREJECT CSV; skips the HTTP fetch.")
    p.add_argument("--output", default=os.path.join(_ROOT, "data", "strongreject_small.csv"))
    p.add_argument("--cache_dir", default=os.path.join(_ROOT, "cache", "benchmarks"))
    args = p.parse_args()

    if args.raw_csv:
        raw = args.raw_csv
    else:
        try:
            raw = _fetch_to_cache(STRONGREJECT_URL, args.cache_dir)
        except Exception as e:
            print(
                f"Could not fetch StrongREJECT from {STRONGREJECT_URL}: {e}\n"
                f"Download the CSV manually and rerun with "
                f"--raw_csv /path/to/strongreject_dataset.csv",
                file=sys.stderr,
            )
            return 2

    rows = _read_rows(raw)
    print(f"Read {len(rows)} StrongREJECT rows from {raw}")

    rng = random.Random(args.seed)
    sampled = _stratified_sample(rows, args.n, rng)
    normalized = _normalize(sampled)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["goal", "target", "intention", "category"])
        writer.writeheader()
        writer.writerows(normalized)

    print(f"Wrote {len(normalized)} rows → {args.output}")
    cats = sorted({r['category'] for r in normalized})
    print(f"Categories present ({len(cats)}): {cats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
