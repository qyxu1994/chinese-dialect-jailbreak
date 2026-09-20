#!/usr/bin/env python3
"""Show progress of a second-judge backfill by counting how many record
rows already have a given judge's score populated.

Usage:
    python scripts/judge_progress.py --judge qwen-max
    python scripts/judge_progress.py --judge qwen-max --records_dir results/private_raw
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(
        description="Show progress of second-judge backfill.")
    p.add_argument("--judge", required=True,
                   help="Judge model name (e.g. qwen-max, claude-sonnet-4-6)")
    p.add_argument("--records_dir", default="results/private_raw")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.records_dir, "record_*.jsonl")))
    if not files:
        print(f"no record_*.jsonl files under {args.records_dir}", file=sys.stderr)
        return 1

    total = judged = 0
    by_file = []
    for f in files:
        n = j = 0
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if args.judge in (d.get("judge_scores") or {}):
                j += 1
        total += n
        judged += j
        by_file.append((os.path.basename(f), j, n))

    width = max(len(name) for name, _, _ in by_file)
    for name, j, n in by_file:
        bar = "DONE" if n > 0 and j == n else f"{j:3d}/{n:<3d}"
        print(f"  {name.ljust(width)}  {bar}")
    pct = 100 * judged / total if total else 0.0
    print(f"\n  total: {judged}/{total} rows judged ({pct:.1f}%) with '{args.judge}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
