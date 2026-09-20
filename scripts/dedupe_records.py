#!/usr/bin/env python3
"""Dedupe record_*.jsonl files by `id`, keeping the LAST occurrence per id.

Background: prior to commit 010bf56 ("harden resume: ... dedup completed_ids"),
gen.py could write multiple rows for the same prompt id within a single session
when the FOA loop retried a fly. Several record files in results/private_raw/
ended up with 1+ extra rows per id; if the duplicates have different `success`
or `score` values, naive row-count aggregation is biased.

Policy: keep the LAST row per id. Rationale — the file is append-only and the
last write is the most recent state under the current code path. This matches
how `get_completed_ids()` interprets the file (later rows shadow earlier ones).

Atomic: writes the deduped output to <file>.tmp then renames over the original.
Idempotent: re-running on an already-deduped file is a no-op.

Usage:
    python scripts/dedupe_records.py results/private_raw/record_*.jsonl
    python scripts/dedupe_records.py --check results/private_raw/record_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def scan(path: str) -> tuple[int, int, list[int]]:
    """Return (n_rows, n_unique_ids, list_of_dup_ids) for the file."""
    ids: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("id")
            except json.JSONDecodeError:
                continue
            if rid is not None:
                ids.append(rid)
    seen: set = set()
    dups: list = []
    for rid in ids:
        if rid in seen:
            dups.append(rid)
        seen.add(rid)
    return len(ids), len(seen), sorted(set(dups))


def dedupe(path: str) -> tuple[int, int]:
    """Rewrite `path` keeping only the last row per id. Returns (kept, dropped)."""
    last_by_id: dict = {}
    order: list = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rid = json.loads(stripped).get("id")
            except json.JSONDecodeError:
                continue
            if rid is None:
                continue
            if rid not in last_by_id:
                order.append(rid)
            last_by_id[rid] = stripped
    # Preserve first-seen order of ids (stable + readable).
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for rid in order:
            f.write(last_by_id[rid] + "\n")
        f.flush()
        os.fsync(f.fileno())
    n_kept = len(last_by_id)
    n_total_before = sum(1 for _ in open(path) if _.strip())
    os.replace(tmp_path, path)
    return n_kept, n_total_before - n_kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--check", action="store_true",
                   help="Report duplicates without modifying files.")
    args = p.parse_args()

    rc = 0
    for path in args.files:
        if not os.path.exists(path):
            print(f"{path}: missing", file=sys.stderr)
            rc = 2
            continue
        n_rows, n_uniq, dup_ids = scan(path)
        if n_rows == n_uniq:
            print(f"{path}: clean ({n_rows} rows, {n_uniq} unique ids)")
            continue
        n_dups = n_rows - n_uniq
        if args.check:
            print(f"{path}: {n_rows} rows, {n_uniq} unique, {n_dups} dups across {len(dup_ids)} ids")
            rc = 1
            continue
        kept, dropped = dedupe(path)
        print(f"{path}: deduped — kept {kept} rows, dropped {dropped} dups")
    return rc


if __name__ == "__main__":
    sys.exit(main())
