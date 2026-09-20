#!/usr/bin/env python3
"""Print Phase B/C run progress at a glance.

Walks `configs/ablation_matrix.yaml`, counts JSONL rows per expected cell,
estimates % completion, lists which cell is currently running (by mtime of
the per-cell log), and shows the most recent log lines.

Usage:
    python scripts/progress.py                     # all known run blocks
    python scripts/progress.py --run phase_b_ablation
    python scripts/progress.py --run phase_b_ablation --tail 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CONFIGS = os.path.join(_ROOT, "configs", "ablation_matrix.yaml")


def _count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def _count_jsonl_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _record_path(cell: dict, output_dir: str) -> str:
    target_slug = cell["target"].replace(".", "-")
    bench_slug = cell["benchmark"]["name"].replace("/", "_").replace(".", "-")
    base = f"{cell['dialect']}_{target_slug}_{cell['ablation_condition']}_{bench_slug}"
    return os.path.join(output_dir, f"record_{base}.jsonl")


def _cell_slug(cell: dict) -> str:
    judges = "+".join(cell["judges"])
    return (
        f"{cell['dialect']}__{cell['target']}__{cell['ablation_condition']}"
        f"__tau{cell['tau']}__{cell['benchmark']['name']}__judges-{judges}"
    )


def _enumerate_cells(block: dict) -> list[dict]:
    return [
        {"dialect": d, "target": t, "ablation_condition": c,
         "judges": j, "tau": tau, "benchmark": b}
        for d, t, c, j, tau, b in itertools.product(
            block["dialects"], block["targets"], block["ablation_conditions"],
            block["judges"], block["tau"], block["benchmarks"],
        )
    ]


def _human_age(ts: float) -> str:
    delta = max(0, time.time() - ts)
    if delta < 60:
        return f"{delta:.0f}s ago"
    if delta < 3600:
        return f"{delta/60:.1f}m ago"
    return f"{delta/3600:.1f}h ago"


def _print_block(block: dict, settings: dict, tail: int) -> None:
    name = block["name"]
    output_dir = os.path.normpath(
        os.path.join(_ROOT, "code", settings.get("output_dir", "../results/private_raw"))
    )
    cells = _enumerate_cells(block)

    print(f"\n=== {name}  ({len(cells)} cells)  output_dir={output_dir} ===")

    n_complete = n_running = n_pending = 0
    total_done = total_expected = 0
    rows = []
    for cell in cells:
        bench_path = os.path.join(_ROOT, cell["benchmark"]["input_file"].lstrip("../"))
        target_rows = _count_csv_rows(bench_path) or 50
        rec_path = _record_path(cell, output_dir)
        have = _count_jsonl_rows(rec_path)
        total_done += have
        total_expected += target_rows
        if have >= target_rows:
            status = "✓"; n_complete += 1
        elif have > 0:
            status = "⋯"; n_running += 1
        else:
            status = " "; n_pending += 1
        rows.append((status, have, target_rows, _cell_slug(cell)))

    pct = (100.0 * total_done / total_expected) if total_expected else 0.0
    print(f"  rows: {total_done}/{total_expected} ({pct:.1f}%)   "
          f"cells: {n_complete} done, {n_running} running, {n_pending} pending")

    for status, have, target, slug in rows:
        print(f"  {status}  {have:>3}/{target:<3}  {slug}")

    # Most recently-touched log = best guess at the currently-running cell.
    log_dir = os.path.join(_ROOT, "results", "private_raw", "logs", name)
    logs = sorted(glob.glob(os.path.join(log_dir, "*.log")), key=os.path.getmtime, reverse=True)
    if logs:
        newest = logs[0]
        print(f"\n  most recent log: {os.path.basename(newest)}  "
              f"({_human_age(os.path.getmtime(newest))})")
        if tail > 0:
            with open(newest, errors="replace") as f:
                content = f.read().splitlines()[-tail:]
            for line in content:
                print(f"    | {line}")


def main() -> int:
    p = argparse.ArgumentParser(description="Show progress of matrix run blocks.")
    p.add_argument("--run", default=None, help="A specific run block; default: all blocks.")
    p.add_argument("--tail", type=int, default=0,
                   help="Show the last N lines of the most recent per-cell log.")
    args = p.parse_args()

    with open(_CONFIGS) as f:
        matrix = yaml.safe_load(f)

    defaults = matrix.get("defaults", {}) or {}
    blocks = matrix["runs"]
    if args.run:
        blocks = [b for b in blocks if b["name"] == args.run]
        if not blocks:
            print(f"Unknown run name '{args.run}'", file=sys.stderr)
            return 1

    for block in blocks:
        settings = dict(defaults)
        settings.update(block.get("overrides", {}) or {})
        _print_block(block, settings, args.tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
