#!/usr/bin/env python3
"""Resume-loop supervisor for run_matrix.py.

Re-launches `python run_matrix.py --run <name>` until every cell in the
named run block has its full target row count, or --max_attempts retries
are exhausted. Backs off between attempts so a sustained outage (quota
exhausted, network down) sleeps quietly rather than hot-spinning.

Per-cell completion is computed identically to run_matrix.py (matching
record_path + target_rows from the input CSV), so partial cells resume
on the next attempt and complete cells skip via the existing
already-complete check inside run_matrix.py.

Usage:
    python scripts/resume_loop.py phase_b_generic_controls
    python scripts/resume_loop.py phase_e_matched_budget_baselines \\
        --max_attempts 30 --sleep_seconds 300
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CODE = os.path.join(_ROOT, "code")
_CONFIGS = os.path.join(_ROOT, "configs")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_matrix as rm  # noqa: E402


def _count_complete(run_block: dict, defaults: dict) -> tuple[int, int]:
    settings = dict(defaults)
    settings.update(run_block.get("overrides", {}) or {})
    output_dir_abs = os.path.normpath(
        os.path.join(_CODE, settings.get("output_dir", "../results/private_raw"))
    )
    cells = rm._enumerate_cells(run_block)
    n_complete = 0
    for cell in cells:
        input_file_abs = os.path.normpath(
            os.path.join(_ROOT, cell["benchmark"]["input_file"].lstrip("../"))
        )
        if not os.path.exists(input_file_abs):
            continue
        target = rm._count_csv_rows(input_file_abs)
        path = rm._record_path(cell, output_dir_abs)
        existing = rm._count_jsonl_rows(path)
        if target > 0 and existing >= target:
            n_complete += 1
    return n_complete, len(cells)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run block name from configs/ablation_matrix.yaml")
    p.add_argument("--max_attempts", type=int, default=20,
                   help="Maximum supervisor retries before giving up (default 20).")
    p.add_argument("--sleep_seconds", type=int, default=300,
                   help="Seconds to sleep between attempts (default 300).")
    args = p.parse_args()

    matrix_path = os.path.join(_CONFIGS, "ablation_matrix.yaml")
    with open(matrix_path, encoding="utf-8") as f:
        matrix = yaml.safe_load(f)
    defaults = matrix.get("defaults", {}) or {}
    run_block = next((r for r in matrix["runs"] if r["name"] == args.run), None)
    if run_block is None:
        print(f"Unknown run name '{args.run}'", file=sys.stderr)
        return 2

    for attempt in range(1, args.max_attempts + 1):
        done, total = _count_complete(run_block, defaults)
        print(f"[{_ts()}] supervisor attempt {attempt}/{args.max_attempts}: "
              f"{done}/{total} cells complete", flush=True)
        if done >= total:
            print(f"[{_ts()}] ALL CELLS COMPLETE for {args.run}", flush=True)
            return 0

        rc = subprocess.call(
            [sys.executable, "-u", "run_matrix.py", "--run", args.run],
            cwd=_HERE,
        )
        print(f"[{_ts()}] run_matrix.py exited rc={rc}", flush=True)

        done, total = _count_complete(run_block, defaults)
        if done >= total:
            print(f"[{_ts()}] ALL CELLS COMPLETE for {args.run}", flush=True)
            return 0

        if attempt < args.max_attempts:
            print(f"[{_ts()}] sleeping {args.sleep_seconds}s before retry "
                  f"({done}/{total} cells done)", flush=True)
            time.sleep(args.sleep_seconds)

    done, total = _count_complete(run_block, defaults)
    print(f"[{_ts()}] MAX_ATTEMPTS reached; {done}/{total} cells complete",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
