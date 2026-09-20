#!/usr/bin/env python3
"""Iterate one named run-block from configs/ablation_matrix.yaml and dispatch
`python gen.py …` per cell.

Each cell = one (dialect, target, ablation_condition, judges, tau, benchmark).
Stdout/stderr per cell go to results/private_raw/logs/<run_name>/<cell>.log.
The driver is resume-friendly: a cell whose record JSONL already has 50 rows
is skipped unless --force is passed.

Usage:
    python scripts/run_matrix.py --run phase_b_ablation
    python scripts/run_matrix.py --run phase_b_tau120 --dry-run
    python scripts/run_matrix.py --run phase_a_main --max-cells 1
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CODE = os.path.join(_ROOT, "code")
_CONFIGS = os.path.join(_ROOT, "configs")


def _count_csv_rows(path: str) -> int:
    """Count data rows (excluding header) in a CSV — used to know when a cell is complete."""
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def _count_jsonl_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _cell_slug(cell: dict) -> str:
    """Stable filesystem-safe identifier for a cell, used in log filenames."""
    judges = "+".join(cell["judges"])
    benchmark = cell["benchmark"]["name"]
    optimizer = cell.get("optimizer", "foa")
    slug = (
        f"{cell['dialect']}__{cell['target']}__{cell['ablation_condition']}"
        f"__tau{cell['tau']}__{benchmark}__judges-{judges}"
    )
    if optimizer != "foa":
        slug += f"__opt-{optimizer}"
    return slug


def _enumerate_cells(run_block: dict) -> list[dict]:
    """Cross-product the lists in a run block into individual cells."""
    cells = []
    optimizers = run_block.get("optimizers") or ["foa"]
    for dialect, target, cond, judges, tau, benchmark, optimizer in itertools.product(
        run_block["dialects"],
        run_block["targets"],
        run_block["ablation_conditions"],
        run_block["judges"],
        run_block["tau"],
        run_block["benchmarks"],
        optimizers,
    ):
        cells.append({
            "dialect": dialect,
            "target": target,
            "ablation_condition": cond,
            "judges": judges,
            "tau": tau,
            "benchmark": benchmark,
            "optimizer": optimizer,
        })
    return cells


def _build_gen_cmd(cell: dict, settings: dict, run_id: str) -> list[str]:
    """Construct the `python gen.py …` argv for one cell.

    `settings` is the effective {defaults ⊕ run-block overrides} dict —
    callers must merge those before invoking this helper.
    """
    cmd = [
        sys.executable, "gen.py",
        "--input_file", cell["benchmark"]["input_file"],
        "--dialect", cell["dialect"],
        "--target_model", cell["target"],
        "--ablation_condition", cell["ablation_condition"],
        "--early_stop_threshold", str(cell["tau"]),
        "--benchmark", cell["benchmark"]["name"],
        "--population_size", str(settings.get("population_size", 5)),
        "--max_iter", str(settings.get("max_iter", 5)),
        "--data_format", settings.get("data_format", "csv"),
        "--cache_dir", settings.get("cache_dir", "../cache"),
        "--output_dir", settings.get("output_dir", "../results/private_raw"),
        "--public_dir", settings.get("public_dir", "../results/public"),
        "--resume", "true" if settings.get("resume", True) else "false",
        "--run_id", run_id,
    ]
    for judge in cell["judges"]:
        cmd += ["--judge_model", judge]
    optimizer = cell.get("optimizer", "foa")
    if optimizer != "foa":
        cmd += ["--optimizer", optimizer]
    return cmd


def _record_path(cell: dict, output_dir: str) -> str:
    target_slug = cell["target"].replace(".", "-")
    benchmark_slug = cell["benchmark"]["name"].replace("/", "_").replace(".", "-")
    base = f"{cell['dialect']}_{target_slug}_{cell['ablation_condition']}_{benchmark_slug}"
    optimizer = cell.get("optimizer", "foa")
    if optimizer != "foa":
        base += f"_{optimizer}"
    return os.path.join(output_dir, f"record_{base}.jsonl")


def main() -> int:
    p = argparse.ArgumentParser(description="Iterate ablation_matrix.yaml run blocks")
    p.add_argument("--run", required=True, help="Name of a run block in configs/ablation_matrix.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the gen.py commands but do not execute them.")
    p.add_argument("--max-cells", type=int, default=None,
                   help="Stop after running this many cells (debugging).")
    p.add_argument("--force", action="store_true",
                   help="Run cells even if their record file already has all benchmark rows.")
    args = p.parse_args()

    matrix_path = os.path.join(_CONFIGS, "ablation_matrix.yaml")
    with open(matrix_path, encoding="utf-8") as f:
        matrix = yaml.safe_load(f)

    defaults = matrix.get("defaults", {}) or {}
    run_block = next((r for r in matrix["runs"] if r["name"] == args.run), None)
    if run_block is None:
        print(f"Unknown run name '{args.run}'. Available: {[r['name'] for r in matrix['runs']]}",
              file=sys.stderr)
        return 1

    # Effective settings = defaults ⊕ per-run overrides. Lets a smoke block
    # route its output to a separate dir without re-stating every default.
    settings = dict(defaults)
    settings.update(run_block.get("overrides", {}) or {})

    # Evaluation policy guardrail: a run whose name suggests a smoke must
    # write to a clearly-isolated output_dir. This is the second line of
    # defense beyond the benchmark slug in record filenames — it ensures a
    # smoke can never share a directory with a full eval.
    if "smoke" in args.run and "smoke" not in settings.get("output_dir", ""):
        print(
            f"[ERROR] Run '{args.run}' looks like a smoke run but its output_dir "
            f"({settings.get('output_dir')!r}) does not contain 'smoke'.\n"
            f"Add `overrides: {{output_dir: ../results/private_raw/smoke}}` to the "
            f"run block in configs/ablation_matrix.yaml to keep smoke artifacts "
            f"from contaminating full-eval results.",
            file=sys.stderr,
        )
        return 3

    cells = _enumerate_cells(run_block)
    print(f"Run '{args.run}': {len(cells)} cells (output_dir={settings.get('output_dir')})")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + args.run
    log_root = os.path.join(_ROOT, "results", "private_raw", "logs", args.run)
    os.makedirs(log_root, exist_ok=True)

    output_dir_abs = os.path.normpath(
        os.path.join(_CODE, settings.get("output_dir", "../results/private_raw"))
    )

    summary = {"completed": 0, "skipped": 0, "failed": 0, "dry_run": 0}

    for i, cell in enumerate(cells):
        if args.max_cells is not None and i >= args.max_cells:
            print(f"--max-cells={args.max_cells} reached; stopping.")
            break

        slug = _cell_slug(cell)
        input_file_abs = os.path.normpath(
            os.path.join(_ROOT, cell["benchmark"]["input_file"].lstrip("../"))
        )
        if not os.path.exists(input_file_abs):
            print(f"[{i+1}/{len(cells)}] {slug} — input file missing ({input_file_abs}), skipping")
            summary["skipped"] += 1
            continue
        target_rows = _count_csv_rows(input_file_abs)
        rec_path = _record_path(cell, output_dir_abs)
        existing = _count_jsonl_rows(rec_path)
        complete = target_rows > 0 and existing >= target_rows

        prefix = f"[{i+1}/{len(cells)}] {slug}"
        if complete and not args.force:
            print(f"{prefix} — already complete ({existing}/{target_rows} rows), skipping")
            summary["skipped"] += 1
            continue

        cmd = _build_gen_cmd(cell, settings, run_id)
        log_path = os.path.join(log_root, f"{slug}.log")

        if args.dry_run:
            print(f"{prefix} — dry-run\n   cmd: {' '.join(cmd)}\n   log: {log_path}")
            summary["dry_run"] += 1
            continue

        # Exclusive per-cell lock: prevents two concurrent run_matrix invocations
        # from running the same cell simultaneously, which would produce duplicate
        # records in the JSONL output.
        lock_path = os.path.join(log_root, f"{slug}.lock")
        try:
            lock_file = open(lock_path, "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            lock_file.close()
            print(f"{prefix} — locked by another run_matrix process, skipping")
            summary["skipped"] += 1
            continue

        try:
            # Re-check completion after acquiring lock: another process may have
            # just finished this cell between our initial check and lock acquisition.
            existing = _count_jsonl_rows(rec_path)
            complete = target_rows > 0 and existing >= target_rows
            if complete and not args.force:
                print(f"{prefix} — completed by another process ({existing}/{target_rows} rows), skipping")
                summary["skipped"] += 1
                continue

            print(f"{prefix} — running ({existing}/{target_rows} existing rows)")
            with open(log_path, "ab") as logf:
                logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting cell ===\n".encode())
                try:
                    proc = subprocess.run(cmd, cwd=_CODE, stdout=logf, stderr=subprocess.STDOUT)
                except KeyboardInterrupt:
                    print("\nInterrupted; abandoning matrix.")
                    return 130
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            try:
                os.unlink(lock_path)
            except OSError:
                pass

        if proc.returncode == 0:
            summary["completed"] += 1
        else:
            summary["failed"] += 1
            print(f"{prefix} — FAILED (rc={proc.returncode}); see {log_path}")

    print(json.dumps({"run": args.run, **summary}, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
