#!/usr/bin/env python3
"""Estimate the API spend of one or more matrix run blocks before launching.

Reads `configs/ablation_matrix.yaml` and prices each cell using rough
per-token costs. Numbers are conservative point estimates intended for
budget planning, not invoicing — actual costs depend on real token counts
and may differ by 2–3× either way.

Each FOA "attempt" issues four LLM calls:
  1. attack LLM (deepseek-chat) generates the mutated adversarial prompt
  2. target LLM (the cell's --target_model) replies
  3. translation LLM (deepseek-chat) renders the response back to English
  4. judge LLM(s) (--judge_model, repeatable) score it

Non-FOA conditions (english_original / mandarin_translation /
naive_dialect_translation) issue 1 attack-translation call + 1 target call
+ 1 translation call (skipped for english_original) + judge call(s).

Usage:
    python scripts/cost_preflight.py                 # all run blocks
    python scripts/cost_preflight.py --run phase_c_new_targets
    python scripts/cost_preflight.py --output results/public/c_run_estimate.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ── per-model rough cost-per-LLM-call ($USD), assuming ~500 input + 500
# output tokens per call. Updated 2026-05; refresh from each provider's
# pricing page before billing-sensitive estimates.
MODEL_COST_PER_CALL = {
    "gpt-4o":           0.0050,
    "deepseek-chat":    0.0005,
    "deepseek-reasoner": 0.0010,
    "claude-sonnet-4-6": 0.0075,
    "qwen-max":         0.0030,  # via OpenRouter
}

# Attack + translation always use deepseek-chat in the current pipeline.
ATTACK_COST = MODEL_COST_PER_CALL["deepseek-chat"]
TRANSLATE_COST = MODEL_COST_PER_CALL["deepseek-chat"]

# Average attempts (FOA evaluations) per goal, by early-stop threshold.
# Empirical from Phase A runs at tau=80; tau=120 doubles roughly.
AVG_ATTEMPTS_BY_TAU = {80: 4, 120: 8}

NON_FOA_CONDITIONS = {
    "english_original": {"foa_attempts": 1, "skip_translation": True,  "skip_attack": True},
    "mandarin_translation": {"foa_attempts": 1, "skip_translation": False, "skip_attack": False},
    "naive_dialect_translation": {"foa_attempts": 1, "skip_translation": False, "skip_attack": False},
}


def _count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def _target_cost(target: str) -> float:
    return MODEL_COST_PER_CALL.get(target, 0.005)  # default to gpt-4o-level


def _judge_cost(judges: list[str]) -> float:
    return sum(MODEL_COST_PER_CALL.get(j, 0.005) for j in judges)


def _estimate_cell(cell: dict) -> dict:
    cond = cell["ablation_condition"]
    tau = cell["tau"]
    benchmark_path = os.path.join(_ROOT, cell["benchmark"]["input_file"].lstrip("../"))
    n_prompts = _count_csv_rows(benchmark_path)
    if n_prompts == 0:
        n_prompts = 50  # fall back to the AdvBench default if file not present yet

    if cond in NON_FOA_CONDITIONS:
        spec = NON_FOA_CONDITIONS[cond]
        attempts = spec["foa_attempts"]
        per_attempt = (
            (0 if spec["skip_attack"] else ATTACK_COST)
            + _target_cost(cell["target"])
            + (0 if spec["skip_translation"] else TRANSLATE_COST)
            + _judge_cost(cell["judges"])
        )
    else:
        attempts = AVG_ATTEMPTS_BY_TAU.get(tau, 4)
        per_attempt = (
            ATTACK_COST
            + _target_cost(cell["target"])
            + TRANSLATE_COST
            + _judge_cost(cell["judges"])
        )
    cell_cost = n_prompts * attempts * per_attempt
    return {
        "n_prompts": n_prompts,
        "attempts_per_prompt": attempts,
        "per_attempt_cost": round(per_attempt, 5),
        "cell_cost": round(cell_cost, 2),
    }


def _enumerate_cells(run_block: dict) -> list[dict]:
    cells = []
    for d, t, c, j, tau, b in itertools.product(
        run_block["dialects"], run_block["targets"],
        run_block["ablation_conditions"], run_block["judges"],
        run_block["tau"], run_block["benchmarks"],
    ):
        cells.append({
            "dialect": d, "target": t, "ablation_condition": c,
            "judges": j, "tau": tau, "benchmark": b,
        })
    return cells


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate API spend for matrix run blocks.")
    p.add_argument("--run", default=None, help="A specific run-block name; default: all blocks.")
    p.add_argument("--output", default=os.path.join(_ROOT, "results", "public", "c_run_estimate.md"))
    args = p.parse_args()

    matrix_path = os.path.join(_ROOT, "configs", "ablation_matrix.yaml")
    with open(matrix_path, encoding="utf-8") as f:
        matrix = yaml.safe_load(f)

    blocks = matrix["runs"]
    if args.run:
        blocks = [r for r in blocks if r["name"] == args.run]
        if not blocks:
            print(f"Unknown run '{args.run}'", file=sys.stderr)
            return 1

    lines = ["# Cost preflight estimate", "",
             "_Rough budget estimate — actual API spend may vary 2–3× either direction._", ""]

    grand_total = 0.0
    for block in blocks:
        cells = _enumerate_cells(block)
        block_total = 0.0
        lines.append(f"## Run block: `{block['name']}` ({len(cells)} cells)")
        lines.append("")
        lines.append("| dialect | target | condition | tau | benchmark | n_prompts | attempts | $/attempt | cell $ |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for cell in cells:
            est = _estimate_cell(cell)
            block_total += est["cell_cost"]
            lines.append(
                f"| {cell['dialect']} | {cell['target']} | {cell['ablation_condition']} | "
                f"{cell['tau']} | {cell['benchmark']['name']} | "
                f"{est['n_prompts']} | {est['attempts_per_prompt']} | "
                f"${est['per_attempt_cost']:.4f} | ${est['cell_cost']:.2f} |"
            )
        lines.append("")
        lines.append(f"**Block subtotal: ${block_total:.2f}**")
        lines.append("")
        grand_total += block_total

    if len(blocks) > 1:
        lines.append(f"## Grand total: ${grand_total:.2f}")
        lines.append("")
    lines.append("### Pricing assumptions")
    lines.append("")
    lines.append("| model | $/call (≈500 in / 500 out) |")
    lines.append("|---|---:|")
    for m, c in MODEL_COST_PER_CALL.items():
        lines.append(f"| {m} | ${c:.4f} |")
    lines.append("")
    lines.append(f"Avg FOA attempts per goal: tau=80 → {AVG_ATTEMPTS_BY_TAU[80]}, "
                 f"tau=120 → {AVG_ATTEMPTS_BY_TAU[120]} (empirical, prior Phase A runs).")
    lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")
    print(f"Grand total estimate: ${grand_total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
