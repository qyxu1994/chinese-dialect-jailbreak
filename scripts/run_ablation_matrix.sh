#!/usr/bin/env bash
# Phase B.3 — full 3 dialects × 6 conditions × 2 targets × tau=80 ablation grid.
# 36 cells × 50 prompts on AdvBench-50. Resume-friendly.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python "$HERE/run_matrix.py" --run phase_b_ablation "$@"
