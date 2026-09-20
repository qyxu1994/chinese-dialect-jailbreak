#!/usr/bin/env bash
# Phase B.3 — tau=120 reruns of the full_dialect_cultural_frame condition for
# direct CC-BOS-style peak-toxicity comparison. 6 cells × 50 prompts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python "$HERE/run_matrix.py" --run phase_b_tau120 "$@"
