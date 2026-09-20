#!/usr/bin/env python3
"""Sample blinded prompt subsets for R5 IRR coding.

Picks 20 successful FOA prompts (from the FINAL data, not the v1 snapshot)
stratified across living dialects (Shanghainese, Cantonese) and 20 non-FOA
prompts as a template-artifact control. Output CSVs are blinded (no dialect /
target / condition columns adjacent to the prompt text) and seeded for
reproducibility.

The output CSVs contain raw adversarial prompts and MUST NOT be committed.
See analysis/coding_irr/ gitignore entry.

Usage:
    python scripts/sample_qualitative_subset.py
    python scripts/sample_qualitative_subset.py --seed 2026 --n_foa 20 --n_non_foa 20
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys


def collect(records_dir: str, condition_filter, success_only: bool,
            dialects: set[str]) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(records_dir, 'record_*.jsonl')):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get('dialect') not in dialects:
                    continue
                if not condition_filter(d.get('ablation_condition'),
                                        d.get('optimizer') or 'foa'):
                    continue
                if success_only and not d.get('success'):
                    continue
                rows.append(d)
    return rows


def write_sample(out_path: str, sample: list[dict], blind_prefix: str) -> None:
    """Write a blinded CSV: blind_id + source_id only; the dialect / target /
    condition / optimizer fields are written so the unblinder can recover them
    AFTER coding, but the coder's tool should hide every column past `prompt`."""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['blind_id', 'source_id', 'adversarial_prompt',
                    '_dialect', '_target', '_condition', '_optimizer'])
        for i, d in enumerate(sample):
            w.writerow([
                f'{blind_prefix}_{i:02d}',
                d.get('id'),
                d.get('adversarial_prompt', ''),
                d.get('dialect'),
                d.get('target_model'),
                d.get('ablation_condition'),
                d.get('optimizer') or 'foa',
            ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--records_dir', default='results/private_raw')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out_dir', default='analysis/coding_irr')
    p.add_argument('--n_foa', type=int, default=20)
    p.add_argument('--n_non_foa', type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # 20 successful FOA prompts on the full method, restricted to living dialects.
    foa = collect(
        args.records_dir,
        lambda c, o: c == 'full_dialect_cultural_frame' and o == 'foa',
        success_only=True,
        dialects={'shanghainese', 'cantonese'},
    )
    if len(foa) < args.n_foa:
        print(f'ERROR: only {len(foa)} eligible FOA prompts available; '
              f'need {args.n_foa}', file=sys.stderr)
        return 1
    foa_sample = random.sample(foa, args.n_foa)

    # 20 non-FOA prompts as template-artifact control.
    # Includes: non-FOA ablation conditions (single-call baselines)
    # OR FOA conditions evaluated under a non-foa optimizer.
    non_foa = collect(
        args.records_dir,
        lambda c, o: (c in {'english_original', 'mandarin_translation',
                            'naive_dialect_translation'}
                      or o != 'foa'),
        success_only=False,
        dialects={'shanghainese', 'cantonese', 'classical_chinese'},
    )
    if len(non_foa) < args.n_non_foa:
        print(f'ERROR: only {len(non_foa)} eligible non-FOA prompts; '
              f'need {args.n_non_foa}', file=sys.stderr)
        return 1
    non_foa_sample = random.sample(non_foa, args.n_non_foa)

    foa_path = os.path.join(args.out_dir, 'sample_foa.csv')
    nf_path = os.path.join(args.out_dir, 'sample_non_foa.csv')
    write_sample(foa_path, foa_sample, 'foa')
    write_sample(nf_path, non_foa_sample, 'nonfoa')
    print(f'wrote {foa_path}: {len(foa_sample)} rows')
    print(f'wrote {nf_path}: {len(non_foa_sample)} rows')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
