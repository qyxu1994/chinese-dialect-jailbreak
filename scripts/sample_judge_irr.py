"""Stratified subsample of paper-cell records for inter-judge IRR backfill.

Emits analysis/irr/judge_qwen_manifest.json — a list of (file, [row_ids])
listing 5 rows per (dialect x target x condition) paper-cell. Designed to be
consumed by scripts/second_judge_backfill.py --sample_manifest.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import random
import sys


_DIALECTS = ('classical_chinese', 'shanghainese', 'cantonese')
_TARGETS = ('gpt-4o', 'deepseek-reasoner')
_CONDITIONS = (
    'english_original',
    'mandarin_translation',
    'naive_dialect_translation',
    'dialect_without_cultural_frame',
    'mandarin_with_cultural_frame',
    'full_dialect_cultural_frame',
)

# phase_a (full method only) + phase_b (all 6 conditions)
# phase_a uses 'full_dialect_cultural_frame' so it overlaps with one phase_b
# cell; we count it once per (dialect, target).
PAPER_CELL_FILENAMES: tuple[str, ...] = tuple(
    f'record_{d}_{t}_{c}_advbench.jsonl'
    for d in _DIALECTS
    for t in _TARGETS
    for c in _CONDITIONS
)
# That's 3 * 2 * 6 = 36 phase_b cells. phase_a's 6 cells share the
# 'full_dialect_cultural_frame' filename with phase_b — both phases write
# to the same record file, so the 36-cell list already covers phase_a's
# rows. Verified by tests/test_judge_irr_sampler.py.


def sample_file(path: str, n: int, rng: random.Random) -> list[str]:
    """Sample up to n eligible row ids from a JSONL record file.

    Eligible: has a non-empty 'id' AND a non-empty 'model_response' or
    'raw_response'. Returns [] if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    eligible_ids: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get('id')
            if not row_id:
                continue
            response = row.get('model_response') or row.get('raw_response')
            if not response:
                continue
            eligible_ids.append(row_id)
    if not eligible_ids:
        return []
    k = min(n, len(eligible_ids))
    return rng.sample(eligible_ids, k)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records_dir', default='results/private_raw')
    p.add_argument('--n_per_cell', type=int, default=6)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out_path',
                   default='analysis/irr/judge_qwen_manifest.json')
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    samples = []
    for fname in PAPER_CELL_FILENAMES:
        full = os.path.join(args.records_dir, fname)
        row_ids = sample_file(full, args.n_per_cell, rng)
        samples.append({
            'file': os.path.relpath(full),
            'row_ids': row_ids,
            'skipped_reason': None if row_ids else 'empty_or_missing',
        })

    manifest = {
        'judge': 'qwen-max',
        'seed': args.seed,
        'n_per_cell': args.n_per_cell,
        'generated_at': _dt.datetime.now().isoformat(timespec='seconds'),
        'samples': samples,
    }
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total = sum(len(s['row_ids']) for s in samples)
    skipped = sum(1 for s in samples if not s['row_ids'])
    print(f'wrote {args.out_path}: {total} row_ids across '
          f'{len(samples) - skipped}/{len(samples)} files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
