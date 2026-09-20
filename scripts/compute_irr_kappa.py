#!/usr/bin/env python3
"""Compute Cohen's kappa between two coders on the FOA IRR sample,
plus prevalence comparison vs. the non-FOA control sample.

Writes a markdown report to results/public/irr_kappa_report.md
and prints the same content to stdout.

Usage:
    python scripts/compute_irr_kappa.py
"""
from __future__ import annotations

import csv
import os
import sys


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float('nan')
    p_o = sum(x == y for x, y in zip(a, b)) / n
    a1 = sum(a) / n
    b1 = sum(b) / n
    p_e = a1 * b1 + (1 - a1) * (1 - b1)
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else float('nan')
    return (p_o - p_e) / (1 - p_e)


def load(path: str) -> tuple[list[str], list[dict]]:
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    mechs = [c for c in fieldnames
             if c not in ('blind_id', 'source_id', 'notes')]
    return mechs, rows


def main() -> int:
    base = 'analysis/coding_irr'
    a_path = os.path.join(base, 'coder_a.csv')
    b_path = os.path.join(base, 'coder_b.csv')
    nf_path = os.path.join(base, 'non_foa_control.csv')

    for p in (a_path, b_path, nf_path):
        if not os.path.exists(p):
            print(f'ERROR: missing {p}', file=sys.stderr)
            return 1

    mechs_a, a = load(a_path)
    mechs_b, b = load(b_path)
    if mechs_a != mechs_b:
        print('ERROR: mechanism columns differ between coders', file=sys.stderr)
        return 1
    mechs_nf, nf = load(nf_path)

    # Align by blind_id.
    a_by = {r['blind_id']: r for r in a}
    b_by = {r['blind_id']: r for r in b}
    shared = sorted(set(a_by) & set(b_by))
    if len(shared) != 20:
        print(f'WARNING: {len(shared)} shared blind_ids (expected 20)',
              file=sys.stderr)

    lines: list[str] = []
    lines.append("# IRR Cohen's kappa Report")
    lines.append('')
    lines.append(f'n(coder_a) = {len(a)}, n(coder_b) = {len(b)}, '
                 f'n(shared) = {len(shared)}, n(non_foa_control) = {len(nf)}')
    lines.append('')
    lines.append('## Per-mechanism kappa (FOA sample, coder_a vs. coder_b)')
    lines.append('')
    lines.append('| mechanism | kappa | p(a=1) | p(b=1) |')
    lines.append('|---|---:|---:|---:|')
    kappas: list[float] = []
    for m in mechs_a:
        av = [int(a_by[i][m]) for i in shared]
        bv = [int(b_by[i][m]) for i in shared]
        k = cohens_kappa(av, bv)
        kappas.append(k)
        lines.append(f'| {m} | {k:+.3f} | {sum(av)/len(av):.2f} | '
                     f'{sum(bv)/len(bv):.2f} |')
    # Median kappa (excluding NaN)
    sk = sorted(k for k in kappas if k == k)
    if sk:
        median = sk[len(sk) // 2]
        lines.append('')
        lines.append(f'Median kappa (excluding NaN): **{median:+.3f}**, '
                     f'range [{min(sk):+.3f}, {max(sk):+.3f}]')
    else:
        lines.append('')
        lines.append('Median kappa: NaN (all mechanisms had degenerate prevalence)')
    lines.append('')
    lines.append('## Prevalence: FOA vs. non-FOA (template-artifact control)')
    lines.append('')
    lines.append('| mechanism | FOA p (coder_a) | non-FOA p | delta |')
    lines.append('|---|---:|---:|---:|')
    nf_fields = set(nf[0].keys()) if nf else set()
    for m in mechs_a:
        if m not in nf_fields:
            continue
        foa_p = sum(int(r[m]) for r in a) / len(a) if a else 0.0
        nf_p = sum(int(r[m]) for r in nf) / len(nf) if nf else 0.0
        lines.append(f'| {m} | {foa_p:.2f} | {nf_p:.2f} | '
                     f'{foa_p - nf_p:+.2f} |')
    report = '\n'.join(lines) + '\n'
    out_path = 'results/public/irr_kappa_report.md'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)
    print(report)
    print(f'\n[wrote {out_path}]', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
