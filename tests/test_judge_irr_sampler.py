import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TESTS_DIR)
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import sample_judge_irr as sji  # noqa: E402


class TestAllowList(unittest.TestCase):
    def test_paper_cell_filenames_have_36_entries(self):
        names = sji.PAPER_CELL_FILENAMES
        self.assertEqual(len(names), 36)
        self.assertEqual(len(set(names)), 36, "no duplicates")

    def test_paper_cell_filenames_cover_3_dialects(self):
        names = sji.PAPER_CELL_FILENAMES
        for dialect in ('classical_chinese', 'shanghainese', 'cantonese'):
            count = sum(1 for n in names if dialect in n)
            self.assertEqual(count, 12)

    def test_paper_cell_filenames_cover_both_targets(self):
        names = sji.PAPER_CELL_FILENAMES
        for target in ('gpt-4o', 'deepseek-reasoner'):
            count = sum(1 for n in names if target in n)
            self.assertEqual(count, 18)


import json
import random
import tempfile


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


class TestSampleFile(unittest.TestCase):
    def test_returns_n_row_ids_when_file_has_more(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'rec.jsonl')
            rows = [{'id': f'r{i}', 'model_response': 'ok'} for i in range(50)]
            _write_jsonl(p, rows)
            picked = sji.sample_file(p, n=6, rng=random.Random(2026))
            self.assertEqual(len(picked), 6)
            self.assertEqual(len(set(picked)), 6, "no duplicate ids")

    def test_returns_all_when_under_n(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'rec.jsonl')
            rows = [{'id': f'r{i}', 'model_response': 'ok'} for i in range(3)]
            _write_jsonl(p, rows)
            picked = sji.sample_file(p, n=6, rng=random.Random(2026))
            self.assertEqual(sorted(picked), ['r0', 'r1', 'r2'])

    def test_skips_rows_without_response(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'rec.jsonl')
            rows = [
                {'id': 'r0', 'model_response': 'ok'},
                {'id': 'r1'},                          # no response field
                {'id': 'r2', 'model_response': ''},    # empty
                {'id': 'r3', 'raw_response': 'fb'},    # raw_response fallback
            ]
            _write_jsonl(p, rows)
            picked = sji.sample_file(p, n=10, rng=random.Random(2026))
            self.assertEqual(sorted(picked), ['r0', 'r3'])

    def test_skips_rows_without_id(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'rec.jsonl')
            rows = [
                {'id': 'r0', 'model_response': 'ok'},
                {'model_response': 'ok'},  # no id
            ]
            _write_jsonl(p, rows)
            picked = sji.sample_file(p, n=10, rng=random.Random(2026))
            self.assertEqual(picked, ['r0'])

    def test_missing_file_returns_empty(self):
        picked = sji.sample_file('/nonexistent/path.jsonl', n=6,
                                 rng=random.Random(2026))
        self.assertEqual(picked, [])

    def test_deterministic_with_same_seed(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'rec.jsonl')
            rows = [{'id': f'r{i}', 'model_response': 'ok'} for i in range(50)]
            _write_jsonl(p, rows)
            a = sji.sample_file(p, n=6, rng=random.Random(2026))
            b = sji.sample_file(p, n=6, rng=random.Random(2026))
            self.assertEqual(a, b)


import subprocess


class TestCli(unittest.TestCase):
    def test_main_writes_manifest_against_real_records_dir(self):
        # End-to-end against the real results/private_raw. Should produce a
        # 36-entry manifest. Skip if the records dir is empty (smoke envs).
        records_dir = os.path.join(ROOT_DIR, 'results', 'private_raw')
        if not os.path.isdir(records_dir):
            self.skipTest('no records dir; smoke-only env')
        any_file = any(os.path.exists(os.path.join(records_dir, n))
                       for n in sji.PAPER_CELL_FILENAMES)
        if not any_file:
            self.skipTest('no paper-cell record files present')
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, 'manifest.json')
            rc = sji.main([
                '--records_dir', records_dir,
                '--n_per_cell', '6',
                '--seed', '2026',
                '--out_path', out_path,
            ])
            self.assertEqual(rc, 0)
            with open(out_path) as f:
                manifest = json.load(f)
            self.assertEqual(manifest['judge'], 'qwen-max')
            self.assertEqual(manifest['n_per_cell'], 6)
            self.assertEqual(manifest['seed'], 2026)
            self.assertEqual(len(manifest['samples']), 36)
            # At least some files should have row_ids populated.
            nonempty = [s for s in manifest['samples'] if s['row_ids']]
            self.assertGreater(len(nonempty), 0)


if __name__ == '__main__':
    unittest.main()
