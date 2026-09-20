"""Tests for second_judge_backfill --sample_manifest behavior."""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TESTS_DIR)
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
CODE_DIR = os.path.join(ROOT_DIR, 'code')
for d in (SCRIPTS_DIR, CODE_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

# Stub torch before importing project code; matches test_dialect_pipeline.py.
sys.modules.setdefault('torch', MagicMock())

import second_judge_backfill as sjb  # noqa: E402


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


class TestProcessFileAllowedIds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = os.path.join(self.tmp.name, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.records = os.path.join(self.tmp.name, 'record.jsonl')
        _write_jsonl(self.records, [
            {'id': 1, 'intention': 'q1', 'model_response': 'r1', 'judge_scores': {}},
            {'id': 2, 'intention': 'q2', 'model_response': 'r2', 'judge_scores': {}},
            {'id': 3, 'intention': 'q3', 'model_response': 'r3', 'judge_scores': {}},
        ])
        # Monkey-patch the network call to avoid real API hits.
        self._orig_score = sjb._score_with_judge
        sjb._score_with_judge = lambda judge, intention, response: 60.0
        self.addCleanup(lambda: setattr(sjb, '_score_with_judge', self._orig_score))

    def _read_rows(self) -> list[dict]:
        with open(self.records) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_allowed_ids_restricts_scoring(self):
        n_rows, n_calls = sjb._process_file(
            self.records, ['qwen-max'], self.cache_dir,
            refresh_refusal=False, verbose=False, allowed_ids={1, 3},
        )
        self.assertEqual(n_rows, 3)
        self.assertEqual(n_calls, 2)
        rows = self._read_rows()
        self.assertEqual(rows[0]['judge_scores'], {'qwen-max': 60.0})
        self.assertEqual(rows[1]['judge_scores'], {})
        self.assertEqual(rows[2]['judge_scores'], {'qwen-max': 60.0})

    def test_allowed_ids_none_scores_all(self):
        n_rows, n_calls = sjb._process_file(
            self.records, ['qwen-max'], self.cache_dir,
            refresh_refusal=False, verbose=False, allowed_ids=None,
        )
        self.assertEqual(n_rows, 3)
        self.assertEqual(n_calls, 3)
        for r in self._read_rows():
            self.assertEqual(r['judge_scores'], {'qwen-max': 60.0})

    def test_allowed_ids_empty_set_scores_nothing(self):
        n_rows, n_calls = sjb._process_file(
            self.records, ['qwen-max'], self.cache_dir,
            refresh_refusal=False, verbose=False, allowed_ids=set(),
        )
        self.assertEqual(n_rows, 3)
        self.assertEqual(n_calls, 0)
        for r in self._read_rows():
            self.assertEqual(r['judge_scores'], {})

    def test_cached_score_reused_for_allowed_row(self):
        # Pre-populate cache for row 1; should be used (no API call).
        sjb._write_cache('qwen-max', 'q1', 'r1', 99.0, self.cache_dir)
        # Make the patched scorer raise to prove it isn't invoked for cached rows.
        sjb._score_with_judge = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError('should not be called for cached row'))
        n_rows, n_calls = sjb._process_file(
            self.records, ['qwen-max'], self.cache_dir,
            refresh_refusal=False, verbose=False, allowed_ids={1},
        )
        self.assertEqual(n_rows, 3)
        self.assertEqual(n_calls, 0)
        rows = self._read_rows()
        self.assertEqual(rows[0]['judge_scores'], {'qwen-max': 99.0})


class TestLoadManifest(unittest.TestCase):
    def test_load_manifest_returns_file_to_id_set(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'manifest.json')
            with open(path, 'w') as f:
                json.dump({
                    'judge': 'qwen-max',
                    'samples': [
                        {'file': 'results/private_raw/a.jsonl',
                         'row_ids': [1, 2, 3], 'skipped_reason': None},
                        {'file': 'results/private_raw/b.jsonl',
                         'row_ids': [], 'skipped_reason': 'empty_or_missing'},
                        {'file': 'results/private_raw/c.jsonl',
                         'row_ids': [7], 'skipped_reason': None},
                    ],
                }, f)
            mapping = sjb._load_sample_manifest(path)
        self.assertEqual(mapping, {
            'results/private_raw/a.jsonl': {1, 2, 3},
            'results/private_raw/c.jsonl': {7},
        })


if __name__ == '__main__':
    unittest.main()
