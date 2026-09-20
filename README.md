# What Drives Dialectal Jailbreaks?

Official code for **“What Drives Dialectal Jailbreaks? An Ablation of Surface Form, Cultural Framing, and Strategy Banks,”** accepted at EMNLP 2026.

This repository extends the [CC-BOS](https://github.com/xunhuang123/CC-BOS) fruit-fly optimization framework to study dialectal jailbreaks in Classical Chinese, Shanghainese/Wu, and Cantonese/Yue. It includes the dialect registry, strategy banks, translation and quality-control pipeline, ablation conditions, resumable experiment drivers, aggregate-analysis scripts, and mocked tests.

## Responsible release

This code is intended for authorized safety research. Optimized adversarial prompts, harmful model responses, raw per-example evaluation records, API credentials, and internal research artifacts are intentionally excluded. Generated raw records are written to `results/private_raw/`, which is gitignored. Only aggregate, non-content metrics should be shared.

## Setup

Requirements:

- Python 3.8+
- Conda
- API access for the model providers you plan to evaluate

Create the environment and configure credentials:

```bash
conda env create -f environment.yml
conda activate ase
cp .env.example .env
```

Fill only the required variables in `.env`. Shell environment variables take precedence. The supported providers and model roles are defined in `code/config.py`.

The benchmark data are not distributed in this repository. Obtain them from their original sources, review their terms, and place normalized inputs under `data/`. Every input CSV must contain `goal`, `target`, and `intention` columns. See `data/data.md` for dataset citations.

## Run an experiment

`gen.py` must be run from `code/` because project modules use sibling imports:

```bash
cd code
python gen.py \
  --input_file ../data/advbench_50.csv \
  --dialect shanghainese \
  --target_model gpt-4o \
  --judge_model gpt-4o \
  --ablation_condition full_dialect_cultural_frame
```

Important options include:

- `--dialect`: `classical_chinese`, `shanghainese`, `cantonese`, or the `generic` control
- `--target_model`: a target registered in `code/config.py`
- `--judge_model`: repeatable; the first judge controls early stopping and ASR
- `--ablation_condition`: one of the conditions in `configs/ablation_dimensions.yaml`
- `--optimizer`: `foa`, `random_one`, `best_of_k`, or `greedy_coord`
- `--early_stop_threshold`: normally `80` or `120`
- `--resume true`: skips completed prompt IDs and supports interrupted runs
- `--output_dir`: raw JSONL destination, defaulting to `results/private_raw/`

The optimizer writes three dialect- and condition-namespaced JSONL files: final prompts, full evaluation records, and translation-quality flags. These files may contain sensitive content and must remain private.

## Experiment matrix and analysis

Run a named matrix block from the repository root:

```bash
python scripts/run_matrix.py --run phase_b_ablation --dry-run
python scripts/run_matrix.py --run phase_b_ablation
```

The matrix runner uses stable cell names, per-cell locks, separate smoke-test output, and row-count checkpoints to avoid duplicate work and resume partial evaluations.

Create publication-safe aggregate tables from private records:

```bash
python scripts/aggregate_results.py
python scripts/compute_confidence_intervals.py
```

Review aggregate outputs before sharing them. Do not publish raw JSONL files.

## Tests

All LLM calls are mocked; API keys are not required:

```bash
python tests/test_dialect_pipeline.py
python tests/test_judge_irr_sampler.py
python tests/test_second_judge_backfill_manifest.py
```

## Repository layout

```text
code/       optimizer, model registry, dialect strategies, translation, scoring
configs/    ablation definitions and experiment matrix
scripts/    resumable experiment and aggregate-analysis utilities
tests/      mocked regression tests
data/       dataset documentation and non-sensitive support metadata
```

## Citation

```bibtex
@inproceedings{xu2026dialectal,
  title     = {What Drives Dialectal Jailbreaks? An Ablation of Surface Form, Cultural Framing, and Strategy Banks},
  author    = {Xu, Qingyang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

This project builds on CC-BOS. Please also cite the original CC-BOS paper when using the optimization framework.

## License

This project is released under the [MIT License](LICENSE).
