# Golden Reference Datasets

This directory maintains manifests and metadata for the golden benchmark datasets prescribed by System Design Section 102:

- `small_smoke_dataset`: Minimal dataset for fast regression and unit testing (~100 cells).
- `human_pbmc_dataset`: Standard human PBMC 3k/10k dataset for cell typing and marker recovery.
- `mouse_dataset`: Standard mouse tissue dataset for organism-specific annotation tests.
- `multi_sample_dataset`: Multi-sample dataset for testing sample-aware QC and batch structures.
- `batch_effect_dataset`: Multi-batch dataset for batch evaluation and integration tests.
- `known_treatment_dataset`: Condition-controlled dataset (e.g. Treatment vs Control with replicates) for differential expression verification.
- `large_benchmark_dataset`: High-scale dataset (up to 1M cells) for resource and memory stress testing.

## Data Storage Policy
Raw count matrices are not committed directly to Git. They are tracked via immutable SHA-256 digests in `golden_manifest.json` and mirrored in R2 reference storage.
