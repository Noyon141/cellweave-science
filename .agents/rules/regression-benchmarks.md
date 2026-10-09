---
trigger: model_decision
description: "Use for scientific tests, golden datasets, benchmarks, pipeline releases, algorithm changes, data-size qualification, or performance regression."
---

# Scientific Regression / Golden Datasets

Every scientific change runs:
lint → type checks → unit tests → scientific regression suite → golden dataset tests → container build → SBOM → vulnerability scan → benchmark.

Golden suite includes at least:
- small smoke dataset
- human PBMC
- mouse
- multi-sample
- batch-effect
- known-treatment
- large benchmark

Evaluate:
- cells after filtering
- QC distributions
- PCA properties
- cluster consistency
- known marker recovery
- annotation accuracy
- DE direction
- pathway expectations
- runtime
- memory

Use scientifically meaningful tolerances.
Do not require byte-identical UMAP when stochastic behavior makes that inappropriate.

Data-size benchmarks:
10k, 50k, 100k, 500k, 1M cells.

Measure import time, peak RAM, CPU, GPU, artifact size, visualization responsiveness, cost.

Benchmark results determine plan limits. Do not invent limits first.

Every production pipeline version passes benchmarks.
