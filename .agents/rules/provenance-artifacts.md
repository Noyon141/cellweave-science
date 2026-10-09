---
trigger: always_on
description: "Requires immutable scientific provenance, artifact lineage, manifests, checksums, version pinning, and reproducible exports."
---

# Provenance / Artifact Lineage

Every completed run creates immutable Run Manifest.

Required provenance includes:
- pipeline version
- step versions
- container digest
- input checksum
- parameters
- random seeds
- species
- reference genome/build when available
- annotation model/version
- reference database versions
- software dependency versions
- outputs
- timestamps

Pipeline versions cannot be edited after production use.

Artifacts record:
generator, inputs, step, version, checksum, storage metadata.

Lineage reconstructs parent → child derivation.

Visualization is never the only copy of numerical derived data.

Reproducibility Bundle supports:
run-manifest.json, parameters.json, software-versions.json, artifact-manifest.json, results/, README.

Reports are artifacts, not truth stores.
