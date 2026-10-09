---
trigger: always_on
description: "Constrains scientific job resource profiles, cost ceilings, scoped storage access, credential blast radius, and logging."
---

# Resources / Cost / Security

Logical profiles:
CPU_SMALL, CPU_MEMORY, CPU_HIGH_MEMORY, GPU_SMALL, GPU_LARGE.

Resource estimates use:
cells, genes, sparsity, dataset size, method, batch count.

Respect enforced:
estimated compute units, max compute units, max runtime, max memory class.

OOM after resource ceiling is not an infinite-retry condition.

Science container receives only:
- specific input signed URL
- specific output upload capability
- manifest fragment
- required parameters

Never receive:
- BetterAuth credentials
- Polar credentials
- full R2 master credentials
- DB admin credentials

Never execute uploaded user code.

Never log scientific matrices or expression vectors.
