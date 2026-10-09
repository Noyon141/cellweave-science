---
trigger: always_on
description: "Preserves CellWeave's scientific compute boundary and prevents SaaS/backend responsibilities from entering the Python engine."
---

# Science Boundary

This repository computes science and artifacts.

Never:
- implement customer authentication
- implement organization/project authorization database logic
- store business billing truth
- expose the customer-facing REST API
- depend on BetterAuth or Polar credentials
- connect with primary DB admin credentials
- make Modal a domain-wide direct dependency outside its adapter/app boundary
- invent cross-repository payload schemas instead of generated Pydantic models

Inputs/outputs are explicit contracts and scoped storage capabilities.

Large scientific data stays in object storage/artifacts, not operational Postgres.
