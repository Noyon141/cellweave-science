---
trigger: model_decision
description: "Use for local scientific execution, Docker/dev environments, or switching LOCAL versus MODAL compute adapters."
---

# Local Science Execution

The scientific pipeline must run locally on a small test dataset without Modal.

Execution mode is selected through provider adapters:
- LOCAL
- MODAL

Local execution does not bypass:
- canonical AnnData behavior
- provenance
- deterministic seeds
- validation
- resource awareness
- tests

Do not fork scientific logic into separate local and production implementations.
