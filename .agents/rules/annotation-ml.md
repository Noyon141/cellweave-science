---
trigger: model_decision
description: "Use for cell-type annotation, annotation providers, confidence, human review, scVI/scANVI, or GPU-routed advanced analysis."
---

# Annotation / Advanced ML

Initial organisms:
- human
- mouse

Use `AnnotationProvider` abstraction.

Automated annotation output records:
- predicted_label
- confidence
- annotation_method
- model_name
- model_version
- reference_dataset
- timestamp

If confidence is insufficient, use `Unassigned`; never fabricate certainty.

Human review is first-class:
accept, reject, replace, merge, split.

Manual correction never erases automated provenance.

scVI/scANVI:
- advanced analysis
- not required for every basic run
- GPU used only when beneficial/required
- batch-aware behavior and versions are recorded

Do not select a production annotation model/provider that the design has not selected unless the repository already contains an authorized decision.
