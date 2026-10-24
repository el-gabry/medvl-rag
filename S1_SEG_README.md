# S1-SEG-01 — Anatomical lung segmentation implementation (research-only)

**Reference branch:** `feat/embedding-extraction-cli`
**Reference HEAD:** `5dd6685fb287d53820e73977cf7cb4c583794b88`
**Scope:** new files only. No modification to existing encoder/retrieval/E06 modules.
**Implementation checkpoint (2026-10-24, before first S1 commit):** S1 source files were applied locally in WSL and six targeted segmentation tests passed. Real Montgomery validation and Kaggle S1 training had not been performed at this checkpoint. Git history is authoritative for subsequent commits.

## What is implemented

- Strict local Montgomery CXR ↔ `ManualMask/leftMask` ↔ `ManualMask/rightMask` pairing (no automatic image downloads).
- Fixed, case-key-disjoint SHA-256 split (70/15/15), explicitly **not a verified patient-level split** without clinical identity data.
- Pixel-space-preserving image and mask letterbox transforms; reverse mapping to original resolution.
- A small, two-channel U-Net baseline, no pretrained checkpoint or claimed clinical validation.
- Dice/IoU/area bias and QA flags for empty and overlapping lung predictions.
- A frozen, *train-only* average-anatomical-mask prior as the mandatory non-learned comparator.
- One local dataset/model smoke and a bounded, single final held-out evaluation after validation-only model selection.
- Aggregate JSON metrics, SHA-256 of checkpoint, and explicit limitations (no case-level outputs published).

## Data layout (local or permitted private compute only)

```text
/path/to/MontgomerySet/
  CXR_png/MCUCXR_0001_0.png
  ManualMask/leftMask/MCUCXR_0001_0.png
  ManualMask/rightMask/MCUCXR_0001_0.png
  ...
```

Official source: https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html

Review the dataset's exact research, redistribution and cloud-processing permissions **before** placing the data on Kaggle or publishing examples. No patient images or masks are included in this patch.

## Current execution workflow

1. WSL is the source of truth for reusable code and tests.
2. The nine S1 files are already present locally; do not
   reapply the patch or recreate the project structure.
3. Six targeted segmentation tests passed previously.
   Do not rerun them without a relevant code change.
4. Review source changes before authorizing commit/push.
5. Use the existing private Kaggle Master Notebook for
   real-data smoke, GPU training and held-out evaluation.
6. Verify data access and cloud-processing permissions
   before mounting Montgomery in Kaggle.
7. Record aggregate metrics and experiment provenance
   without exposing restricted medical data.
8. Close S1 based on the milestone gate below, then move
   to independent finding detection.

## Local evidence in the patch-building sandbox

- 6 focused synthetic *software contract* tests passed.
- One full **synthetic-data** train/validation/test software pipeline completed and wrote a checkpoint + aggregate metrics.
- The original sandbox did not evaluate real Montgomery images or Kaggle GPU. Subsequently, the user WSL targeted S1 suite passed 6/6 tests. The older 159-test project regression is separate evidence and was not rerun for S1.
- Synthetic mask Dice numbers **cannot** be reported as medical model performance.

## Milestone gate (not yet passed)

S1-SEG can close only after: a legal, real paired Montgomery smoke; fixed split confirmation; legitimate held-out Dice/IoU; comparison to train-only prior; aggregate failure/latency metrics; provenance; one allowed overlay/failure example; and a decision `GO` / `CONDITIONAL` / `DROP`. A successful S1 milestone does not establish accuracy for Cardiomegaly or Pleural Effusion. Those belong to downstream finding/verification experiments, always including an unsegmented full-image baseline.
