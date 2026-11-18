# Software validation — 2026-11-11

## Completed for this adapted version

- Read the supplied Excel and the 557,318,434-byte ZIP. Matched and decoded all 1,210 listed DFU2 images; every image is 512×512.
- Imported the original ZIP through the new CLI. Verified the output CSVs and manifest reload successfully with train/val/test counts of 700/174/186 and exact pixel hashes disjoint across splits.
- Excluded 122 unknown labels and 28 duplicate copies with known labels, retaining original split membership. Preserved original source files. The exported reports record the rules and exclusions.
- All 11 unit tests passed: Excel BOM and garbled outer path handling; val/validation mapping; unknown and invalid labels; duplicate precedence and conflicting-label rejection; complete patient-map requirements, leading zeros and cross-split patient detection; ZIP selection excluding masks/other datasets; no invented patient IDs; manifest consistency; path traversal rejection; plus the previous patient-split, duplicate, threshold, letterbox, checkpoint and frozen-BatchNorm checks.
- Four images from the real **training split only**, including both classes, passed the default 224×224 EfficientNet-B0 forward and BCE backward computation with official ImageNet weights. This was an input/gradient smoke check, not a training run; no real-data checkpoint was retained.
- A separate artificial seven-image Excel/ZIP fixture exercised import, a frozen-head epoch and a fine-tuning epoch, validation-only threshold selection, held-out artificial evaluation, checkpoint reload and single-image inference. Unknown-patient metadata persisted end to end; evaluation correctly returned null for patient counts and patient-cluster confidence intervals.
- Streamlit AppTest loaded the synthetic checkpoint and rendered the notice that patient separation has not been verified.

## Environment

Linux, Python 3.12.14; torch 2.5.1+cpu, torchvision 0.20.1+cpu;
NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, Pillow 11.0.0,
matplotlib 3.9.2, Streamlit 1.41.1, openpyxl 3.1.5.

The user's installation guide targets an isolated Python 3.10 environment in WSL. This turn did not validate Python 3.10, the user's GPU, CUDA execution or their local installation.

## Not established

- No full training on the user's real data and no real test-set model inference were performed.
- No referral performance, patient-independent generalization, calibration, clinical usefulness or deployment readiness is established.
- Patient identities and near-duplicate images have not been verified. The supplied workbook has no patient identifiers or textual definition of the 0/1 labels.
- Synthetic metrics are software checks only and are not reported as research findings.


