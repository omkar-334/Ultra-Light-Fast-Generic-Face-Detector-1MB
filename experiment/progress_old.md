# Experiment Progress

## Dataset
- **HuggingFace**: [omkar334/no-face-detection](https://huggingface.co/datasets/omkar334/no-face-detection) (private)
- **Source**: CVAT XML annotations + webcam images (640x480)
- **Size**: 1,546 images across 47 sessions
- **Labels**: face_present (1030), looking_away (255), no_face (214), corrupted (23), ambiguous (19), camera_off (5)

## Evaluation: Ultra-Light Face Detection (2026-03-30)

**Task**: Binary classification — does the model detect a face? Ground truth: `face_present` (label=0) vs all other labels.

**Device**: Apple MPS (Metal Performance Shaders) | **Threshold**: 0.6

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | Specificity | Avg Inference | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| slim-320 | 0.5925 | 0.8257 | 0.4922 | 0.6168 | 0.7926 | 5.2ms | 191.0 |
| slim-640 | 0.6028 | 0.8432 | 0.4961 | 0.6247 | 0.8159 | 6.3ms | 159.4 |
| **RFB-320** | **0.6028** | **0.8114** | **0.5262** | **0.6384** | 0.7558 | 5.4ms | 186.6 |
| RFB-640 | 0.5783 | 0.8327 | 0.4592 | 0.5920 | 0.8159 | 7.2ms | 138.7 |

### Per-Class Face Detection Rate

| Class | slim-320 | slim-640 | RFB-320 | RFB-640 |
|-------|----------|----------|---------|---------|
| face_present (n=1030) | 49.2% | 49.6% | 52.6% | 45.9% |
| no_face (n=214) | 7.9% | 4.2% | 11.2% | 5.6% |
| looking_away (n=255) | 34.5% | 32.5% | 38.8% | 31.4% |
| corrupted (n=23) | 0.0% | 0.0% | 0.0% | 0.0% |
| camera_off (n=5) | 0.0% | 0.0% | 0.0% | 0.0% |
| ambiguous (n=19) | 10.5% | 15.8% | 15.8% | 15.8% |

### Key Observations

1. **Low recall across all models** (~46-53%): The models fail to detect faces in roughly half the `face_present` frames. This is likely because these are webcam images with varying lighting, angles, and partial occlusion — conditions that differ from the WIDER FACE training data.

2. **Best overall: RFB-320** — highest F1 (0.6384) and recall (52.6%), though slightly lower specificity. Best balance of detecting real faces while rejecting no-face frames.

3. **Best specificity: slim-640 / RFB-640** (0.8159) — the 640 variants are slightly better at correctly identifying "no face" frames, but at the cost of recall.

4. **False positive analysis**: Even for `no_face` ground truth, models detect faces 4-11% of the time, suggesting some frames may have partial faces visible or annotation disagreements.

5. **Speed**: All variants run at 139-191 FPS on MPS — speed is not a bottleneck.

## Scripts
- `experiment/scripts/create_hf_dataset.py` — Parse CVAT XML + images, upload to HuggingFace
- `experiment/scripts/evaluate.py` — Run face detection evaluation on all model variants

## Logs
- `experiment/logs/evaluate.log` — Full evaluation console output
- `experiment/logs/results.json` — Detailed per-image results with inference times
