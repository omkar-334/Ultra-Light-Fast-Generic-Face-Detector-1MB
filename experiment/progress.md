# Experiment Progress

## Dataset
- **HuggingFace**: [omkar334/no-face-detection](https://huggingface.co/datasets/omkar334/no-face-detection) (private)
- **Source**: CVAT XML annotations + webcam images (640x480)
- **Size**: 1,546 images across 47 sessions

### Label Distribution (v2 — with occluded_face)

| Label | Count | Description |
|-------|-------|-------------|
| face_present | 971 | Face clearly visible |
| looking_away | 221 | Looking to the side or down |
| occluded_face | 179 | Face partially visible but annotated as "no face" — corrected |
| no_face | 129 | No face visible at all |
| corrupted | 23 | Image corrupted or missing |
| ambiguous | 18 | Unclear classification |
| camera_off | 5 | Camera is off |

## Evaluation v2: Ultra-Light Face Detection (2026-03-30)

**Task**: Binary classification — face visible (`face_present` + `occluded_face`) vs no face visible (all other labels).

**Device**: Apple MPS | **Threshold**: 0.6

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | Specificity | Avg Inference | FPS |
|-------|----------|-----------|--------|----|-------------|--------------|-----|
| slim-320 | 0.5524 | 0.8730 | 0.4661 | 0.6077 | 0.8030 | 4.7ms | 214.0 |
| slim-640 | 0.5537 | 0.8795 | 0.4635 | 0.6071 | 0.8157 | 6.3ms | 158.1 |
| **RFB-320** | **0.5744** | **0.8683** | **0.5043** | **0.6381** | 0.7778 | 5.3ms | 190.3 |
| RFB-640 | 0.5291 | 0.8715 | 0.4304 | 0.5763 | 0.8157 | 7.1ms | 140.0 |

### Per-Class Face Detection Rate

| Class | slim-320 | slim-640 | RFB-320 | RFB-640 |
|-------|----------|----------|---------|---------|
| face_present (n=971) | 48.1% | 47.9% | 50.9% | 43.9% |
| occluded_face (n=179) | 38.5% | 38.0% | 48.0% | 38.5% |
| no_face (n=129) | 1.6% | 0.8% | 2.3% | 1.6% |
| looking_away (n=221) | 33.5% | 31.7% | 37.6% | 31.2% |
| corrupted (n=23) | 0.0% | 0.0% | 0.0% | 0.0% |
| camera_off (n=5) | 0.0% | 0.0% | 0.0% | 0.0% |
| ambiguous (n=18) | 11.1% | 11.1% | 11.1% | 11.1% |

### Key Observations

1. **Precision is high (~87%)** — when the model says "face", it's almost always right.
2. **Recall is low (~43-50%)** — the model misses faces in roughly half the images. Root causes (from visual inspection):
   - Low-angle webcams cropping faces at bottom of frame
   - Poor lighting / low contrast (dark skin + dim room)
   - Motion blur destroying edge features
   - Failures cluster by session (specific users with bad camera setups), not random
3. **Best overall: RFB-320** — highest F1 (0.6381) and recall (50.4%), good speed (190 FPS).
4. **no_face false positive rate is very low** (0.8-2.3%) — model rarely hallucinates faces.
5. **occluded_face detection at ~38-48%** — model detects roughly half of partially occluded faces.
6. **Correcting mislabeled data** (v1 → v2) improved precision from ~0.81-0.84 to ~0.87-0.88 by removing false positives that were actually correct detections on mislabeled images.

### Comparison: v1 (before label correction) vs v2

| Model | v1 F1 | v2 F1 | v1 Precision | v2 Precision | v1 Recall | v2 Recall |
|-------|-------|-------|-------------|-------------|----------|----------|
| slim-320 | 0.6168 | 0.6077 | 0.8257 | 0.8730 | 0.4922 | 0.4661 |
| slim-640 | 0.6247 | 0.6071 | 0.8432 | 0.8795 | 0.4961 | 0.4635 |
| RFB-320 | 0.6384 | 0.6381 | 0.8114 | 0.8683 | 0.5262 | 0.5043 |
| RFB-640 | 0.5920 | 0.5763 | 0.8327 | 0.8715 | 0.4592 | 0.4304 |

Precision improved ~5% across the board. Recall dropped slightly because the positive class is now larger (1150 vs 1030) with the harder `occluded_face` examples included.

## Scripts
- `experiment/scripts/create_hf_dataset.py` — Parse CVAT XML + images, upload to HuggingFace
- `experiment/scripts/relabel_and_push.py` — Relabel specific images as occluded_face, re-push
- `experiment/scripts/evaluate.py` — Run face detection evaluation on all model variants

## Logs
- `experiment/logs/evaluate.log` — v1 evaluation (original labels)
- `experiment/logs/results.json` — v1 per-image results
- `experiment/logs/evaluate_v2.log` — v2 evaluation (corrected labels)
- `experiment/logs/results_v2.json` — v2 per-image results with inference times
- `experiment/logs/relabel.log` — Relabeling output
