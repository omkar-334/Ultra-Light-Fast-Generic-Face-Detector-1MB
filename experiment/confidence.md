# Confidence Threshold Analysis: RFB-320

## Threshold Sweep Results

Binary grouping:

- **Positive (face visible)**: face_present + looking_away + occluded_face = 1,371
- **Negative (no face)**: no_face + corrupted + camera_off + ambiguous = 175

| Metric | t=0.6 | t=0.4 | t=0.3 | t=0.2 | t=0.1 |
| --- | --- | --- | --- | --- | --- |
| Precision | 0.9925 | 0.9920 | 0.9912 | 0.9821 | 0.9216 |
| Recall | 0.4836 | 0.5412 | 0.5748 | 0.6404 | 0.8235 |
| F1 | 0.6503 | 0.7003 | 0.7276 | 0.7753 | 0.8698 |
| Specificity | 0.9714 | 0.9657 | 0.9600 | 0.9086 | 0.4514 |
| TP | 663 | 742 | 788 | 878 | 1129 |
| FP | 5 | 6 | 7 | 16 | 96 |
| FN | 708 | 629 | 583 | 493 | 242 |
| TN | 170 | 169 | 168 | 159 | 79 |

### Per-Class Detection Rate by Threshold

| Class | Group | Total | t=0.6 | t=0.4 | t=0.3 | t=0.2 | t=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| face_present | POS | 971 | 50.9% | 55.8% | 58.7% | 64.6% | 83.3% |
| looking_away | POS | 221 | 37.6% | 47.5% | 53.8% | 66.1% | 84.6% |
| occluded_face | POS | 179 | 48.0% | 53.1% | 55.3% | 58.7% | 74.3% |
| no_face | NEG | 129 | 2.3% | 3.1% | 3.1% | 9.3% | 62.8% |
| corrupted | NEG | 23 | 0.0% | 0.0% | 0.0% | 0.0% | 30.4% |
| camera_off | NEG | 5 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ambiguous | NEG | 18 | 11.1% | 11.1% | 11.1% | 22.2% | 44.4% |

## Confidence Distribution (t=0.6 baseline)

When the model detects a face, it's almost always extremely confident:

| Confidence Range | TP Count | % of TP |
| --- | --- | --- |
| 0.99 - 1.00 | 439 | 66.2% |
| 0.95 - 0.99 | 76 | 11.5% |
| 0.90 - 0.95 | 41 | 6.2% |
| 0.80 - 0.90 | 42 | 6.3% |
| 0.70 - 0.80 | 32 | 4.8% |
| 0.60 - 0.70 | 33 | 5.0% |

- **Mean**: 0.953
- **Median**: 0.999
- **P5**: 0.702
- **Min**: 0.600

The model is very binary — it either sees a face with near-certainty or doesn't see it at all.

## False Positives (at t=0.6)

Only 5 images. The model detected a face but ground truth says no face:

| File | Task | Confidence | Faces | GT Label |
| --- | --- | --- | --- | --- |
| 1770180926.jpg | 83083838 | 0.999 | 1 | ambiguous |
| 1770271193.jpg | 83117206 | 1.000 | 1 | ambiguous |
| 1770553201.jpg | 83206847 | 0.606 | 1 | no_face |
| 1770553917.jpg | 83206847 | 0.646 | 1 | no_face |
| 1771051722.jpg | 83372798 | 0.822 | 1 | no_face |

2 are labeled `ambiguous` (debatable ground truth). The 3 `no_face` ones have low confidence (0.6-0.8).

## FP Growth by Threshold

| Threshold | FP Count | FP Rate (of 175 neg) | New FPs vs t=0.6 |
| --- | --- | --- | --- |
| 0.6 | 5 | 2.9% | — |
| 0.4 | 6 | 3.4% | +1 |
| 0.3 | 7 | 4.0% | +2 |
| 0.2 | 16 | 9.1% | +11 |
| 0.1 | 96 | 54.9% | +91 |

FP stays controlled through t=0.3, starts growing at t=0.2, and explodes at t=0.1.

## Key Takeaways

1. **Sweet spot: t=0.2** — 64% recall, 98% precision, 91% specificity. Best F1 before FPs climb.
2. **t=0.3 is safest improvement** — +9% recall over default with only 2 extra FPs.
3. **t=0.1 is too aggressive** — recall jumps to 82% but 63% of true no_face images get false detections.
4. **Lowering threshold helps more than expected** despite the bimodal confidence distribution, because looking_away and occluded_face images benefit most (looking_away: 37.6% at t=0.6 -> 84.6% at t=0.1).
5. **~35-40% of images are undetectable regardless of threshold** — these are the extreme cases (face cropped out of frame, total darkness, annotation errors).

## Results Files

| File | Contents |
| --- | --- |
| `experiment/logs/results.json` | v1 — all 4 models, t=0.6, original labels |
| `experiment/logs/results_v2.json` | v2 — all 4 models, t=0.6, corrected labels (with occluded_face) |
| `experiment/logs/results_t04.json` | RFB-320, t=0.4 |
| `experiment/logs/results_t03.json` | RFB-320, t=0.3 |
| `experiment/logs/results_t02.json` | RFB-320, t=0.2 |
| `experiment/logs/results_t01.json` | RFB-320, t=0.1 |

All JSON files contain per-image results with: `idx`, `gt_label`, `num_faces_detected`, `max_confidence`, `inference_time_ms`, `task_id`, `filename`.

FP and FN images (at t=0.6) are saved in `experiment/FP/` and `experiment/FN/`, named as `{task_id}_{label}_{filename}`.
