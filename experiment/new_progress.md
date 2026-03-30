# Evaluation Progress: Ultra-Light Face Detection

## Dataset

- **HuggingFace**: [omkar334/no-face-detection](https://huggingface.co/datasets/omkar334/no-face-detection) (private)
- **Size**: 1,546 images, 640x480, 47 webcam sessions
- **Labels**: face_present (971), looking_away (221), occluded_face (179), no_face (129), corrupted (23), ambiguous (18), camera_off (5)

## Binary Grouping

- **Positive (face visible)**: face_present + looking_away + occluded_face = 1,371
- **Negative (no face)**: no_face + corrupted + camera_off + ambiguous = 175

## Results (RFB-320, best model, threshold=0.6)

| Metric | Value |
| --- | --- |
| Accuracy | 0.5388 |
| Precision | 0.9925 |
| Recall | 0.4836 |
| F1 | 0.6503 |
| Specificity | 0.9714 |
| TP | 663 |
| FP | 5 |
| FN | 708 |
| TN | 170 |

### All Model Variants

| Model | Accuracy | Precision | Recall | F1 | Specificity | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| slim-320 | 0.5052 | 0.9935 | 0.4449 | 0.6146 | 0.9771 | 214 |
| slim-640 | 0.5013 | 0.9950 | 0.4398 | 0.6100 | 0.9829 | 158 |
| **RFB-320** | **0.5388** | **0.9925** | **0.4836** | **0.6503** | 0.9714 | 190 |
| RFB-640 | 0.4754 | 0.9930 | 0.4114 | 0.5817 | 0.9771 | 140 |

### Per-Class Detection Rate (RFB-320)

| Class | Group | Detected | Total | Rate |
| --- | --- | --- | --- | --- |
| face_present | POS | 494 | 971 | 50.9% |
| occluded_face | POS | 86 | 179 | 48.0% |
| looking_away | POS | 83 | 221 | 37.6% |
| no_face | NEG | 3 | 129 | 2.3% |
| corrupted | NEG | 0 | 23 | 0.0% |
| camera_off | NEG | 0 | 5 | 0.0% |
| ambiguous | NEG | 2 | 18 | 11.1% |

## Why Is Recall So Low? (708 false negatives analyzed)

### Two categories of failure

**55% of FN (391 images) come from 10 high-miss-rate tasks (>70% miss rate)**. These are sessions where the model fails on almost every frame from a specific user. The remaining 45% (317 images) are scattered across moderate-miss tasks.

### High-miss-rate tasks (>70% miss) — 10 tasks, 391 FN

| Task | Missed/Total | Miss Rate | Root Cause |
| --- | --- | --- | --- |
| 83102778 | 110/149 | 74% | Dark skin + low contrast + many frames where person covers face with hand or is cropped at bottom — many should be relabeled as occluded_face |
| 83113902 | 71/77 | 92% | Face at extreme bottom of frame, mostly hair visible, very low angle webcam |
| 82467586 | 52/54 | 96% | Extremely dark/blurry, low-quality webcam, motion blur throughout |
| 83470544 | 50/51 | 98% | Face in bottom third, hand covering mouth, low-angle shot |
| 82978427 | 24/27 | 89% | Extreme close-up, face fills entire frame but is blurry + dark, hair covering face |
| 82962527 | 28/33 | 85% | Low-angle webcam, face cropped at bottom, glasses with glare |
| 83108338 | 24/33 | 73% | Similar low-angle + dark lighting pattern |
| 82371601 | 15/16 | 94% | Almost all occluded_face — only forehead/glasses visible |
| 83117206 | 14/18 | 78% | Looking away + occluded frames |
| 82498193 | 3/4 | 75% | Small sample, same dark/cropped pattern |

**Visual patterns in these tasks:**

1. **Face cropped at bottom of frame** (low-angle webcam) — only top of head + forehead visible. The SSD model needs to see nose/mouth/eyes to fire a detection. Example: task 83113902, 83470544.
2. **Extreme low light + dark skin** — face blends into dark background, no edge contrast. Example: task 82467586, 83449196.
3. **Hand/object covering face** — labeled as face_present but face isn't actually detectable. Example: task 83102778 (1770217596.jpg — hand covering entire face). **Many of these are annotation errors** that should be occluded_face or no_face.
4. **Motion blur** — person moving, face is a smear. Example: task 82978427.

### Moderate-miss tasks (30-70% miss) — the real model failures

| Task | Missed/Total | Miss Rate | Notes |
| --- | --- | --- | --- |
| 83449196 | 59/91 | 65% | Very dark room, only glasses glare visible |
| 83211091 | 34/74 | 46% | Looking up/away at angle, face visible but not frontal |
| 83086948 | 23/102 | 23% | Mixed — some dark frames, some good |
| 83145319 | 22/60 | 37% | Intermittent lighting issues |
| 83342430 | 22/49 | 45% | Occluded + dark |
| 83210967 | 21/65 | 32% | Face at angle |
| 83414507 | 21/40 | 52% | Mixed dark + occluded |
| 82936059 | 18/30 | 60% | Headphones, face clearly visible — genuine model miss |
| 83276172 | 11/19 | 58% | Dark + angled |
| 82805889 | 9/13 | 69% | Dark room |

**Notable genuine failures:**
- Task 82936059: Person wearing headphones, looking straight at camera, decent lighting — model should detect this but doesn't (60% miss rate)
- Task 83211091: Person with glasses looking slightly upward — face clearly visible but at non-frontal angle (46% miss rate)

### Summary: What's really happening

| Category | Est. FN Count | % of 708 FN |
| --- | --- | --- |
| Annotation errors (face not actually visible, should be occluded/no_face) | ~150-200 | ~21-28% |
| Low-angle webcam (face cropped at bottom) | ~150-200 | ~21-28% |
| Extreme low light / dark skin + dark room | ~150-200 | ~21-28% |
| Genuine model failures (face clearly visible, good conditions) | ~100-150 | ~14-21% |

**If we exclude the ~150-200 annotation errors, true recall would be closer to 55-60%.**

The remaining ~40% miss rate is split between:
- Conditions the model was never trained for (low-angle cropped webcam shots)
- Lighting conditions where any lightweight SSD would struggle
- A smaller set of genuine failures on reasonably visible faces

### Recommendations

1. **Fix annotations**: ~150-200 images labeled face_present actually have no detectable face (hand covering, extreme crop). Relabeling these would immediately improve measured recall by ~10%.
2. **Lower threshold**: Try threshold=0.3-0.4 to see if the model detects faces at lower confidence in dark/angled shots — trading specificity for recall.
3. **Different model**: Ultra-Light was designed for speed on WIDER FACE, not robustness on degraded webcam footage. A model fine-tuned on similar webcam data, or a heavier model (RetinaFace, SCRFD), would likely perform significantly better on this distribution.
