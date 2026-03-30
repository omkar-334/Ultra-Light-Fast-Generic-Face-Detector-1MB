"""
Relabel specific images as 'occluded_face' and re-push dataset to HuggingFace.

Adds a new label class 'occluded_face' and relabels:
- Specific filenames (from manual review)
- All images from tasks 83208691, 83342430, 83396091, 83414507
"""

import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ".")

from datasets import load_dataset, ClassLabel, Features, Value, Image

DATASET_ID = "omkar334/no-face-detection"

# Individual filenames to relabel (normalize: strip extensions, re-add .jpg)
OCCLUDED_FILENAMES = {
    "1767376014.jpg",
    "1767376019.jpg",
    "1767376024.jpg",
    "1767943788.jpg",
    "1767943848.jpg",
    "1767943908.jpg",
    "1767944029.jpg",
    "1767944568.jpg",
    "1767944628.jpg",
    "1767944688.jpg",
    "1767944748.jpg",
    "1767944808.jpg",
    "1767944868.jpg",
    "1767944929.jpg",
    "1767944988.jpg",
    "1767945048.jpg",
    "1768044074.jpg",
    "1768044079.jpg",
    "1768044085.jpg",
    "1768044094.jpg",
    "1768295178.jpg",
    "1768295183.jpg",
    "1768295203.jpg",
    "1768295258.jpg",
    "1768295263.jpg",
    "1768295278.jpg",
    "1768295308.jpg",
    "1768295313.jpg",
    "1769269168.jpg",
    "1769691450.jpg",
    "1769694406.jpg",
    "1770221139.jpg",
    "1770221149.jpg",
    "1770221159.jpg",
    "1770221289.jpg",
    "1770253812.jpg",
    "1770254388.jpg",
    "1770270548.jpg",
    "1770271219.jpg",
    "1770271224.jpg",
    "1770271229.jpg",
    "1770329809.jpg",
    "1770502430.jpg",
    "1770502437.jpg",
    "1770559451.jpg",
    "1770559452.jpg",
    "1770559466.jpg",
    "1770559467.jpg",
    "1770652872.jpg",
    "1770884609.jpg",
    "1770884614.jpg",
    "1771326959.jpg",
    "1771327324.jpg",
    "1771327329.jpg",
    "1771328429.jpg",
}

# All images from these tasks get relabeled
OCCLUDED_TASKS = {"83208691", "83342430", "83396091", "83414507"}

# New label set (added occluded_face)
NEW_LABEL_NAMES = [
    "face_present",
    "no_face",
    "looking_away",
    "corrupted",
    "camera_off",
    "ambiguous",
    "occluded_face",
]

OCCLUDED_FACE_IDX = NEW_LABEL_NAMES.index("occluded_face")


def main():
    print("Loading dataset...")
    ds = load_dataset(DATASET_ID, split="train")
    old_labels = ds.features["label"]
    print(f"Loaded {len(ds)} images, old labels: {old_labels.names}")

    # Build old label index -> name mapping
    old_names = old_labels.names

    def relabel(example):
        filename = example["filename"]
        task_id = example["task_id"]
        old_label_name = old_names[example["label"]]

        if filename in OCCLUDED_FILENAMES or task_id in OCCLUDED_TASKS:
            return {"label": OCCLUDED_FACE_IDX}
        else:
            # Map old label name to new index (same position)
            return {"label": NEW_LABEL_NAMES.index(old_label_name)}

    # Cast label to int first so we can remap freely
    ds = ds.cast_column("label", Value("int64"))
    ds = ds.map(relabel)

    # Cast back to ClassLabel with new names
    new_features = ds.features.copy()
    new_features["label"] = ClassLabel(names=NEW_LABEL_NAMES)
    ds = ds.cast(new_features)

    # Print new distribution
    from collections import Counter
    counts = Counter()
    for row in ds:
        counts[row["label"]] += 1
    print(f"\nNew label distribution:")
    for i, name in enumerate(NEW_LABEL_NAMES):
        print(f"  {name}: {counts.get(i, 0)}")
    print(f"  Total: {sum(counts.values())}")

    print("\nUploading to HuggingFace...")
    ds.push_to_hub(DATASET_ID, private=True)
    print("Done!")


if __name__ == "__main__":
    main()
