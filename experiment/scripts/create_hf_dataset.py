"""
Convert CVAT XML annotations + images into a HuggingFace dataset and upload.

Structure:
  annotated_no_face_data/<task_id>/<image>.jpg
  no_face_annotations/<task_id>.xml

Labels (frame-level tags from CVAT):
  - no_face: "No Face or Partial Face Missing (For Cheating)"
  - looking_away: "Looking Away Sharply (Looking to the Side or Down, Gaze if Off)"
  - corrupted: "Image Corrupted or Missing"
  - camera_off: "Camera is Off"
  - ambiguous: "Ambiguous"
  - face_present: no tag (face is visible)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from datasets import Dataset, Features, Value, Image, ClassLabel
from PIL import Image as PILImage


LABEL_MAP = {
    "No Face or Partial Face Missing (For Cheating)": "no_face",
    "Looking Away Sharply (Looking to the Side or Down, Gaze if Off)": "looking_away",
    "Image Corrupted or Missing": "corrupted",
    "Camera is Off": "camera_off",
    "Ambiguous": "ambiguous",
}

LABEL_NAMES = ["face_present", "no_face", "looking_away", "corrupted", "camera_off", "ambiguous"]

IMG_DIR = Path("annotated_no_face_data")
ANN_DIR = Path("no_face_annotations")


def parse_annotations():
    records = []
    for xml_file in sorted(ANN_DIR.glob("*.xml")):
        task_id = xml_file.stem
        tree = ET.parse(xml_file)

        for image_el in tree.findall(".//image"):
            filename = image_el.get("name")
            width = int(image_el.get("width"))
            height = int(image_el.get("height"))

            tags = [t.get("label") for t in image_el.findall("tag")]
            tags = [t for t in tags if t]  # filter empty

            # Pick the most specific label (priority order)
            label = "face_present"
            for tag in tags:
                mapped = LABEL_MAP.get(tag)
                if mapped:
                    label = mapped
                    break

            img_path = IMG_DIR / task_id / filename
            if not img_path.exists():
                print(f"WARN: missing image {img_path}")
                continue

            records.append({
                "image": str(img_path),
                "label": label,
                "task_id": task_id,
                "filename": filename,
                "width": width,
                "height": height,
            })

    return records


def main():
    print("Parsing annotations...")
    records = parse_annotations()
    print(f"Found {len(records)} images")

    # Print label distribution
    from collections import Counter
    counts = Counter(r["label"] for r in records)
    for label in LABEL_NAMES:
        print(f"  {label}: {counts.get(label, 0)}")

    # Build dataset
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=LABEL_NAMES),
        "task_id": Value("string"),
        "filename": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
    })

    ds = Dataset.from_list(records, features=features)
    print(f"\nDataset: {ds}")
    print(ds[0])

    # Verify a few images load correctly
    print("\nVerifying image loading...")
    for i in [0, len(ds) // 2, len(ds) - 1]:
        img = ds[i]["image"]
        assert img is not None, f"Failed to load image at index {i}"
        print(f"  [{i}] {ds[i]['filename']} -> {img.size}, label={ds[i]['label']}")

    print("\nLocal validation passed. Uploading to HuggingFace...")
    ds.push_to_hub(
        "omkar334/no-face-detection",
        private=True,
    )
    print("Done! Dataset uploaded.")


if __name__ == "__main__":
    main()
