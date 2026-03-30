


"""
Evaluate Ultra-Light face detection models on the no-face-detection dataset.

Loads dataset from HuggingFace, runs inference with all 4 model variants,
and reports binary classification metrics (face detected vs not).

Usage:
    python evaluate.py
    python evaluate.py --net_type slim --input_size 320 --threshold 0.5
    python evaluate.py --all  # evaluate all 4 model variants
"""

import argparse
import io
import json
import os
import sys
import time

# Must run from repo root for vision imports
os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ".")

import cv2
import numpy as np
import torch
from datasets import load_dataset

from vision.ssd.config.fd_config import define_img_size


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model(net_type, input_size, device):
    """Load model and predictor. Must be called after define_img_size()."""
    define_img_size(input_size)

    # Import after define_img_size (it sets global config)
    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

    label_path = "./models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]

    if net_type == "slim":
        model_path = f"models/pretrained/version-slim-{input_size}.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=1500, device=device)
    elif net_type == "RFB":
        model_path = f"models/pretrained/version-RFB-{input_size}.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500, device=device)
    else:
        print(f"Unknown net_type: {net_type}")
        sys.exit(1)

    net.load(model_path)
    return predictor


def evaluate_model(predictor, dataset, threshold, device):
    """Run inference on all images and collect predictions."""
    results = []
    total = len(dataset)
    inference_times = []

    for idx in range(total):
        row = dataset[idx]
        pil_img = row["image"]
        label = row["label"]  # integer class label

        # Convert PIL -> numpy RGB (what the predictor expects)
        img_rgb = np.array(pil_img)
        if img_rgb.ndim == 2:  # grayscale
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        # Suppress predictor's per-call print
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        t0 = time.perf_counter()
        boxes, labels_pred, probs = predictor.predict(img_rgb, 750, threshold)
        t1 = time.perf_counter()
        sys.stdout = old_stdout
        inference_time_ms = (t1 - t0) * 1000
        inference_times.append(inference_time_ms)

        num_faces = boxes.size(0)
        max_prob = float(probs.max()) if num_faces > 0 else 0.0

        results.append({
            "idx": idx,
            "gt_label": label,
            "num_faces_detected": num_faces,
            "max_confidence": max_prob,
            "inference_time_ms": inference_time_ms,
            "task_id": row["task_id"],
            "filename": row["filename"],
        })

        if (idx + 1) % 200 == 0:
            avg_ms = np.mean(inference_times[-200:])
            print(f"  [{idx+1}/{total}] avg inference: {avg_ms:.1f}ms")

    return results, inference_times


def compute_metrics(results, label_names):
    """
    Compute binary classification metrics.
    Binary task: face visible (face_present or occluded_face) vs no face visible.

    Model predicts "face present" if it detects >= 1 face.
    Ground truth "face present" if label is face_present or occluded_face.
    """
    # Labels where a face IS visible (model should detect)
    face_visible_labels = set()
    for i, name in enumerate(label_names):
        if name in ("face_present", "occluded_face"):
            face_visible_labels.add(i)

    tp = fp = tn = fn = 0
    per_class = {name: {"total": 0, "face_detected": 0} for name in label_names}

    for r in results:
        gt_face_present = (r["gt_label"] in face_visible_labels)
        pred_face_present = (r["num_faces_detected"] > 0)
        gt_label_name = label_names[r["gt_label"]]

        per_class[gt_label_name]["total"] += 1
        if pred_face_present:
            per_class[gt_label_name]["face_detected"] += 1

        if gt_face_present and pred_face_present:
            tp += 1
        elif gt_face_present and not pred_face_present:
            fn += 1
        elif not gt_face_present and pred_face_present:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # "No face detection rate" — how often the model correctly finds NO face
    # when the ground truth says no face (true negative rate / specificity)
    no_face_total = tn + fp
    specificity = tn / no_face_total if no_face_total > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "per_class": per_class,
    }


def print_results(metrics, inference_times, net_type, input_size, threshold):
    print(f"\n{'='*60}")
    print(f"Model: {net_type}-{input_size} | Threshold: {threshold}")
    print(f"{'='*60}")

    print(f"\nBinary Classification: face_present vs no_face")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}  (of predicted faces, how many are correct)")
    print(f"  Recall:      {metrics['recall']:.4f}  (of actual faces, how many are detected)")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}  (of actual no-face, how many are caught)")

    print(f"\nConfusion Matrix:")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}  TN={metrics['tn']}")

    print(f"\nPer-class face detection rate:")
    for name, stats in metrics["per_class"].items():
        total = stats["total"]
        detected = stats["face_detected"]
        rate = detected / total if total > 0 else 0
        print(f"  {name:40s}  {detected:4d}/{total:4d}  ({rate:.1%})")

    avg_ms = np.mean(inference_times)
    p50_ms = np.percentile(inference_times, 50)
    p95_ms = np.percentile(inference_times, 95)
    print(f"\nInference Speed:")
    print(f"  Mean: {avg_ms:.1f}ms | P50: {p50_ms:.1f}ms | P95: {p95_ms:.1f}ms")
    print(f"  FPS:  {1000/avg_ms:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face detection on no-face dataset")
    parser.add_argument("--net_type", default="RFB", choices=["RFB", "slim"])
    parser.add_argument("--input_size", default=640, type=int, choices=[320, 640])
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--all", action="store_true", help="Evaluate all 4 model variants")
    parser.add_argument("--dataset", default="omkar334/no-face-detection", type=str)
    parser.add_argument("--output", default=None, type=str, help="Save results JSON to file")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("Loading dataset from HuggingFace...")
    ds = load_dataset(args.dataset, split="train", download_mode="force_redownload")

    # Fix: HF may cache stale ClassLabel schema — ensure occluded_face is included
    EXPECTED_LABELS = ["face_present", "no_face", "looking_away", "corrupted", "camera_off", "ambiguous", "occluded_face"]
    label_names = ds.features["label"].names
    if max(ds["label"]) >= len(label_names):
        label_names = EXPECTED_LABELS
    print(f"Loaded {len(ds)} images, labels: {label_names}")

    if args.all:
        variants = [
            ("slim", 320), ("slim", 640),
            ("RFB", 320), ("RFB", 640),
        ]
    else:
        variants = [(args.net_type, args.input_size)]

    all_results = {}

    for net_type, input_size in variants:
        print(f"\n--- Loading {net_type}-{input_size} ---")
        predictor = load_model(net_type, input_size, device)

        print(f"Running inference on {len(ds)} images...")
        results, inference_times = evaluate_model(predictor, ds, args.threshold, device)

        metrics = compute_metrics(results, label_names)
        print_results(metrics, inference_times, net_type, input_size, args.threshold)

        all_results[f"{net_type}-{input_size}"] = {
            "metrics": {k: v for k, v in metrics.items() if k != "per_class"},
            "per_class": metrics["per_class"],
            "per_image_results": results,
            "avg_inference_ms": float(np.mean(inference_times)),
            "threshold": args.threshold,
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
