"""Label decoding example for JIGSAWS PyTorch Dataset.

This example shows how to:
1. Use the LabelEncoder to decode model predictions
2. Decode INTEGER and ONE_HOT encoded labels
3. Handle batch decoding for evaluation
"""

import numpy as np
import torch

from jigsaws_pytorch_dataset import KinematicsDataset, LabelEncoder
from jigsaws_pytorch_dataset.options import (
    KinematicsSamplingMode,
    LabelsFormat,
    UnlabeledDataPolicy,
)


def main():
    dataset_path = "./dataset/Suturing/"

    # --- Step 1: Create dataset and get encoder ---
    print("=" * 50)
    print("Step 1: Create dataset and get label encoder")
    print("=" * 50)

    dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    encoder = dataset.get_label_encoder()
    print(f"Number of classes: {encoder.num_classes}")
    print(f"Class labels: {encoder.classes}")
    print(f"Gesture map: {encoder.gesture_map}")
    print()

    # --- Step 2: Simulate model predictions ---
    print("=" * 50)
    print("Step 2: Decode simulated model predictions")
    print("=" * 50)

    # Simulate a batch of integer predictions
    predictions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"Predictions (integers): {predictions.tolist()}")

    # Decode to gesture names
    decoded = encoder.decode(predictions)
    print(f"Decoded labels: {decoded.tolist()}")
    print()

    # --- Step 3: Decode single predictions ---
    print("=" * 50)
    print("Step 3: Decode single predictions")
    print("=" * 50)

    for i in range(min(5, encoder.num_classes)):
        gesture = encoder.decode_single(i)
        print(f"Class {i} -> {gesture}")
    print()

    # --- Step 4: Decode from softmax/logits ---
    print("=" * 50)
    print("Step 4: Decode from model output (logits)")
    print("=" * 50)

    # Simulate logits output from a model
    batch_size = 4
    logits = torch.randn(batch_size, encoder.num_classes)
    print(f"Logits shape: {logits.shape}")

    # Get predicted classes
    predicted_classes = logits.argmax(dim=-1)
    print(f"Predicted classes: {predicted_classes.tolist()}")

    # Decode
    decoded = encoder.decode(predicted_classes)
    print(f"Decoded gestures: {decoded.tolist()}")
    print()

    # --- Step 5: Working with ONE_HOT labels ---
    print("=" * 50)
    print("Step 5: Working with ONE_HOT labels")
    print("=" * 50)

    one_hot_dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.ONE_HOT,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    oh_encoder = one_hot_dataset.get_label_encoder()

    # Get a sample with one-hot label
    features, one_hot_label = one_hot_dataset[0]
    print(f"One-hot label shape: {one_hot_label.shape}")
    print(f"One-hot label: {one_hot_label}")

    # Decode one-hot directly
    decoded = oh_encoder.decode(one_hot_label.unsqueeze(0))
    print(f"Decoded from one-hot: {decoded[0]}")
    print()

    # --- Step 6: Compute accuracy with decoding ---
    print("=" * 50)
    print("Step 6: Compute accuracy with decoded labels")
    print("=" * 50)

    # Simulate ground truth and predictions
    num_samples = 100
    ground_truth = torch.randint(0, encoder.num_classes, (num_samples,))
    # Simulate predictions with some errors
    predictions = ground_truth.clone()
    noise_indices = torch.randperm(num_samples)[:20]
    predictions[noise_indices] = torch.randint(0, encoder.num_classes, (20,))

    # Decode both
    gt_gestures = encoder.decode(ground_truth)
    pred_gestures = encoder.decode(predictions)

    # Compute accuracy
    correct = (gt_gestures == pred_gestures).sum()
    accuracy = correct / num_samples * 100
    print(f"Accuracy: {accuracy:.1f}%")

    # Per-class accuracy
    print("\nPer-class breakdown:")
    for i, gesture in enumerate(encoder.classes):
        mask = ground_truth == i
        if mask.sum() > 0:
            class_correct = (predictions[mask] == i).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total * 100
            print(f"  {gesture}: {class_acc:.1f}% ({class_correct}/{class_total})")
    print()

    # --- Step 7: Access grouping (for Access paper) ---
    print("=" * 50)
    print("Step 7: Working with gesture groupings")
    print("=" * 50)

    grouped_dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        gesture_grouping="access_suturing",
    )

    grouped_encoder = grouped_dataset.get_label_encoder()
    print(f"Grouped classes: {grouped_encoder.classes}")
    print(f"Number of groups: {grouped_encoder.num_classes}")

    # Decode grouped predictions
    grouped_preds = torch.tensor([0, 1, 2, 3, 4, 5])
    decoded_grouped = grouped_encoder.decode(grouped_preds)
    print(f"Decoded grouped: {decoded_grouped.tolist()}")


if __name__ == "__main__":
    main()
