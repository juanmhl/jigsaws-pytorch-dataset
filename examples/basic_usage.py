"""Basic usage example for JIGSAWS PyTorch Dataset.

This example shows how to:
1. Create a KinematicsDataset in different modes
2. Iterate over the data
3. Use with PyTorch DataLoader
"""

import torch
from torch.utils.data import DataLoader

from jigsaws_pytorch_dataset import KinematicsDataset
from jigsaws_pytorch_dataset.options import (
    KinematicsSamplingMode,
    LabelsFormat,
    UnlabeledDataPolicy,
    Users,
    Trials,
)


def main():
    # Path to your JIGSAWS dataset
    dataset_path = "./dataset/Suturing/"

    # --- Example 1: Sequence mode with RAW labels ---
    print("=" * 50)
    print("Example 1: SEQUENCE mode with RAW labels")
    print("=" * 50)

    dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.RAW,
    )

    print(f"Dataset length: {len(dataset)}")
    features, labels, length = dataset[0]
    print(f"First sequence shape: {features.shape}")
    print(f"First sequence length: {length}")
    print(f"Labels type: {type(labels)}")
    print(f"First few labels: {labels[:5]}")
    print()

    # --- Example 2: Sample mode with INTEGER labels ---
    print("=" * 50)
    print("Example 2: SAMPLE mode with INTEGER labels")
    print("=" * 50)

    dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    features, label = dataset[0]
    print(f"Sample shape: {features.shape}")
    print(f"Label: {label}")
    print()

    # --- Example 3: Filter by users and trials ---
    print("=" * 50)
    print("Example 3: Filter by users and trials")
    print("=" * 50)

    # Only use users B and C, trials 1-3
    dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        users_set=(Users.B, Users.C),
        trials_set=(Trials.T1, Trials.T2, Trials.T3),
    )

    print(f"Filtered dataset length: {len(dataset)}")
    print()

    # --- Example 4: Using with DataLoader ---
    print("=" * 50)
    print("Example 4: Using with DataLoader (SAMPLE mode)")
    print("=" * 50)

    dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Get one batch
    batch_features, batch_labels = next(iter(dataloader))
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print()

    # --- Example 5: Access gesture grouping ---
    print("=" * 50)
    print("Example 5: Access gesture grouping")
    print("=" * 50)

    dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        gesture_grouping="access_suturing",  # Groups G1-G11 into Q0-Q5
    )

    print(f"Number of classes with grouping: {dataset.num_classes}")
    encoder = dataset.get_label_encoder()
    print(f"Class labels: {encoder.classes}")


if __name__ == "__main__":
    main()
