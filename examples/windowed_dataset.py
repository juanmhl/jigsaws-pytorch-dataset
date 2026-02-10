"""WindowedDataset example for JIGSAWS PyTorch Dataset.

This example shows how to:
1. Create a WindowedDataset from a sequence dataset
2. Use different label strategies (last, middle, majority)
3. Configure window size and stride
"""

import torch
from torch.utils.data import DataLoader

from jigsaws_pytorch_dataset import KinematicsDataset, WindowedDataset
from jigsaws_pytorch_dataset.options import (
    KinematicsSamplingMode,
    LabelsFormat,
    UnlabeledDataPolicy,
    Users,
    Trials,
)


def main():
    dataset_path = "./dataset/Suturing/"

    # --- Step 1: Create base sequence dataset ---
    print("=" * 50)
    print("Step 1: Create base SEQUENCE dataset")
    print("=" * 50)

    base_dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    print(f"Base dataset length: {len(base_dataset)} sequences")
    features, labels, length = base_dataset[0]
    print(f"First sequence: {length} frames, {features.shape[1]} features")
    print(f"Number of classes: {base_dataset.num_classes}")
    print()

    # --- Step 2: Create windowed dataset with "last" label strategy ---
    print("=" * 50)
    print("Step 2: WindowedDataset with 'last' label strategy")
    print("=" * 50)

    windowed = WindowedDataset(
        dataset=base_dataset,
        window_size=32,
        stride=1,
        label_strategy="last",  # Use label of last frame (default)
    )

    print(f"Windowed dataset length: {len(windowed)} windows")
    features, label = windowed[0]
    print(f"Window shape: {features.shape}")
    print(f"Label (last frame): {label}")
    print()

    # --- Step 3: Compare different label strategies ---
    print("=" * 50)
    print("Step 3: Compare label strategies")
    print("=" * 50)

    for strategy in ["last", "middle", "majority"]:
        windowed = WindowedDataset(
            dataset=base_dataset,
            window_size=32,
            stride=1,
            label_strategy=strategy,
        )
        _, label = windowed[0]
        print(f"Strategy '{strategy}': label = {label}")
    print()

    # --- Step 4: Different window sizes and strides ---
    print("=" * 50)
    print("Step 4: Different window sizes and strides")
    print("=" * 50)

    configs = [
        (16, 1, "Small window, stride 1"),
        (32, 1, "Medium window, stride 1"),
        (64, 1, "Large window, stride 1"),
        (32, 16, "Medium window, stride 16 (less overlap)"),
        (32, 32, "Medium window, non-overlapping"),
    ]

    for window_size, stride, desc in configs:
        windowed = WindowedDataset(
            dataset=base_dataset,
            window_size=window_size,
            stride=stride,
        )
        print(f"{desc}: {len(windowed)} windows")
    print()

    # --- Step 5: Using with DataLoader ---
    print("=" * 50)
    print("Step 5: Using WindowedDataset with DataLoader")
    print("=" * 50)

    windowed = WindowedDataset(
        dataset=base_dataset,
        window_size=32,
        stride=8,
        label_strategy="last",
    )

    dataloader = DataLoader(windowed, batch_size=64, shuffle=True)

    batch_features, batch_labels = next(iter(dataloader))
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print()

    # --- Step 6: Access label encoder through wrapper ---
    print("=" * 50)
    print("Step 6: Access label encoder through wrapper")
    print("=" * 50)

    encoder = windowed.get_label_encoder()
    print(f"Number of classes: {windowed.num_classes}")
    print(f"Class labels: {encoder.classes}")

    # Decode a batch of predictions
    predictions = torch.randint(0, windowed.num_classes, (5,))
    decoded = encoder.decode(predictions)
    print(f"Sample predictions: {predictions.tolist()}")
    print(f"Decoded labels: {decoded.tolist()}")
    print()

    # --- Step 7: With custom transform ---
    print("=" * 50)
    print("Step 7: WindowedDataset with transform")
    print("=" * 50)

    def select_features(x):
        # Only keep first 14 features (e.g., one arm's kinematics)
        return x[:, :14]

    base_with_transform = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        transform=select_features,
    )

    windowed_transformed = WindowedDataset(
        dataset=base_with_transform,
        window_size=32,
        stride=1,
    )

    features, label = windowed_transformed[0]
    print(f"Window with transform shape: {features.shape}")
    print("(Note: transform is applied before windowing)")


if __name__ == "__main__":
    main()
