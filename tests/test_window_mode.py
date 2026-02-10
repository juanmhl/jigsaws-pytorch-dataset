"""Tests for WindowedDataset and related functionality.

This test module covers:
- WindowedDataset with different label strategies
- Sliding window overlap consistency
- Integration with the new refactored structure
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add src to path to allow imports if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from jigsaws_pytorch_dataset import KinematicsDataset, WindowedDataset
from jigsaws_pytorch_dataset.options import KinematicsSamplingMode, LabelsFormat, UnlabeledDataPolicy, Users, Trials


def test_windowed_dataset_raw_labels():
    print("\n--- Testing WindowedDataset with RAW labels ---")
    window_size = 32
    stride = 10

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.RAW,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    windowed = WindowedDataset(base_dataset, window_size=window_size, stride=stride)

    print(f"Dataset length: {len(windowed)}")
    if len(windowed) == 0:
        print("WARNING: Dataset is empty. Check dataset path.")
        return

    # Check first item
    data, label = windowed[0]
    print(f"Sample 0 data shape: {data.shape}")
    print(f"Sample 0 label: {label}")

    assert data.shape[0] == window_size, f"Expected window size {window_size}, got {data.shape[0]}"
    assert isinstance(label, str), f"Expected label to be string (RAW), got {type(label)}"

    print("RAW labels test passed.")


def test_windowed_dataset_integer_labels():
    print("\n--- Testing WindowedDataset with INTEGER labels ---")
    window_size = 20
    stride = 20  # Non-overlapping

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    windowed = WindowedDataset(base_dataset, window_size=window_size, stride=stride)

    print(f"Dataset length: {len(windowed)}")
    if len(windowed) == 0:
        return

    data, label = windowed[0]
    print(f"Sample 0 data shape: {data.shape}")
    print(f"Sample 0 label: {label}")

    assert data.shape[0] == window_size
    assert isinstance(label, torch.Tensor), f"Expected label to be torch.Tensor, got {type(label)}"
    assert label.ndim == 0, f"Expected scalar tensor for integer label, got shape {label.shape}"

    print("INTEGER labels test passed.")


def test_windowed_dataset_one_hot_labels():
    print("\n--- Testing WindowedDataset with ONE_HOT labels ---")
    window_size = 50
    stride = 5

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.ONE_HOT,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
    )

    windowed = WindowedDataset(base_dataset, window_size=window_size, stride=stride)

    print(f"Dataset length: {len(windowed)}")
    if len(windowed) == 0:
        return

    data, label = windowed[0]
    print(f"Sample 0 data shape: {data.shape}")
    print(f"Sample 0 label shape: {label.shape}")

    assert data.shape[0] == window_size
    assert len(label.shape) == 1, "Label should be 1D array"
    print(f"Number of classes: {label.shape[0]}")

    print("ONE_HOT labels test passed.")


def test_window_overlap_consistency():
    print("\n--- Testing Window Overlap Consistency ---")
    window_size = 5
    stride = 1

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.KEEP,
        users_set=(Users.B,),
        trials_set=(Trials.T1,),
    )

    windowed = WindowedDataset(base_dataset, window_size=window_size, stride=stride)

    print(f"Dataset length (User B, Trial T1): {len(windowed)}")

    if len(windowed) > 1:
        w0, l0 = windowed[0]
        w1, l1 = windowed[1]

        print("Checking overlap consistency with stride 1...")
        diff = torch.abs(w0[1:] - w1[:-1]).sum()
        print(f"Difference between overlapping parts: {diff}")
        assert diff < 1e-6, "Sliding window overlap mismatch!"
        print("Overlap consistency check passed.")


def test_label_strategies():
    print("\n--- Testing Label Strategies ---")
    window_size = 10
    stride = 5

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.KEEP,
        users_set=(Users.B,),
        trials_set=(Trials.T1,),
    )

    for strategy in ["last", "middle", "majority"]:
        windowed = WindowedDataset(
            base_dataset,
            window_size=window_size,
            stride=stride,
            label_strategy=strategy,
        )

        if len(windowed) > 0:
            data, label = windowed[0]
            print(f"Strategy '{strategy}': label = {label}")
            assert isinstance(label, torch.Tensor), f"Expected tensor, got {type(label)}"

    print("Label strategies test passed.")


def test_num_classes_passthrough():
    print("\n--- Testing num_classes passthrough ---")

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
    )

    windowed = WindowedDataset(base_dataset, window_size=32, stride=1)

    assert windowed.num_classes == base_dataset.num_classes
    print(f"num_classes: {windowed.num_classes}")
    print("num_classes passthrough test passed.")


def test_label_encoder_passthrough():
    print("\n--- Testing label encoder passthrough ---")

    base_dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
    )

    windowed = WindowedDataset(base_dataset, window_size=32, stride=1)

    encoder = windowed.get_label_encoder()
    assert encoder is base_dataset.get_label_encoder()
    print(f"Classes: {encoder.classes}")
    print("label encoder passthrough test passed.")


if __name__ == "__main__":
    try:
        test_windowed_dataset_raw_labels()
        test_windowed_dataset_integer_labels()
        test_windowed_dataset_one_hot_labels()
        test_window_overlap_consistency()
        test_label_strategies()
        test_num_classes_passthrough()
        test_label_encoder_passthrough()
        print("\nAll WindowedDataset tests passed successfully!")
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
