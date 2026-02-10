"""Scaler workflow example for JIGSAWS PyTorch Dataset.

This example shows how to:
1. Fit a scaler on training data
2. Save and load scalers
3. Apply scaling via transforms
4. Use inverse_transform to recover original values
"""

import torch
from torch.utils.data import DataLoader

from jigsaws_pytorch_dataset import KinematicsDataset, MinMaxScaler, StandardScaler
from jigsaws_pytorch_dataset.options import (
    KinematicsSamplingMode,
    LabelsFormat,
    Users,
    Trials,
)


def main():
    dataset_path = "./dataset/Suturing/"

    # --- Step 1: Create train/test split by users ---
    print("=" * 50)
    print("Step 1: Create datasets with user-based split")
    print("=" * 50)

    train_users = (Users.B, Users.C, Users.D, Users.E, Users.F, Users.G)
    test_users = (Users.H, Users.I)

    train_dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        users_set=train_users,
    )

    test_dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        users_set=test_users,
        label_encoder=train_dataset.get_label_encoder(),  # Share encoder!
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    # --- Step 2: Fit scaler on training data ---
    print("=" * 50)
    print("Step 2: Fit scaler on training data")
    print("=" * 50)

    # Get all training data
    train_data = train_dataset.get_all_data()
    print(f"Training data shape: {train_data.shape}")

    # Fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    print(f"Data min (first 5 features): {train_data.min(dim=0)[0][:5]}")
    print(f"Data max (first 5 features): {train_data.max(dim=0)[0][:5]}")
    print()

    # --- Step 3: Save and load scaler ---
    print("=" * 50)
    print("Step 3: Save and load scaler")
    print("=" * 50)

    scaler.save("scaler.pt")
    print("Scaler saved to scaler.pt")

    # Load it back
    loaded_scaler = MinMaxScaler.load("scaler.pt")
    print("Scaler loaded from scaler.pt")
    print()

    # --- Step 4: Apply scaling in a custom transform ---
    print("=" * 50)
    print("Step 4: Apply scaling via transform")
    print("=" * 50)

    def scale_transform(x):
        return loaded_scaler.transform(x)

    # Create new dataset with transform
    scaled_train_dataset = KinematicsDataset(
        dir=dataset_path,
        mode=KinematicsSamplingMode.SAMPLE,
        labels_format=LabelsFormat.INTEGER,
        users_set=train_users,
        transform=scale_transform,
    )

    # Check that scaling was applied
    features, label = scaled_train_dataset[0]
    print(f"Scaled features range: [{features.min():.3f}, {features.max():.3f}]")
    print()

    # --- Step 5: Use inverse_transform to recover original values ---
    print("=" * 50)
    print("Step 5: Inverse transform to recover original values")
    print("=" * 50)

    # Get original (unscaled) sample
    original_features, _ = train_dataset[0]

    # Scale it
    scaled = loaded_scaler.transform(original_features)
    print(f"Scaled features (first 5): {scaled[:5]}")

    # Inverse transform
    recovered = loaded_scaler.inverse_transform(scaled)
    print(f"Recovered features (first 5): {recovered[:5]}")
    print(f"Original features (first 5): {original_features[:5]}")

    # Check they match
    diff = torch.abs(original_features - recovered).max()
    print(f"Max difference after round-trip: {diff:.10f}")
    print()

    # --- Step 6: StandardScaler example ---
    print("=" * 50)
    print("Step 6: StandardScaler example")
    print("=" * 50)

    std_scaler = StandardScaler()
    std_scaler.fit(train_data)

    scaled_sample = std_scaler.transform(original_features)
    print(f"Standardized mean (should be ~0): {scaled_sample.mean():.4f}")
    print(f"Standardized std (should be ~1): {scaled_sample.std():.4f}")

    # Save and load
    std_scaler.save("std_scaler.pt")
    loaded_std = StandardScaler.load("std_scaler.pt")

    recovered_std = loaded_std.inverse_transform(scaled_sample)
    diff_std = torch.abs(original_features - recovered_std).max()
    print(f"Max difference after StandardScaler round-trip: {diff_std:.10f}")

    # Cleanup
    import os
    os.remove("scaler.pt")
    os.remove("std_scaler.pt")
    print("\nCleanup: Removed scaler files")


if __name__ == "__main__":
    main()
