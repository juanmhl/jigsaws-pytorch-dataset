import torch
from torch.utils.data import DataLoader
from jigsaws_pytorch_dataset import KinematicsDataset
from jigsaws_pytorch_dataset.options import KinematicsSamplingMode, LabelsFormat, Users, Trials, UnlabeledDataPolicy
from jigsaws_pytorch_dataset.transforms import extract_PSM_kinematics
from jigsaws_pytorch_dataset.data_scalers.scalers import MinMaxScaler
from jigsaws_pytorch_dataset.collate_fns import collate_fn_seqs_with_padding

def test_basic_dataset_creation():
    print("--- Testing Basic Dataset Creation ---")
    # Test with default parameters (SEQUENCE mode, RAW labels)
    suturing_dataset = KinematicsDataset(
        dir="./dataset/Suturing/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.ONE_HOT,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        users_set=(Users.B, Users.C, Users.D, Users.E, Users.F, Users.G, Users.H, Users.I),
        trials_set=(Trials.T1, Trials.T2, Trials.T3, Trials.T4, Trials.T5),
        # transform=extract_PSM_kinematics
    )
    
    print(f"Total sequences: {len(suturing_dataset)}")
    
    # Check the shape of the first sequence
    first_sequence, first_labels = suturing_dataset[0]
    print(f"Shape of first sequence: {first_sequence.shape}")
    print(type(first_sequence))
    print(f"Labels for first sequence: {first_labels.shape}")
    print(type(first_labels))
    print("-" * 30)

def test_dataloader_with_collate_fn():
    """
    Tests the DataLoader with the custom collate function for all label formats.
    """
    print("--- Testing DataLoader with Collate Function ---")
    batch_size = 4

    # Test case 1: ONE_HOT labels
    print("\n--- Testing with ONE_HOT labels ---")
    dataset_one_hot = KinematicsDataset(
        dir="./dataset/Suturing/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.ONE_HOT,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        transform=extract_PSM_kinematics,
    )
    dataloader_one_hot = DataLoader(
        dataset_one_hot,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_seqs_with_padding
    )
    features, labels, lengths = next(iter(dataloader_one_hot))
    print(f"Features batch shape: {features.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Lengths batch: {lengths}")
    print(f"Labels type: {labels.dtype}")
    assert features.dim() == 3 and features.shape[0] == batch_size
    assert labels.dim() == 3 and labels.shape[0] == batch_size # (batch, max_len, num_classes)
    assert labels.dtype == torch.float

    # Test case 2: INTEGER labels
    print("\n--- Testing with INTEGER labels ---")
    dataset_integer = KinematicsDataset(
        dir="./dataset/Suturing/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        transform=extract_PSM_kinematics,
    )
    dataloader_integer = DataLoader(
        dataset_integer,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_seqs_with_padding
    )
    features, labels, lengths = next(iter(dataloader_integer))
    print(f"Features batch shape: {features.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Lengths batch: {lengths}")
    print(f"Labels type: {labels.dtype}")
    assert features.dim() == 3 and features.shape[0] == batch_size
    assert labels.dim() == 2 and labels.shape[0] == batch_size # (batch, max_len)
    assert labels.dtype == torch.long

    # Test case 3: RAW labels
    print("\n--- Testing with RAW labels ---")
    dataset_raw = KinematicsDataset(
        dir="./dataset/Suturing/",
        mode=KinematicsSamplingMode.SEQUENCE,
        labels_format=LabelsFormat.RAW,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        transform=extract_PSM_kinematics,
    )
    dataloader_raw = DataLoader(
        dataset_raw,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_seqs_with_padding
    )
    features, labels, lengths = next(iter(dataloader_raw))
    print(f"Features batch shape: {features.shape}")
    print(f"Labels is a tuple of lists of strings. Length: {len(labels)}")
    print(f"Lengths batch: {lengths}")
    assert features.dim() == 3 and features.shape[0] == batch_size
    assert isinstance(labels, tuple) and len(labels) == batch_size
    assert isinstance(labels[0], list) and isinstance(labels[0][0], str)
    
    print("--- DataLoader Test Complete ---\n")

def test_data_scaling():
    """
    Tests the data scaling process using separate train and test datasets.
    """
    print("--- Testing Data Scaling ---")

    # 1. Define training and testing user sets
    train_users = (Users.B, Users.C, Users.D, Users.E, Users.F)
    test_users = (Users.G, Users.H, Users.I)

    # 2. Create datasets with the kinematics transform
    train_dataset = KinematicsDataset(
        dir='dataset/Suturing',
        mode=KinematicsSamplingMode.SEQUENCE,
        users_set=train_users,
        labels_format=LabelsFormat.ONE_HOT,
        transform=extract_PSM_kinematics
    )

    test_dataset = KinematicsDataset(
        dir='dataset/Suturing',
        mode=KinematicsSamplingMode.SEQUENCE,
        users_set=test_users,
        labels_format=LabelsFormat.ONE_HOT,
        transform=extract_PSM_kinematics
    )

    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")

    # Get all transformed data BEFORE scaling and print stats
    unscaled_train_data = train_dataset.get_all_transformed_data()
    unscaled_test_data = test_dataset.get_all_transformed_data()

    print("\nVerification on unscaled training data (BEFORE scaling):")
    print(f"Min: {torch.min(unscaled_train_data, dim=0)[0]}")
    print(f"Max: {torch.max(unscaled_train_data, dim=0)[0]}")

    print("\nVerification on unscaled test data (BEFORE scaling):")
    print(f"Min: {torch.min(unscaled_test_data, dim=0)[0]}")
    print(f"Max: {torch.max(unscaled_test_data, dim=0)[0]}")

    # 3. Initialize a scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 4. Fit the scaler ONLY on the training data
    train_dataset.fit_scaler(scaler)

    # 5. Show a sample from the test data BEFORE scaling
    if len(test_dataset) > 0:
        unscaled_test_sample, _ = test_dataset[0]
        print(f"\nFirst test sample BEFORE scaling:\n{unscaled_test_sample}")

    # 6. Apply the same fitted scaler to the test data
    test_dataset.set_scaler(scaler)

    # 7. Show the same sample from the test data AFTER scaling
    if len(test_dataset) > 0:
        scaled_test_sample, _ = test_dataset[0]
        print(f"\nFirst test sample AFTER scaling:\n{scaled_test_sample}")

    # 8. Verify scaling on training data
    # Get all scaled data from the training set
    if train_dataset.scaler:
        scaled_train_data = train_dataset.scaler.transform(unscaled_train_data)
    else:
        scaled_train_data = unscaled_train_data
    
    print("\nVerification on scaled training data (AFTER scaling):")
    print(f"Min (should be close to 0): {torch.min(scaled_train_data, dim=0)[0]}")
    print(f"Max (should be close to 1): {torch.max(scaled_train_data, dim=0)[0]}")

    # 9. Verify scaling on test data
    if test_dataset.scaler:
        scaled_test_data = test_dataset.scaler.transform(unscaled_test_data)
    else:
        scaled_test_data = unscaled_test_data

    print("\nVerification on scaled test data (AFTER scaling):")
    print(f"Min (should be >= 0): {torch.min(scaled_test_data, dim=0)[0]}")
    print(f"Max (should be <= 1): {torch.max(scaled_test_data, dim=0)[0]}")
    
    print("--- Data Scaling Test Complete ---\n")

def main():
    # test_basic_dataset_creation()
    # test_data_scaling()
    test_dataloader_with_collate_fn()

if __name__ == "__main__":
    main()
