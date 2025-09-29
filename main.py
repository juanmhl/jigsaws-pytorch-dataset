import torch
import numpy as np
from kinematics_dataset import KinematicsDataset
from kinematics_sampling_mode import KinematicsSamplingMode
from users import Users
from trials import Trials
from labels_format import LabelsFormat
from data_scalers.scalers import StandardScaler, MinMaxScaler
from transforms.extract_PSM_kinematics import extract_PSM_kinematics

def main():
    # Check for PyTorch and CUDA
    try:
        print("PyTorch imported successfully.")
        print("CUDA available:", torch.cuda.is_available())
    except ImportError:
        print("PyTorch is not installed.")
        return

    print("\n" + "="*50 + "\n")
    
    # --- Dataset Testing ---
    input_dir = "dataset/Suturing/"

    print("--- Testing with LabelsFormat.RAW ---")
    try:
        raw_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SEQUENCE, 
            labels_format=LabelsFormat.RAW
        )

        # Calculate and print memory usage
        data_bytes = sum(arr.nbytes for arr in raw_dataset.data)
        labels_bytes = sum(arr.nbytes for arr in raw_dataset.labels)
        total_mb = (data_bytes + labels_bytes) / (1024**2)
        print(f"Estimated memory usage of the loaded dataset: {total_mb:.2f} MB")

        # Test accessing a sample trial from the loaded data structure
        sample_user = 'B'
        sample_trial = 1
        if sample_user in raw_dataset.kinematics_data and sample_trial in raw_dataset.kinematics_data[sample_user]:
            kinematics_sample = raw_dataset.kinematics_data[sample_user][sample_trial]
            labels_sample = raw_dataset.labels_data[sample_user][sample_trial]
            print(f"User '{sample_user}', Trial {sample_trial}:")
            print(f"  Kinematics data shape: {kinematics_sample.shape}")
            print(f"  First row of kinematics data: {kinematics_sample[0]}")
            print(f"  Labels data shape: {labels_sample.shape}")
            print(f"  Sample labels (first 10): {labels_sample[:10]}")
        else:
            print(f"Sample trial not found for User '{sample_user}', Trial {sample_trial}.")
            print("Available users:", list(raw_dataset.kinematics_data.keys()))

    except FileNotFoundError:
        print(f"Error: The directory '{input_dir}' was not found.")
        print("Please ensure the dataset is correctly placed.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n" + "="*50 + "\n")

    print("--- Testing with LabelsFormat.INTEGER ---")
    try:
        int_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SEQUENCE,
            labels_format=LabelsFormat.INTEGER
        )
        if sample_user in int_dataset.kinematics_data and sample_trial in int_dataset.kinematics_data[sample_user]:
            labels_sample = int_dataset.labels_data[sample_user][sample_trial]
            print(f"User '{sample_user}', Trial {sample_trial}:")
            print(f"  Sample labels (first 10): {labels_sample[:10]}")
            print(f"  Unique integer labels in trial: {np.unique(labels_sample)}")
    except FileNotFoundError:
        print(f"Error: The directory '{input_dir}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n" + "="*50 + "\n")

    print("--- Testing with LabelsFormat.ONE_HOT ---")
    try:
        one_hot_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SEQUENCE,
            labels_format=LabelsFormat.ONE_HOT
        )
        if sample_user in one_hot_dataset.kinematics_data and sample_trial in one_hot_dataset.kinematics_data[sample_user]:
            labels_sample = one_hot_dataset.labels_data[sample_user][sample_trial]
            print(f"User '{sample_user}', Trial {sample_trial}:")
            print(f"  Labels data shape: {labels_sample.shape}")
            print(f"  Sample one-hot label (first one): {labels_sample[0]}")
    except FileNotFoundError:
        print(f"Error: The directory '{input_dir}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("--- Train/Val/Test Split and Scaling Workflow ---")
    try:
        # Define user and trial splits for train, validation, and test sets
        train_users = (Users.B, Users.C, Users.D, Users.E)
        val_users = (Users.F, Users.G)
        test_users = (Users.H, Users.I)
        
        all_trials = (Trials.T1, Trials.T2, Trials.T3, Trials.T4, Trials.T5)

        # --- Training Set ---
        # 1. Create the training dataset without a scaler first.
        print("Creating training dataset...")
        train_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SAMPLE,
            labels_format=LabelsFormat.INTEGER,
            users_set=train_users,
            trials_set=all_trials,
            transform=extract_PSM_kinematics,
            scaler=None  # No scaler passed initially
        )

        # 2. Fit the scaler on the transformed training data using the new method.
        print("Fitting scaler on training data...")
        scaler = train_dataset.fit_scaler(StandardScaler())
        
        # --- Validation Set ---
        # 3. Create the validation dataset, applying the same fitted scaler.
        print("Creating validation dataset with the same scaler...")
        val_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SAMPLE,
            labels_format=LabelsFormat.INTEGER,
            users_set=val_users,
            trials_set=all_trials,
            transform=extract_PSM_kinematics,
            scaler=scaler  # Reuse the scaler fitted on training data
        )

        # --- Test Set ---
        # 4. Create the test dataset, also with the same scaler.
        print("Creating test dataset with the same scaler...")
        test_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SAMPLE,
            labels_format=LabelsFormat.INTEGER,
            users_set=test_users,
            trials_set=all_trials,
            transform=extract_PSM_kinematics,
            scaler=scaler  # Reuse the scaler
        )

        print("--- Verification ---")
        # Verify that the scaler is applied correctly
        train_sample, _ = train_dataset[0]
        val_sample, _ = val_dataset[0]
        test_sample, _ = test_dataset[0]

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        print(f"First scaled train sample (first 5 features): {train_sample[:5]}")
        print(f"First scaled validation sample (first 5 features): {val_sample[:5]}")
        print(f"First scaled test sample (first 5 features): {test_sample[:5]}")

        # Optional: Check the mean and std of the scaled training data
        all_scaled_train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        print(f"Mean of all scaled training data: {torch.mean(all_scaled_train_data, dim=0)}")
        print(f"Std of all scaled training data: {torch.std(all_scaled_train_data, dim=0)}")

    except Exception as e:
        print(f"An error occurred during the train/val/test split workflow: {e}")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

