from typing import List
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict
from enum import Enum
from kinematics_sampling_mode import KinematicsSamplingMode
from labels_format import LabelsFormat

class KinematicsDataset(Dataset):
    def __init__(self, dir: str, mode: KinematicsSamplingMode, labels_format: LabelsFormat = LabelsFormat.RAW, transform=None):
        """
        Args:
            dir (str): Directory containing data.
            mode (KinematicsSamplingMode): Sampling mode.
            labels_format (LabelsFormat): The desired format for the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        dir_kinematics = os.path.join(dir, "kinematics", "AllGestures")
        dir_labels = os.path.join(dir, "transcriptions")

        self.kinematics_data = defaultdict(dict)
        self.labels_data = defaultdict(dict)

        file_pattern = re.compile(r".*_([B-I])(\d{3})\.txt")

        # Auto-detect gesture mapping for integer and one-hot encoding
        self.gesture_map = None
        num_classes = 0
        if labels_format in [LabelsFormat.INTEGER, LabelsFormat.ONE_HOT]:
            unique_gesture_nums = {0}  # Start with 0 for G0
            for kinematics_filename in os.listdir(dir_kinematics):
                label_filepath = os.path.join(dir_labels, kinematics_filename)
                if os.path.exists(label_filepath):
                    try:
                        labels_df = pd.read_csv(label_filepath, sep=r'\s+', header=None, usecols=[2])
                        if not labels_df.empty:
                            gest_nums_in_file = set(labels_df[2].str[1:].astype(int))
                            unique_gesture_nums.update(gest_nums_in_file)
                    except pd.errors.EmptyDataError:
                        # Some label files might be empty, just skip them
                        continue
            
            sorted_gestures = sorted(list(unique_gesture_nums))
            self.gesture_map = {gest_num: i for i, gest_num in enumerate(sorted_gestures)}
            num_classes = len(self.gesture_map)

            print("Gesture mapping enabled for INTEGER or ONE_HOT format:")
            print("Original Gesture -> Mapped Integer")
            # Sort map by original gesture number for clear printing
            for original_gest, mapped_int in sorted(self.gesture_map.items()):
                print(f"G{original_gest} -> {mapped_int}")
            print("-" * 30) # Separator for clarity
        
        for kinematics_filename in os.listdir(dir_kinematics):
            label_filepath = os.path.join(dir_labels, kinematics_filename)

            if os.path.exists(label_filepath):
                match = file_pattern.match(kinematics_filename)
                if match:
                    user, trial_str = match.groups()
                    trial = int(trial_str)
                    
                    kinematics_filepath = os.path.join(dir_kinematics, kinematics_filename)
                    
                    # Read kinematics data
                    kinematics_df = pd.read_csv(kinematics_filepath, sep=r'\s+', header=None)
                    self.kinematics_data[user][trial] = kinematics_df.values

                    # Read and process labels data
                    num_samples = self.kinematics_data[user][trial].shape[0]
                    labels = np.full(num_samples, 'G0', dtype='<U2') # Default label 'G0'

                    labels_df = pd.read_csv(label_filepath, sep=r'\s+', header=None)
                    for _, row in labels_df.iterrows():
                        start, end, label = int(row[0]), int(row[1]), row[2]
                        labels[start-1:end] = label # Files are 1-indexed, numpy is 0-indexed
                    
                    if labels_format == LabelsFormat.INTEGER:
                        # Remove 'G', convert to int, and apply mapping
                        labels = np.array([self.gesture_map[int(l[1:])] for l in labels])
                    elif labels_format == LabelsFormat.ONE_HOT:
                        # Remove 'G', convert to int, and apply mapping
                        int_labels = np.array([self.gesture_map[int(l[1:])] for l in labels])
                        # Then one-hot encode using auto-detected number of classes
                        labels = np.eye(num_classes)[int_labels]

                    self.labels_data[user][trial] = labels
        






    def __len__(self):
        # TODO
        pass

    def __getitem__(self, idx):
       # TODO
       pass

if __name__ == '__main__':
    # Example usage and testing
    input_dir = "dataset/Suturing/"

    print("--- Testing with LabelsFormat.RAW ---")
    try:
        raw_dataset = KinematicsDataset(
            dir=input_dir,
            mode=KinematicsSamplingMode.SEQUENCE, 
            labels_format=LabelsFormat.RAW
        )
        # Test accessing a sample trial
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