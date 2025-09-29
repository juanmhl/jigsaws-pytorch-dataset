from typing import List, Tuple
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict
from enum import Enum
from kinematics_sampling_mode import KinematicsSamplingMode
from labels_format import LabelsFormat
from users import Users
from trials import Trials
from unlabeled_data_policy import UnlabeledDataPolicy
from data_scalers.scalers import BaseScaler
import torch

class KinematicsDataset(Dataset):
    def __init__(self, 
                 dir: str, 
                 mode: KinematicsSamplingMode = KinematicsSamplingMode.SEQUENCE, 
                 labels_format: LabelsFormat = LabelsFormat.RAW, 
                 unlabeled_policy: UnlabeledDataPolicy = UnlabeledDataPolicy.KEEP,
                 users_set: Tuple[Users] = (Users.B, Users.C, Users.D, Users.E, Users.F, Users.G, Users.H, Users.I),
                 trials_set: Tuple[Trials] = (Trials.T1, Trials.T2, Trials.T3, Trials.T4, Trials.T5),
                 transform=None):
        """
        Initializes the JIGSAWS Kinematics Dataset.

        This class handles loading, processing, and structuring the JIGSAWS dataset
        for use with PyTorch's DataLoader.

        Args:
            dir (str): The root directory of the JIGSAWS dataset subset (e.g., "dataset/Suturing/").
            mode (KinematicsSamplingMode, optional): Defines how data is structured. 
                Defaults to KinematicsSamplingMode.SEQUENCE.
                - KinematicsSamplingMode.SEQUENCE: Each item in the dataset is a full trial sequence.
                - KinematicsSamplingMode.SAMPLE: The dataset is flattened into individual samples across all trials.
            labels_format (LabelsFormat, optional): Defines the format of the output labels. 
                Defaults to LabelsFormat.RAW.
                - LabelsFormat.RAW: Labels are kept as strings (e.g., 'G1', 'G5').
                - LabelsFormat.INTEGER: Labels are mapped to a dense, 0-indexed integer range.
                - LabelsFormat.ONE_HOT: Labels are one-hot encoded based on the unique gestures found.
            unlabeled_policy (UnlabeledDataPolicy, optional): Defines how to handle unlabeled samples. 
                Defaults to UnlabeledDataPolicy.KEEP.
                - UnlabeledDataPolicy.KEEP: Unlabeled samples are kept and assigned the default label 'G0'.
                - UnlabeledDataPolicy.IGNORE: Unlabeled samples are filtered out and discarded.
            users_set (Tuple[Users], optional): A tuple of `Users` enum members to include. 
                Defaults to all users in the dataset.
            trials_set (Tuple[Trials], optional): A tuple of `Trials` enum members to include. 
                Defaults to all trials for the selected users.
            transform (callable, optional): Optional transform to be applied on a sample. 
                Applied in __getitem__.
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
        

        # Loading all the data into self.kinematics_data and self.labels_data,
        # first as a dictionary of [user][trial] = np.array
        for kinematics_filename in os.listdir(dir_kinematics):
            label_filepath = os.path.join(dir_labels, kinematics_filename)

            if os.path.exists(label_filepath):
                match = file_pattern.match(kinematics_filename)
                if match:
                    user, trial_str = match.groups()
                    trial = int(trial_str)
                    
                    kinematics_filepath = os.path.join(dir_kinematics, kinematics_filename)
                    
                    # Read kinematics data
                    kinematics_trial_data = pd.read_csv(kinematics_filepath, sep=r'\s+', header=None).values

                    # Read and process labels data
                    num_samples = kinematics_trial_data.shape[0]
                    labels = np.full(num_samples, 'G0', dtype='<U2') # Default label 'G0'

                    labels_df = pd.read_csv(label_filepath, sep=r'\s+', header=None)
                    for _, row in labels_df.iterrows():
                        start, end, label = int(row[0]), int(row[1]), row[2]
                        labels[start-1:end] = label # Files are 1-indexed, numpy is 0-indexed
                    
                    # Handle unlabeled data policy
                    if unlabeled_policy == UnlabeledDataPolicy.IGNORE:
                        labeled_indices = np.where(labels != 'G0')[0]
                        kinematics_trial_data = kinematics_trial_data[labeled_indices]
                        labels = labels[labeled_indices]

                    self.kinematics_data[user][trial] = kinematics_trial_data

                    if labels_format == LabelsFormat.INTEGER:
                        # Remove 'G', convert to int, and apply mapping
                        labels = np.array([self.gesture_map[int(l[1:])] for l in labels])
                    elif labels_format == LabelsFormat.ONE_HOT:
                        # Remove 'G', convert to int, and apply mapping
                        int_labels = np.array([self.gesture_map[int(l[1:])] for l in labels])
                        # Then one-hot encode using auto-detected number of classes
                        labels = np.eye(num_classes)[int_labels]

                    self.labels_data[user][trial] = labels
        


        ### Load only the specified users and trials into self.data and self.labels ###

        self.data = []
        self.labels = []
        
        # Convert enums to their primitive values for key lookup
        user_values = [u.value for u in users_set] if users_set else []
        trial_values = [t.value for t in trials_set] if trials_set else []

        # Use all users/trials if sets are not provided
        if not user_values:
            user_values = self.kinematics_data.keys()
        if not trial_values:
            # Flatten all trial numbers from all users into a single set
            all_trials = set()
            for user in user_values:
                all_trials.update(self.kinematics_data.get(user, {}).keys())
            trial_values = list(all_trials)

        for user in sorted(list(user_values)):
            if user in self.kinematics_data:
                for trial in sorted(list(trial_values)):
                    if trial in self.kinematics_data[user]:
                        kin_trial_data = self.kinematics_data[user][trial]
                        label_trial_data = self.labels_data[user][trial]
                        
                        if mode == KinematicsSamplingMode.SEQUENCE:
                            self.data.append(kin_trial_data)
                            self.labels.append(label_trial_data)
                        elif mode == KinematicsSamplingMode.SAMPLE:
                            self.data.extend(kin_trial_data)
                            self.labels.extend(label_trial_data)
        
        # Convert to numpy arrays for efficiency, especially for SAMPLE mode
        if mode == KinematicsSamplingMode.SAMPLE:
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)

        self.transform = transform
        self.scaler = None
        self.mode = mode

    def get_all_transformed_data(self):
        """
        Applies the transform to the entire dataset and returns the result.
        This is useful for fitting a scaler on the training data.
        """
        if not self.transform:
            raise RuntimeError("A transform must be provided to get transformed data.")

        all_transformed_data = []
        if self.mode == KinematicsSamplingMode.SEQUENCE:
            for seq in self.data:
                transformed_seq = self.transform(torch.from_numpy(seq).float())
                all_transformed_data.append(transformed_seq)
            return torch.cat(all_transformed_data, dim=0)
        elif self.mode == KinematicsSamplingMode.SAMPLE:
            return self.transform(torch.from_numpy(self.data).float())

    def fit_scaler(self, scaler: "BaseScaler"):
        """
        Fits the given scaler on the dataset's transformed data and assigns it.

        Args:
            scaler (BaseScaler): An instance of a scaler to be fitted.
        
        Returns:
            BaseScaler: The fitted scaler.
        """
        if not self.transform:
            raise RuntimeError("A transform must be provided to fit a scaler.")

        print("Fitting scaler on the dataset's data...")
        data_to_fit = self.get_all_transformed_data()
        scaler.fit(data_to_fit)
        self.scaler = scaler
        print("Scaler fitted and assigned to the dataset.")
        return scaler

    def set_scaler(self, scaler: "BaseScaler"):
        """
        Assigns a pre-fitted scaler to the dataset.

        Args:
            scaler (BaseScaler): A pre-fitted scaler instance.
        """
        self.scaler = scaler
        print("Pre-fitted scaler assigned to the dataset.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the raw data sample
        data_sample = self.data[idx]

        # Convert to tensor for transformation
        # The transform function expects a tensor
        data_tensor = torch.from_numpy(data_sample).float()

        # Apply the transformation if it exists
        if self.transform:
            transformed_data = self.transform(data_tensor)
        else:
            transformed_data = data_tensor

        # Apply the scaler if it exists
        if self.scaler:
            # The scaler expects a tensor, which transformed_data already is
            scaled_data = self.scaler.transform(transformed_data)
        else:
            scaled_data = transformed_data

        # Pair with the corresponding label
        sample = (scaled_data, self.labels[idx])

        return sample