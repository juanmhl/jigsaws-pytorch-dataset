"""JIGSAWS Kinematics Dataset for PyTorch."""

import os
import re
from collections import defaultdict
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .gesture_groupings import get_grouping
from .label_encoder import LabelEncoder
from .options import (
    KinematicsSamplingMode,
    LabelsFormat,
    Trials,
    UnlabeledDataPolicy,
    Users,
)


class KinematicsDataset(Dataset):
    """JIGSAWS Kinematics Dataset.

    This class handles loading, processing, and structuring the JIGSAWS dataset
    for use with PyTorch's DataLoader.

    Args:
        dir: Root directory of the JIGSAWS dataset subset (e.g., "dataset/Suturing/").
        mode: How data is structured. Defaults to SEQUENCE.
            - SEQUENCE: Each item is a full trial sequence
            - SAMPLE: Dataset is flattened into individual samples
        labels_format: Output label format. Defaults to RAW.
            - RAW: String labels (e.g., 'G1', 'G5')
            - INTEGER: Dense 0-indexed integers
            - ONE_HOT: One-hot encoded vectors
        unlabeled_policy: How to handle unlabeled samples. Defaults to KEEP.
            - KEEP: Unlabeled samples are kept and assigned 'G0'
            - IGNORE: Unlabeled samples are filtered out
        gesture_grouping: Gesture grouping to use. Can be:
            - str: Name of a predefined grouping (e.g., "access_suturing")
            - dict: Custom mapping (e.g., {'G1': 'Q0', 'G2': 'Q1', ...})
            - None: No grouping (default)
        users_set: Tuple of Users to include. Defaults to all users.
        trials_set: Tuple of Trials to include. Defaults to all trials.
        transform: Optional transform(s) applied lazily in __getitem__.
            Can be a single callable, a list/tuple of callables, or None.
        label_encoder: Optional pre-fitted LabelEncoder. If not provided,
            one will be created and fitted automatically.

    Example:
        >>> dataset = KinematicsDataset(
        ...     dir="./dataset/Suturing/",
        ...     mode=KinematicsSamplingMode.SEQUENCE,
        ...     labels_format=LabelsFormat.INTEGER,
        ... )
        >>> features, labels, length = dataset[0]
    """

    def __init__(
        self,
        dir: str,
        mode: KinematicsSamplingMode = KinematicsSamplingMode.SEQUENCE,
        labels_format: LabelsFormat = LabelsFormat.RAW,
        unlabeled_policy: UnlabeledDataPolicy = UnlabeledDataPolicy.KEEP,
        gesture_grouping: Optional[str | dict] = None,
        users_set: Optional[Tuple[Users, ...]] = None,
        trials_set: Optional[Tuple[Trials, ...]] = None,
        transform: Optional[Union[Callable, Sequence[Callable]]] = None,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        if users_set is None:
            users_set = (Users.B, Users.C, Users.D, Users.E, Users.F, Users.G, Users.H, Users.I)
        if trials_set is None:
            trials_set = (Trials.T1, Trials.T2, Trials.T3, Trials.T4, Trials.T5)

        self.mode = mode
        self.labels_format = labels_format

        # Store transforms as a list
        if transform is None:
            self._transforms: list[Callable] = []
        elif isinstance(transform, (list, tuple)):
            self._transforms = list(transform)
        else:
            self._transforms = [transform]

        # Resolve gesture grouping
        if isinstance(gesture_grouping, str):
            self._gesture_grouping = get_grouping(gesture_grouping)
        else:
            self._gesture_grouping = gesture_grouping

        dir_kinematics = os.path.join(dir, "kinematics", "AllGestures")
        dir_labels = os.path.join(dir, "transcriptions")

        file_pattern = re.compile(r".*_([B-I])(\d{3})\.txt")

        # Load raw data from files
        kinematics_data = defaultdict(dict)
        labels_data = defaultdict(dict)
        raw_labels_for_fitting = []

        for kinematics_filename in os.listdir(dir_kinematics):
            label_filepath = os.path.join(dir_labels, kinematics_filename)

            if not os.path.exists(label_filepath):
                continue

            match = file_pattern.match(kinematics_filename)
            if not match:
                continue

            user, trial_str = match.groups()
            trial = int(trial_str)

            kinematics_filepath = os.path.join(dir_kinematics, kinematics_filename)

            # Read kinematics data
            kinematics_trial_data = pd.read_csv(
                kinematics_filepath, sep=r'\s+', header=None
            ).values

            # Read and process labels
            num_samples = kinematics_trial_data.shape[0]
            labels = np.full(num_samples, 'G0', dtype='<U3')

            labels_df = pd.read_csv(label_filepath, sep=r'\s+', header=None)
            for _, row in labels_df.iterrows():
                start, end, label = int(row[0]), int(row[1]), row[2]
                if self._gesture_grouping:
                    label = self._gesture_grouping.get(label, label)
                labels[start-1:end] = label

            # Handle unlabeled data policy
            if unlabeled_policy == UnlabeledDataPolicy.IGNORE:
                labeled_indices = np.where(labels != 'G0')[0]
                kinematics_trial_data = kinematics_trial_data[labeled_indices]
                labels = labels[labeled_indices]

            kinematics_data[user][trial] = kinematics_trial_data
            labels_data[user][trial] = labels
            raw_labels_for_fitting.append(labels)

        # Set up label encoder
        if label_encoder is not None:
            self._label_encoder = label_encoder
        else:
            self._label_encoder = LabelEncoder(
                labels_format=labels_format,
                gesture_grouping=self._gesture_grouping
            )
            include_unlabeled = unlabeled_policy == UnlabeledDataPolicy.KEEP
            self._label_encoder.fit(raw_labels_for_fitting, include_unlabeled=include_unlabeled)

        # Build dataset from selected users and trials
        self.data = []
        self.labels = []

        user_values = [u.value for u in users_set]
        trial_values = [t.value for t in trials_set]

        for user in sorted(user_values):
            if user not in kinematics_data:
                continue
            for trial in sorted(trial_values):
                if trial not in kinematics_data[user]:
                    continue

                kin_trial_data = kinematics_data[user][trial]
                label_trial_data = labels_data[user][trial]

                # Encode labels
                encoded_labels = self._label_encoder.encode(label_trial_data)

                if mode == KinematicsSamplingMode.SEQUENCE:
                    self.data.append(kin_trial_data)
                    self.labels.append(encoded_labels)
                elif mode == KinematicsSamplingMode.SAMPLE:
                    self.data.extend(kin_trial_data)
                    self.labels.extend(encoded_labels)

        # Convert to numpy arrays for SAMPLE mode
        if mode == KinematicsSamplingMode.SAMPLE:
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a data sample.

        Returns:
            For SEQUENCE mode: (features, labels, length)
            For SAMPLE mode: (features, label)
        """
        data_sample = self.data[idx]

        # Convert to tensor
        if isinstance(data_sample, torch.Tensor):
            data_tensor = data_sample
        else:
            data_tensor = torch.from_numpy(data_sample).float()

        # Apply transforms lazily
        data_tensor = self._apply_transforms(data_tensor)

        # Get label
        label = self.labels[idx]
        label_out = self._convert_label(label)

        if self.mode == KinematicsSamplingMode.SAMPLE:
            return data_tensor, label_out
        else:
            return data_tensor, label_out, data_tensor.shape[0]

    def _convert_label(self, label):
        """Convert label to appropriate tensor type."""
        if isinstance(label, (np.integer, int)):
            return torch.tensor(label, dtype=torch.long)
        elif isinstance(label, np.ndarray):
            if label.dtype.kind in {'U', 'S', 'O'}:
                return label.tolist() if label.ndim > 0 else str(label)
            elif label.dtype.kind == 'i':
                return torch.from_numpy(label).long()
            else:
                return torch.from_numpy(label).float()
        elif isinstance(label, torch.Tensor):
            return label
        else:
            return label

    def _apply_transforms(self, data: torch.Tensor) -> torch.Tensor:
        for t in self._transforms:
            data = t(data)
        return data

    def add_transform(self, transform: Callable) -> None:
        """Append a transform to the pipeline."""
        self._transforms.append(transform)

    @property
    def transform(self) -> list[Callable]:
        """The list of transforms applied to data."""
        return self._transforms

    def get_label_encoder(self) -> LabelEncoder:
        """Get the label encoder for decoding predictions."""
        return self._label_encoder

    @property
    def num_classes(self) -> int:
        """Number of gesture classes."""
        return self._label_encoder.num_classes

    def get_all_data(self) -> torch.Tensor:
        """Returns all data as a single tensor with transforms applied.

        Useful for fitting scalers on transformed data.
        """
        if self.mode == KinematicsSamplingMode.SEQUENCE:
            tensors = []
            for seq in self.data:
                if isinstance(seq, torch.Tensor):
                    t = seq
                else:
                    t = torch.from_numpy(seq).float()
                tensors.append(self._apply_transforms(t))
            return torch.cat(tensors, dim=0)
        else:
            if isinstance(self.data, torch.Tensor):
                all_data = self.data
            else:
                all_data = torch.from_numpy(self.data).float()
            return self._apply_transforms(all_data)
