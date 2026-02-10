"""Windowed dataset wrapper for sliding window access."""

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    """Wraps a sequence dataset to provide sliding window access.

    This wrapper takes a dataset that returns full sequences and provides
    access to fixed-size windows with configurable stride and label strategy.

    Args:
        dataset: A sequence dataset where each item is (features, labels) or
            (features, labels, length). Features should be 2D (seq_len, n_features).
        window_size: Size of each window.
        stride: Step size between consecutive windows. Default is 1.
        label_strategy: How to determine the label for each window:
            - "last": Use the label of the last frame (default, matches original behavior)
            - "middle": Use the label of the middle frame
            - "majority": Use the most common label in the window

    Example:
        >>> base = KinematicsDataset(dir="...", mode=KinematicsSamplingMode.SEQUENCE)
        >>> windowed = WindowedDataset(base, window_size=32, stride=1)
        >>> features, label = windowed[0]
        >>> features.shape
        torch.Size([32, n_features])
    """

    def __init__(
        self,
        dataset: Dataset,
        window_size: int,
        stride: int = 1,
        label_strategy: Literal["last", "majority", "middle"] = "last"
    ):
        self.dataset = dataset
        self.window_size = window_size
        self.stride = stride
        self.label_strategy = label_strategy

        # Pre-compute window indices for each sequence
        self._windows: list[tuple[int, int, int]] = []  # (seq_idx, start, end)
        self._build_windows()

    def _build_windows(self) -> None:
        """Build the list of all valid windows across sequences."""
        for seq_idx in range(len(self.dataset)):
            item = self.dataset[seq_idx]
            features = item[0]

            # Get sequence length
            if isinstance(features, torch.Tensor):
                seq_len = features.shape[0]
            elif isinstance(features, np.ndarray):
                seq_len = features.shape[0]
            else:
                raise TypeError(f"Expected tensor or array, got {type(features)}")

            # Generate windows for this sequence
            if seq_len >= self.window_size:
                for start in range(0, seq_len - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    self._windows.append((seq_idx, start, end))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | str]:
        """Returns (window_features, window_label).

        Args:
            idx: Window index.

        Returns:
            Tuple of (features, label) where features has shape (window_size, n_features).
        """
        seq_idx, start, end = self._windows[idx]

        # Get the full sequence
        item = self.dataset[seq_idx]
        features = item[0]
        labels = item[1]

        # Extract window
        window_features = features[start:end]
        window_labels = labels[start:end]

        # Ensure features are tensor
        if not isinstance(window_features, torch.Tensor):
            window_features = torch.from_numpy(window_features).float()

        # Get label based on strategy
        label = self._get_window_label(window_labels)

        return window_features, label

    def _get_window_label(self, window_labels) -> torch.Tensor | str:
        """Determine the label for a window based on the label strategy."""
        if self.label_strategy == "last":
            label = window_labels[-1]
        elif self.label_strategy == "middle":
            mid_idx = len(window_labels) // 2
            label = window_labels[mid_idx]
        elif self.label_strategy == "majority":
            label = self._majority_label(window_labels)
        else:
            raise ValueError(f"Unknown label strategy: {self.label_strategy}")

        return self._convert_label(label)

    def _majority_label(self, labels):
        """Get the most common label in the sequence."""
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        elif isinstance(labels, np.ndarray):
            labels_np = labels
        else:
            labels_np = np.array(labels)

        # Handle different label types
        if labels_np.dtype.kind in {'U', 'S', 'O'}:
            # String labels
            unique, counts = np.unique(labels_np, return_counts=True)
            return unique[np.argmax(counts)]
        elif labels_np.ndim == 1:
            # Integer labels
            unique, counts = np.unique(labels_np, return_counts=True)
            return unique[np.argmax(counts)]
        else:
            # One-hot: convert to int, find majority, convert back
            int_labels = np.argmax(labels_np, axis=-1)
            unique, counts = np.unique(int_labels, return_counts=True)
            majority_idx = unique[np.argmax(counts)]
            one_hot = np.zeros(labels_np.shape[-1])
            one_hot[majority_idx] = 1.0
            return one_hot

    def _convert_label(self, label) -> torch.Tensor | str:
        """Convert a single label to the appropriate output type."""
        if isinstance(label, str):
            return label
        elif isinstance(label, (np.integer, int)):
            return torch.tensor(label, dtype=torch.long)
        elif isinstance(label, np.ndarray):
            if label.dtype.kind in {'U', 'S', 'O'}:
                return str(label)
            elif label.dtype.kind == 'i':
                return torch.tensor(label, dtype=torch.long)
            else:
                return torch.from_numpy(label).float()
        elif isinstance(label, torch.Tensor):
            return label
        else:
            return label

    @property
    def num_classes(self) -> int:
        """Number of classes if the underlying dataset has this property."""
        if hasattr(self.dataset, 'num_classes'):
            return self.dataset.num_classes
        raise AttributeError("Underlying dataset does not have num_classes property")

    def get_label_encoder(self):
        """Get the label encoder from the underlying dataset if available."""
        if hasattr(self.dataset, 'get_label_encoder'):
            return self.dataset.get_label_encoder()
        raise AttributeError("Underlying dataset does not have get_label_encoder method")
