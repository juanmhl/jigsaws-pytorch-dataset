"""Label encoding for JIGSAWS gesture labels."""

from typing import Optional

import numpy as np
import torch

from .options import LabelsFormat


class LabelEncoder:
    """Encodes gesture labels to INTEGER or ONE_HOT format. Provides decode().

    This class handles the mapping between string gesture labels (e.g., 'G1', 'G5')
    and their encoded representations (integers or one-hot vectors).

    Args:
        labels_format: The output format for encoded labels.
            - LabelsFormat.RAW: No encoding, returns original string labels
            - LabelsFormat.INTEGER: Maps to dense 0-indexed integers
            - LabelsFormat.ONE_HOT: One-hot encoded vectors
        gesture_grouping: Optional dict mapping original gestures to grouped gestures.
            For example: {'G1': 'Q0', 'G2': 'Q1', ...}
    """

    def __init__(
        self,
        labels_format: LabelsFormat = LabelsFormat.INTEGER,
        gesture_grouping: Optional[dict] = None
    ):
        self.labels_format = labels_format
        self.gesture_grouping = gesture_grouping
        self._gesture_map: Optional[dict] = None
        self._reverse_map: Optional[dict] = None
        self._num_classes: int = 0
        self._fitted: bool = False

    def fit(
        self,
        raw_labels: list[np.ndarray],
        include_unlabeled: bool = True
    ) -> "LabelEncoder":
        """Discover unique gestures and build mapping.

        Args:
            raw_labels: List of label arrays (one per sequence/trial).
                Each array contains string labels like 'G0', 'G1', etc.
            include_unlabeled: Whether to include 'G0' (unlabeled) in the mapping.

        Returns:
            self: The fitted encoder for chaining.
        """
        if self.labels_format == LabelsFormat.RAW:
            self._fitted = True
            return self

        # Collect unique gesture numbers
        if include_unlabeled:
            unique_gesture_nums = {0}
        else:
            unique_gesture_nums = set()

        for labels in raw_labels:
            for label in labels:
                # Apply grouping if provided
                if self.gesture_grouping:
                    label = self.gesture_grouping.get(label, label)
                # Extract number from label (e.g., 'G1' -> 1, 'Q0' -> 0)
                gesture_num = int(label[1:])
                unique_gesture_nums.add(gesture_num)

        # Create sorted mapping
        sorted_gestures = sorted(list(unique_gesture_nums))
        self._gesture_map = {gest_num: i for i, gest_num in enumerate(sorted_gestures)}
        self._reverse_map = {i: gest_num for gest_num, i in self._gesture_map.items()}
        self._num_classes = len(self._gesture_map)
        self._fitted = True

        return self

    def encode(self, labels: np.ndarray) -> np.ndarray | torch.Tensor:
        """Encode string labels to configured format.

        Args:
            labels: Array of string labels (e.g., ['G1', 'G2', 'G1', ...])

        Returns:
            Encoded labels:
                - RAW: Original numpy array unchanged
                - INTEGER: numpy array of integers
                - ONE_HOT: numpy array of shape (n_samples, num_classes)
        """
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted before encoding. Call fit() first.")

        if self.labels_format == LabelsFormat.RAW:
            return labels

        # Apply grouping if provided
        if self.gesture_grouping:
            labels = np.array([self.gesture_grouping.get(l, l) for l in labels])

        # Convert to integers using gesture map
        int_labels = np.array([self._gesture_map[int(l[1:])] for l in labels])

        if self.labels_format == LabelsFormat.INTEGER:
            return int_labels
        elif self.labels_format == LabelsFormat.ONE_HOT:
            return np.eye(self._num_classes)[int_labels]

        raise ValueError(f"Unknown labels format: {self.labels_format}")

    def encode_single(self, label: str) -> int | np.ndarray:
        """Encode a single label.

        Args:
            label: A single string label (e.g., 'G1')

        Returns:
            Encoded label (int for INTEGER, 1D array for ONE_HOT)
        """
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted before encoding. Call fit() first.")

        if self.labels_format == LabelsFormat.RAW:
            return label

        if self.gesture_grouping:
            label = self.gesture_grouping.get(label, label)

        int_label = self._gesture_map[int(label[1:])]

        if self.labels_format == LabelsFormat.INTEGER:
            return int_label
        elif self.labels_format == LabelsFormat.ONE_HOT:
            one_hot = np.zeros(self._num_classes)
            one_hot[int_label] = 1.0
            return one_hot

        raise ValueError(f"Unknown labels format: {self.labels_format}")

    def decode(self, encoded: np.ndarray | torch.Tensor) -> np.ndarray:
        """Decode encoded labels back to string labels.

        Args:
            encoded: Encoded labels (integers or one-hot vectors).
                Can be numpy array or torch tensor.

        Returns:
            Array of string labels (e.g., ['G1', 'G2', ...])
        """
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted before decoding. Call fit() first.")

        if self.labels_format == LabelsFormat.RAW:
            if isinstance(encoded, torch.Tensor):
                return encoded.numpy()
            return encoded

        # Convert torch tensor to numpy if needed
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()

        # Handle one-hot: convert to integer indices
        if self.labels_format == LabelsFormat.ONE_HOT or (encoded.ndim > 1 and encoded.shape[-1] > 1):
            encoded = np.argmax(encoded, axis=-1)

        # Decode integers to gesture labels
        prefix = "Q" if self.gesture_grouping else "G"

        # Handle both scalar and array inputs
        if encoded.ndim == 0:
            return np.array([f"{prefix}{self._reverse_map[int(encoded)]}"])

        return np.array([f"{prefix}{self._reverse_map[int(i)]}" for i in encoded])

    def decode_single(self, encoded: int | np.ndarray | torch.Tensor) -> str:
        """Decode a single encoded label back to string.

        Args:
            encoded: Single encoded label (integer or one-hot vector)

        Returns:
            String label (e.g., 'G1')
        """
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted before decoding. Call fit() first.")

        if self.labels_format == LabelsFormat.RAW:
            return str(encoded)

        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()

        # Handle one-hot
        if isinstance(encoded, np.ndarray) and encoded.ndim > 0:
            encoded = int(np.argmax(encoded))
        else:
            encoded = int(encoded)

        prefix = "Q" if self.gesture_grouping else "G"
        return f"{prefix}{self._reverse_map[encoded]}"

    @property
    def num_classes(self) -> int:
        """Number of gesture classes."""
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted first. Call fit().")
        return self._num_classes

    @property
    def gesture_map(self) -> dict:
        """Mapping from gesture number to encoded integer index."""
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted first. Call fit().")
        return self._gesture_map.copy()

    @property
    def classes(self) -> list[str]:
        """List of class labels in order of their encoded indices."""
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fitted first. Call fit().")
        prefix = "Q" if self.gesture_grouping else "G"
        return [f"{prefix}{self._reverse_map[i]}" for i in range(self._num_classes)]
