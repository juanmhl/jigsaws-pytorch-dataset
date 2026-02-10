"""Tests for LabelEncoder class."""

import sys
import os
import numpy as np
import torch

# Add src to path to allow imports if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from jigsaws_pytorch_dataset import LabelEncoder
from jigsaws_pytorch_dataset.options import LabelsFormat


def test_fit_and_encode_integer():
    print("\n--- Testing LabelEncoder fit and encode (INTEGER) ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.INTEGER)

    # Simulate raw labels from dataset
    raw_labels = [
        np.array(['G0', 'G1', 'G1', 'G2', 'G3']),
        np.array(['G1', 'G2', 'G4', 'G5', 'G6']),
    ]

    encoder.fit(raw_labels, include_unlabeled=True)

    print(f"Gesture map: {encoder.gesture_map}")
    print(f"Num classes: {encoder.num_classes}")
    print(f"Classes: {encoder.classes}")

    # Test encoding
    test_labels = np.array(['G0', 'G1', 'G2'])
    encoded = encoder.encode(test_labels)
    print(f"Encoded: {encoded}")

    assert encoder.num_classes == 7  # G0, G1, G2, G3, G4, G5, G6
    assert len(encoded) == 3
    assert encoded[0] == 0  # G0 -> 0
    print("INTEGER encoding test passed.")


def test_fit_and_encode_one_hot():
    print("\n--- Testing LabelEncoder fit and encode (ONE_HOT) ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.ONE_HOT)

    raw_labels = [np.array(['G1', 'G2', 'G3'])]
    encoder.fit(raw_labels, include_unlabeled=False)

    print(f"Num classes: {encoder.num_classes}")

    test_labels = np.array(['G1', 'G2', 'G3'])
    encoded = encoder.encode(test_labels)
    print(f"Encoded shape: {encoded.shape}")

    assert encoded.shape == (3, 3)  # 3 samples, 3 classes
    assert encoded[0, 0] == 1.0  # G1 is first class
    assert encoded[0].sum() == 1.0  # one-hot sum
    print("ONE_HOT encoding test passed.")


def test_decode_integer():
    print("\n--- Testing LabelEncoder decode (INTEGER) ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.INTEGER)
    raw_labels = [np.array(['G1', 'G2', 'G3', 'G4', 'G5'])]
    encoder.fit(raw_labels, include_unlabeled=False)

    # Encode then decode
    original = np.array(['G1', 'G3', 'G5'])
    encoded = encoder.encode(original)
    print(f"Encoded: {encoded}")

    decoded = encoder.decode(encoded)
    print(f"Decoded: {decoded}")

    assert np.array_equal(original, decoded)
    print("INTEGER decode test passed.")


def test_decode_one_hot():
    print("\n--- Testing LabelEncoder decode (ONE_HOT) ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.ONE_HOT)
    raw_labels = [np.array(['G1', 'G2', 'G3'])]
    encoder.fit(raw_labels, include_unlabeled=False)

    # Encode then decode
    original = np.array(['G1', 'G3'])
    encoded = encoder.encode(original)
    print(f"Encoded shape: {encoded.shape}")

    decoded = encoder.decode(encoded)
    print(f"Decoded: {decoded}")

    assert np.array_equal(original, decoded)
    print("ONE_HOT decode test passed.")


def test_decode_torch_tensor():
    print("\n--- Testing LabelEncoder decode with torch.Tensor ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.INTEGER)
    raw_labels = [np.array(['G0', 'G1', 'G2', 'G3'])]
    encoder.fit(raw_labels)

    # Simulate model predictions as tensor
    predictions = torch.tensor([0, 1, 2, 3])
    decoded = encoder.decode(predictions)
    print(f"Decoded from tensor: {decoded}")

    expected = np.array(['G0', 'G1', 'G2', 'G3'])
    assert np.array_equal(decoded, expected)
    print("Torch tensor decode test passed.")


def test_decode_single():
    print("\n--- Testing LabelEncoder decode_single ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.INTEGER)
    raw_labels = [np.array(['G0', 'G1', 'G2'])]
    encoder.fit(raw_labels)

    # Decode single values
    for i in range(3):
        decoded = encoder.decode_single(i)
        print(f"{i} -> {decoded}")
        assert decoded == f'G{i}'

    print("decode_single test passed.")


def test_with_gesture_grouping():
    print("\n--- Testing LabelEncoder with gesture grouping ---")

    grouping = {'G1': 'Q0', 'G2': 'Q1', 'G3': 'Q0'}

    encoder = LabelEncoder(
        labels_format=LabelsFormat.INTEGER,
        gesture_grouping=grouping
    )

    raw_labels = [np.array(['G1', 'G2', 'G3'])]
    encoder.fit(raw_labels, include_unlabeled=False)

    print(f"Num classes: {encoder.num_classes}")
    print(f"Classes: {encoder.classes}")

    # G1 and G3 both map to Q0
    assert encoder.num_classes == 2  # Q0 and Q1

    # Encode
    test_labels = np.array(['G1', 'G3'])  # Both should become Q0
    encoded = encoder.encode(test_labels)
    print(f"Encoded: {encoded}")
    assert encoded[0] == encoded[1]  # Same class

    # Decode (returns Q labels since grouping is active)
    decoded = encoder.decode(encoded)
    print(f"Decoded: {decoded}")
    assert decoded[0] == 'Q0'
    assert decoded[1] == 'Q0'

    print("Gesture grouping test passed.")


def test_raw_format():
    print("\n--- Testing LabelEncoder with RAW format ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.RAW)
    raw_labels = [np.array(['G1', 'G2', 'G3'])]
    encoder.fit(raw_labels)

    test_labels = np.array(['G1', 'G2'])
    encoded = encoder.encode(test_labels)

    # RAW format should return labels unchanged
    assert np.array_equal(test_labels, encoded)
    print("RAW format test passed.")


def test_classes_property():
    print("\n--- Testing LabelEncoder classes property ---")

    encoder = LabelEncoder(labels_format=LabelsFormat.INTEGER)
    raw_labels = [np.array(['G3', 'G1', 'G5'])]  # Not in order
    encoder.fit(raw_labels, include_unlabeled=False)

    classes = encoder.classes
    print(f"Classes: {classes}")

    # Should be sorted by gesture number
    assert classes == ['G1', 'G3', 'G5']
    print("classes property test passed.")


if __name__ == "__main__":
    try:
        test_fit_and_encode_integer()
        test_fit_and_encode_one_hot()
        test_decode_integer()
        test_decode_one_hot()
        test_decode_torch_tensor()
        test_decode_single()
        test_with_gesture_grouping()
        test_raw_format()
        test_classes_property()
        print("\nAll LabelEncoder tests passed successfully!")
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
