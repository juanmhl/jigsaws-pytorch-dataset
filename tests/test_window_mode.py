import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add src to path to allow imports if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from jigsaws_pytorch_dataset import KinematicsDataset
from jigsaws_pytorch_dataset.options import KinematicsSamplingMode, LabelsFormat, UnlabeledDataPolicy

def test_window_mode_raw_labels():
    print("\n--- Testing WINDOW mode with RAW labels ---")
    window_size = 32
    stride = 10
    
    dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.WINDOW,
        labels_format=LabelsFormat.RAW,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        window_size=window_size,
        stride=stride
    )
    
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("WARNING: Dataset is empty. Check dataset path.")
        return

    # Check first item
    data, label = dataset[0]
    print(f"Sample 0 data shape: {data.shape}")
    print(f"Sample 0 label: {label}")
    
    assert data.shape[0] == window_size, f"Expected window size {window_size}, got {data.shape[0]}"
    assert isinstance(label, str), f"Expected label to be string (RAW), got {type(label)}"
    
    print("RAW labels test passed.")

def test_window_mode_integer_labels():
    print("\n--- Testing WINDOW mode with INTEGER labels ---")
    window_size = 20
    stride = 20 # Non-overlapping
    
    dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.WINDOW,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        window_size=window_size,
        stride=stride
    )
    
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        return

    data, label = dataset[0]
    print(f"Sample 0 data shape: {data.shape}")
    print(f"Sample 0 label: {label}")
    
    assert data.shape[0] == window_size
    # In PyTorch datasets, integer labels are typically returned as tensors (LongTensor)
    assert isinstance(label, torch.Tensor), f"Expected label to be torch.Tensor, got {type(label)}"
    assert label.ndim == 0, f"Expected scalar tensor for integer label, got shape {label.shape}"
    
    print("INTEGER labels test passed.")

def test_window_mode_one_hot_labels():
    print("\n--- Testing WINDOW mode with ONE_HOT labels ---")
    window_size = 50
    stride = 5
    
    dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.WINDOW,
        labels_format=LabelsFormat.ONE_HOT,
        unlabeled_policy=UnlabeledDataPolicy.IGNORE,
        window_size=window_size,
        stride=stride
    )
    
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        return

    data, label = dataset[0]
    print(f"Sample 0 data shape: {data.shape}")
    print(f"Sample 0 label shape: {label.shape}")
    
    assert data.shape[0] == window_size
    assert len(label.shape) == 1, "Label should be 1D array"
    # Assuming standard 10 gestures or similar
    print(f"Number of classes: {label.shape[0]}")
    
    print("ONE_HOT labels test passed.")

def test_window_logic():
    print("\n--- Testing Window Logic (Manual Verification) ---")
    # Create a dummy dataset or use a very small subset if possible, 
    # but here we will just check consistency on the real data.
    
    window_size = 5
    stride = 1
    
    # We use a single user/trial to make it deterministic and small
    from jigsaws_pytorch_dataset.options import Users, Trials
    
    dataset = KinematicsDataset(
        dir="./dataset/Suturing_minmax01/",
        mode=KinematicsSamplingMode.WINDOW,
        labels_format=LabelsFormat.INTEGER,
        unlabeled_policy=UnlabeledDataPolicy.KEEP, # Keep everything to check continuity
        users_set=(Users.B,),
        trials_set=(Trials.T1,),
        window_size=window_size,
        stride=stride
    )
    
    print(f"Dataset length (User B, Trial T1): {len(dataset)}")
    
    if len(dataset) > 1:
        w0, l0 = dataset[0]
        w1, l1 = dataset[1]
        
        # With stride 1, w1[0] should equal w0[1]
        # w0 = [t0, t1, t2, t3, t4]
        # w1 = [t1, t2, t3, t4, t5]
        
        print("Checking overlap consistency with stride 1...")
        # Check if w0[1:] is close to w1[:-1]
        diff = np.abs(w0[1:] - w1[:-1]).sum()
        print(f"Difference between overlapping parts: {diff}")
        assert diff < 1e-6, "Sliding window overlap mismatch!"
        print("Overlap consistency check passed.")

if __name__ == "__main__":
    try:
        test_window_mode_raw_labels()
        test_window_mode_integer_labels()
        test_window_mode_one_hot_labels()
        test_window_logic()
        print("\nAll WINDOW mode tests passed successfully!")
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
