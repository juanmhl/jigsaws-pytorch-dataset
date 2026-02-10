# Migration Guide: JIGSAWS PyTorch Dataset Refactoring

This guide explains the breaking changes and new features introduced in the refactoring of the JIGSAWS PyTorch Dataset library.

## Overview

The library has been refactored to:
- Extract reusable components into separate modules
- Add missing functionality (label decoding, scaler persistence)
- Simplify the main `KinematicsDataset` class
- Follow standard PyTorch patterns (lazy transforms)

## Breaking Changes

### 1. WINDOW Mode Removed

**Before:**
```python
dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.WINDOW,
    window_size=32,
    stride=1
)
```

**After:**
```python
from jigsaws_pytorch_dataset import KinematicsDataset, WindowedDataset
from jigsaws_pytorch_dataset.options import KinematicsSamplingMode

# Create base sequence dataset
base = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.SEQUENCE,
    labels_format=LabelsFormat.INTEGER,
)

# Wrap with WindowedDataset
dataset = WindowedDataset(
    base,
    window_size=32,
    stride=1,
    label_strategy="last"  # NEW: "last", "middle", or "majority"
)
```

**Benefits:**
- Configurable label strategy (last, middle, majority)
- Cleaner separation of concerns
- Can wrap any sequence dataset

---

### 2. Scaler Methods Removed

**Before:**
```python
train_dataset.fit_scaler(scaler)
test_dataset.set_scaler(scaler)
```

**After:**
```python
from jigsaws_pytorch_dataset import MinMaxScaler

# Fit scaler externally
scaler = MinMaxScaler()
train_data = train_dataset.get_all_data()
scaler.fit(train_data)

# Save for later use
scaler.save("scaler.pt")

# Option A: Use transform
def my_transform(x):
    return scaler.transform(x)

dataset = KinematicsDataset(..., transform=my_transform)

# Option B: Apply after loading
for features, labels in dataloader:
    features = scaler.transform(features)
```

**Benefits:**
- Explicit control over scaling workflow
- Scaler persistence with save/load
- Inverse transform for recovering original values

---

### 3. Access Grouping Parameter Renamed

**Before:**
```python
dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    use_access_grouping=True
)
```

**After:**
```python
dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    gesture_grouping="access_suturing"  # or pass a custom dict
)
```

**Benefits:**
- Supports named groupings ("access_suturing")
- Supports custom grouping dictionaries
- Extensible for future groupings

---

### 4. Transforms Applied Lazily

**Before:** Transforms were applied at initialization time.

**After:** Transforms are applied in `__getitem__` (lazy evaluation).

```python
def my_transform(x):
    return x[:, :14]  # Select first 14 features

dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    transform=my_transform  # Applied when accessing items
)
```

---

## New Features

### 1. Label Decoding

Decode model predictions back to gesture names:

```python
from jigsaws_pytorch_dataset import KinematicsDataset
import torch

dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    labels_format=LabelsFormat.INTEGER,
)

# Get the label encoder
encoder = dataset.get_label_encoder()

# After model inference
predictions = model(features).argmax(dim=-1)  # tensor([0, 1, 2, 3])
gesture_names = encoder.decode(predictions)   # ['G1', 'G2', 'G3', 'G4']

# Single prediction
gesture = encoder.decode_single(2)  # 'G3'

# Get class information
print(encoder.num_classes)  # 10
print(encoder.classes)      # ['G1', 'G2', ..., 'G11']
```

---

### 2. Scaler Persistence

Save and load fitted scalers:

```python
from jigsaws_pytorch_dataset import MinMaxScaler, StandardScaler

# Fit and save
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(training_data)
scaler.save("scaler.pt")

# Load later
loaded_scaler = MinMaxScaler.load("scaler.pt")

# Or use BaseScaler.load for auto-detection
from jigsaws_pytorch_dataset import BaseScaler
scaler = BaseScaler.load("scaler.pt")  # Returns correct type
```

---

### 3. Inverse Transform

Recover original values from scaled data:

```python
scaler = StandardScaler()
scaler.fit(data)

scaled = scaler.transform(data)
recovered = scaler.inverse_transform(scaled)

# recovered ≈ data
```

---

### 4. WindowedDataset Label Strategies

Control how window labels are determined:

```python
from jigsaws_pytorch_dataset import WindowedDataset

# Use label of last frame (default, matches original behavior)
windowed = WindowedDataset(base, window_size=32, label_strategy="last")

# Use label of middle frame
windowed = WindowedDataset(base, window_size=32, label_strategy="middle")

# Use most common label in window
windowed = WindowedDataset(base, window_size=32, label_strategy="majority")
```

---

### 5. Gesture Groupings Module

Access predefined groupings or define custom ones:

```python
from jigsaws_pytorch_dataset import get_grouping, list_groupings, SUTURING_ACCESS

# List available groupings
print(list_groupings())  # ['access_suturing', 'suturing_access']

# Get a grouping
grouping = get_grouping("access_suturing")
# {'G1': 'Q0', 'G2': 'Q1', 'G3': 'Q2', ...}

# Use directly
print(SUTURING_ACCESS)

# Custom grouping
custom = {'G1': 'A', 'G2': 'A', 'G3': 'B', 'G4': 'B'}
dataset = KinematicsDataset(..., gesture_grouping=custom)
```

---

## New Package Exports

All new components are exported from the package root:

```python
from jigsaws_pytorch_dataset import (
    # Core
    KinematicsDataset,

    # New components
    LabelEncoder,
    WindowedDataset,

    # Gesture groupings
    get_grouping,
    list_groupings,
    SUTURING_ACCESS,

    # Scalers
    BaseScaler,
    StandardScaler,
    MinMaxScaler,
)
```

---

## Complete Migration Example

**Before (old API):**
```python
from jigsaws_pytorch_dataset import KinematicsDataset
from jigsaws_pytorch_dataset.options import KinematicsSamplingMode, LabelsFormat
from jigsaws_pytorch_dataset.data_scalers import MinMaxScaler

# Training dataset
train_dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.WINDOW,
    labels_format=LabelsFormat.INTEGER,
    use_access_grouping=True,
    window_size=32,
    stride=1,
    transform=my_transform,
)

# Fit scaler
scaler = MinMaxScaler()
train_dataset.fit_scaler(scaler)

# Test dataset
test_dataset = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.WINDOW,
    labels_format=LabelsFormat.INTEGER,
    use_access_grouping=True,
    window_size=32,
    stride=1,
    transform=my_transform,
)
test_dataset.set_scaler(scaler)
```

**After (new API):**
```python
from jigsaws_pytorch_dataset import (
    KinematicsDataset,
    WindowedDataset,
    MinMaxScaler,
)
from jigsaws_pytorch_dataset.options import KinematicsSamplingMode, LabelsFormat

# Fit scaler on training data first
train_base = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.SEQUENCE,
    labels_format=LabelsFormat.INTEGER,
    gesture_grouping="access_suturing",
)

scaler = MinMaxScaler()
scaler.fit(train_base.get_all_data())
scaler.save("scaler.pt")  # Persist for later

# Create transform that includes scaling
def transform(x):
    x = my_transform(x)
    return scaler.transform(x)

# Training dataset
train_base = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.SEQUENCE,
    labels_format=LabelsFormat.INTEGER,
    gesture_grouping="access_suturing",
    transform=transform,
)
train_dataset = WindowedDataset(train_base, window_size=32, stride=1)

# Test dataset (share label encoder!)
test_base = KinematicsDataset(
    dir="./dataset/Suturing/",
    mode=KinematicsSamplingMode.SEQUENCE,
    labels_format=LabelsFormat.INTEGER,
    gesture_grouping="access_suturing",
    transform=transform,
    label_encoder=train_base.get_label_encoder(),  # Share encoder
)
test_dataset = WindowedDataset(test_base, window_size=32, stride=1)

# Decode predictions
encoder = train_dataset.get_label_encoder()
predictions = model(batch).argmax(dim=-1)
gesture_names = encoder.decode(predictions)
```

---

## File Structure

```
src/jigsaws_pytorch_dataset/
├── __init__.py                   # Updated exports
├── kinematics_dataset.py         # Simplified (~255 lines)
├── label_encoder.py              # NEW: Label encoding/decoding
├── windowed_dataset.py           # NEW: Window wrapper
├── gesture_groupings.py          # NEW: Grouping configurations
├── data_scalers/
│   ├── __init__.py
│   └── scalers.py                # UPDATED: save/load/inverse_transform
├── transforms/                   # Unchanged
├── collate_fns/                  # Unchanged
└── options/                      # Unchanged

examples/
├── basic_usage.py                # Dataset creation and iteration
├── scaler_workflow.py            # External scaling with save/load
├── windowed_dataset.py           # WindowedDataset usage
└── label_decoding.py             # Decoding model predictions

tests/
├── test_window_mode.py           # Updated for WindowedDataset
├── test_label_encoder.py         # NEW
├── test_scalers.py               # NEW
└── test_gesture_groupings.py     # NEW
```

---

## Questions?

See the example files in `examples/` for complete working code, or run the tests in `tests/` to verify functionality.
