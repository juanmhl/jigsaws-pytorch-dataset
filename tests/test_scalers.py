"""Tests for scaler save/load and inverse_transform."""

import sys
import os
import tempfile
import torch

# Add src to path to allow imports if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from jigsaws_pytorch_dataset import StandardScaler, MinMaxScaler, BaseScaler


def test_standard_scaler_fit_transform():
    print("\n--- Testing StandardScaler fit and transform ---")

    scaler = StandardScaler()
    data = torch.randn(100, 10) * 5 + 3  # Mean ~3, std ~5

    scaler.fit(data)
    transformed = scaler.transform(data)

    print(f"Original mean: {data.mean(dim=0)[:3]}")
    print(f"Transformed mean: {transformed.mean(dim=0)[:3]}")
    print(f"Original std: {data.std(dim=0)[:3]}")
    print(f"Transformed std: {transformed.std(dim=0)[:3]}")

    # Check standardization
    assert torch.abs(transformed.mean(dim=0)).max() < 0.2, "Mean should be ~0"
    assert torch.abs(transformed.std(dim=0) - 1.0).max() < 0.2, "Std should be ~1"

    print("StandardScaler fit/transform test passed.")


def test_minmax_scaler_fit_transform():
    print("\n--- Testing MinMaxScaler fit and transform ---")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = torch.randn(100, 10) * 5 + 3

    scaler.fit(data)
    transformed = scaler.transform(data)

    print(f"Transformed min: {transformed.min(dim=0)[0][:3]}")
    print(f"Transformed max: {transformed.max(dim=0)[0][:3]}")

    # Check range
    assert transformed.min() >= -0.01, "Min should be >= 0"
    assert transformed.max() <= 1.01, "Max should be <= 1"

    print("MinMaxScaler fit/transform test passed.")


def test_standard_scaler_inverse_transform():
    print("\n--- Testing StandardScaler inverse_transform ---")

    scaler = StandardScaler()
    original = torch.randn(100, 10)

    scaler.fit(original)
    transformed = scaler.transform(original)
    recovered = scaler.inverse_transform(transformed)

    diff = torch.abs(original - recovered).max()
    print(f"Max difference after round-trip: {diff:.10f}")

    assert diff < 1e-5, "Inverse transform should recover original data"
    print("StandardScaler inverse_transform test passed.")


def test_minmax_scaler_inverse_transform():
    print("\n--- Testing MinMaxScaler inverse_transform ---")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    original = torch.randn(100, 10) * 10

    scaler.fit(original)
    transformed = scaler.transform(original)
    recovered = scaler.inverse_transform(transformed)

    diff = torch.abs(original - recovered).max()
    print(f"Max difference after round-trip: {diff:.10f}")

    assert diff < 1e-4, "Inverse transform should recover original data"
    print("MinMaxScaler inverse_transform test passed.")


def test_standard_scaler_save_load():
    print("\n--- Testing StandardScaler save/load ---")

    scaler = StandardScaler()
    data = torch.randn(100, 10) * 5 + 3
    scaler.fit(data)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name

    try:
        scaler.save(filepath)
        print(f"Saved to {filepath}")

        loaded = StandardScaler.load(filepath)
        print("Loaded scaler")

        # Check parameters match
        assert torch.allclose(scaler.mean, loaded.mean)
        assert torch.allclose(scaler.std, loaded.std)

        # Check transform works the same
        test_data = torch.randn(10, 10)
        original_result = scaler.transform(test_data)
        loaded_result = loaded.transform(test_data)

        assert torch.allclose(original_result, loaded_result)
        print("StandardScaler save/load test passed.")

    finally:
        os.unlink(filepath)


def test_minmax_scaler_save_load():
    print("\n--- Testing MinMaxScaler save/load ---")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = torch.randn(100, 10) * 5 + 3
    scaler.fit(data)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name

    try:
        scaler.save(filepath)
        print(f"Saved to {filepath}")

        loaded = MinMaxScaler.load(filepath)
        print("Loaded scaler")

        # Check feature range preserved
        assert loaded.feature_range == (-1, 1)

        # Check transform works the same
        test_data = torch.randn(10, 10)
        original_result = scaler.transform(test_data)
        loaded_result = loaded.transform(test_data)

        assert torch.allclose(original_result, loaded_result)
        print("MinMaxScaler save/load test passed.")

    finally:
        os.unlink(filepath)


def test_base_scaler_load_auto_detect():
    print("\n--- Testing BaseScaler.load auto-detection ---")

    # Create both types of scalers
    std_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()

    data = torch.randn(100, 10)
    std_scaler.fit(data)
    mm_scaler.fit(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        std_path = os.path.join(tmpdir, 'std.pt')
        mm_path = os.path.join(tmpdir, 'mm.pt')

        std_scaler.save(std_path)
        mm_scaler.save(mm_path)

        # Load via BaseScaler.load - should detect correct type
        loaded_std = BaseScaler.load(std_path)
        loaded_mm = BaseScaler.load(mm_path)

        assert isinstance(loaded_std, StandardScaler)
        assert isinstance(loaded_mm, MinMaxScaler)
        print("BaseScaler.load auto-detection test passed.")


def test_scaler_chaining():
    print("\n--- Testing scaler method chaining ---")

    data = torch.randn(100, 10)

    # Test chaining works
    scaler = StandardScaler().fit(data)
    assert scaler.mean is not None

    scaler = MinMaxScaler().fit(data)
    assert scaler.min is not None

    print("Scaler chaining test passed.")


def test_unfitted_scaler_error():
    print("\n--- Testing unfitted scaler error handling ---")

    scaler = StandardScaler()

    try:
        scaler.transform(torch.randn(10, 5))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Got expected error: {e}")

    try:
        scaler.inverse_transform(torch.randn(10, 5))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Got expected error: {e}")

    print("Unfitted scaler error handling test passed.")


if __name__ == "__main__":
    try:
        test_standard_scaler_fit_transform()
        test_minmax_scaler_fit_transform()
        test_standard_scaler_inverse_transform()
        test_minmax_scaler_inverse_transform()
        test_standard_scaler_save_load()
        test_minmax_scaler_save_load()
        test_base_scaler_load_auto_detect()
        test_scaler_chaining()
        test_unfitted_scaler_error()
        print("\nAll scaler tests passed successfully!")
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
