"""Tests for gesture groupings module."""

import sys
import os

# Add src to path to allow imports if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from jigsaws_pytorch_dataset import get_grouping, list_groupings, SUTURING_ACCESS


def test_suturing_access_constant():
    print("\n--- Testing SUTURING_ACCESS constant ---")

    print(f"SUTURING_ACCESS: {SUTURING_ACCESS}")

    # Check all expected mappings
    assert SUTURING_ACCESS['G1'] == 'Q0'
    assert SUTURING_ACCESS['G2'] == 'Q1'
    assert SUTURING_ACCESS['G3'] == 'Q2'
    assert SUTURING_ACCESS['G4'] == 'Q4'
    assert SUTURING_ACCESS['G5'] == 'Q0'
    assert SUTURING_ACCESS['G6'] == 'Q3'
    assert SUTURING_ACCESS['G8'] == 'Q1'
    assert SUTURING_ACCESS['G9'] == 'Q3'
    assert SUTURING_ACCESS['G10'] == 'Q3'
    assert SUTURING_ACCESS['G11'] == 'Q5'

    print("SUTURING_ACCESS constant test passed.")


def test_get_grouping():
    print("\n--- Testing get_grouping function ---")

    # Test valid names
    grouping = get_grouping("access_suturing")
    assert grouping == SUTURING_ACCESS
    print("get_grouping('access_suturing') works")

    grouping = get_grouping("suturing_access")  # alias
    assert grouping == SUTURING_ACCESS
    print("get_grouping('suturing_access') alias works")

    # Test that it returns a copy
    grouping = get_grouping("access_suturing")
    grouping['G1'] = 'MODIFIED'
    fresh = get_grouping("access_suturing")
    assert fresh['G1'] == 'Q0', "Should return a copy, not original"
    print("get_grouping returns a copy")

    print("get_grouping test passed.")


def test_get_grouping_invalid():
    print("\n--- Testing get_grouping with invalid name ---")

    try:
        get_grouping("invalid_name")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Got expected error: {e}")
        assert "invalid_name" in str(e)
        assert "access_suturing" in str(e)  # Should list available

    print("get_grouping invalid name test passed.")


def test_list_groupings():
    print("\n--- Testing list_groupings function ---")

    groupings = list_groupings()
    print(f"Available groupings: {groupings}")

    assert "access_suturing" in groupings
    assert "suturing_access" in groupings

    print("list_groupings test passed.")


if __name__ == "__main__":
    try:
        test_suturing_access_constant()
        test_get_grouping()
        test_get_grouping_invalid()
        test_list_groupings()
        print("\nAll gesture groupings tests passed successfully!")
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
