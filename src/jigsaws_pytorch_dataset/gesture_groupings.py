"""Gesture grouping configurations for JIGSAWS.

This module defines mappings that group individual gestures into higher-level
categories. For example, the Access paper groups suturing gestures into
quality-based categories.
"""

SUTURING_ACCESS = {
    'G1': 'Q0',
    'G2': 'Q1',
    'G3': 'Q2',
    'G4': 'Q4',
    'G5': 'Q0',
    'G6': 'Q3',
    'G8': 'Q1',
    'G9': 'Q3',
    'G10': 'Q3',
    'G11': 'Q5'
}

# Registry of available groupings
_GROUPINGS = {
    "access_suturing": SUTURING_ACCESS,
    "suturing_access": SUTURING_ACCESS,  # alias
}


def get_grouping(name: str) -> dict:
    """Get a gesture grouping by name.

    Args:
        name: Name of the grouping. Available options:
            - "access_suturing" or "suturing_access": Access paper grouping for suturing

    Returns:
        Dictionary mapping original gesture labels to grouped labels.

    Raises:
        ValueError: If the grouping name is unknown.

    Example:
        >>> grouping = get_grouping("access_suturing")
        >>> grouping['G1']
        'Q0'
    """
    if name not in _GROUPINGS:
        available = ", ".join(sorted(_GROUPINGS.keys()))
        raise ValueError(f"Unknown grouping '{name}'. Available: {available}")
    return _GROUPINGS[name].copy()


def list_groupings() -> list[str]:
    """List all available gesture grouping names.

    Returns:
        List of grouping names.
    """
    return list(_GROUPINGS.keys())
