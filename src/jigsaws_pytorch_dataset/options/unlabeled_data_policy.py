from enum import Enum, auto

class UnlabeledDataPolicy(Enum):
    """
    Defines how to handle samples that are not explicitly labeled in the source files.
    """
    KEEP = auto()    # Keep unlabeled samples, assigning them a default label (e.g., 'G0').
    IGNORE = auto()  # Filter out and discard unlabeled samples entirely.
