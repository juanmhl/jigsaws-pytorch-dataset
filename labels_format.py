from enum import Enum, auto

class LabelsFormat(Enum):
    RAW = auto()        # Original format (e.g., string or custom)
    INTEGER = auto()    # Integer encoded labels
    ONE_HOT = auto()    # One-hot encoded labels