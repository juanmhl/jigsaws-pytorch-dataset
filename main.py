from kinematics_dataset import KinematicsDataset
from options import KinematicsSamplingMode, LabelsFormat, Users, Trials, UnlabeledDataPolicy
from transforms import extract_PSM_kinematics

def test_basic_dataset_creation():
    print("--- Testing Basic Dataset Creation ---")
    # Test with default parameters (SEQUENCE mode, RAW labels)
    suturing_dataset = KinematicsDataset()
    
    print(f"Total sequences: {len(suturing_dataset)}")
    
    # Check the shape of the first sequence
    first_sequence, first_labels = suturing_dataset[0]
    print(f"Shape of first sequence: {first_sequence.shape}")
    print(f"Labels for first sequence: {first_labels.shape}")
    print("-" * 30)

def main():
    test_basic_dataset_creation()

if __name__ == "__main__":
    main()
