#!/usr/bin/env python3
"""
Preprocessing script for JIGSAWS dataset.

This script reads kinematic data files, applies the PSM kinematics extraction transform,
performs per-user scaling (normalization or min-max), and saves the transformed data to a new directory structure.

Usage:
    python dataset_transform.py --input_dir <input_dir> --output_dir <output_dir> [options]

Example:
    python dataset_transform.py --input_dir dataset/Suturing --output_dir dataset/Suturing_transformed
    python dataset_transform.py --input_dir dataset/Suturing --output_dir dataset/Suturing_minmax --scaling minmax --range -1 1
"""

import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jigsaws_pytorch_dataset.transforms.extract_PSM_kinematics import extract_PSM_kinematics


def load_kinematics_data(input_dir):
    """
    Load all kinematics data from the input directory.
    
    Args:
        input_dir (str): Path to the input directory containing kinematics/AllGestures folder
        
    Returns:
        dict: Dictionary with structure {user: {trial: numpy_array}}
    """
    dir_kinematics = os.path.join(input_dir, "kinematics", "AllGestures")
    
    if not os.path.exists(dir_kinematics):
        raise ValueError(f"Kinematics directory not found: {dir_kinematics}")
    
    kinematics_data = defaultdict(dict)
    file_pattern = re.compile(r".*_([B-I])(\d{3})\.txt")
    
    print("Loading kinematics data...")
    for filename in tqdm(sorted(os.listdir(dir_kinematics))):
        match = file_pattern.match(filename)
        if match:
            user, trial_str = match.groups()
            trial = int(trial_str)
            
            filepath = os.path.join(dir_kinematics, filename)
            data = pd.read_csv(filepath, sep=r'\s+', header=None).values
            
            kinematics_data[user][trial] = data
    
    return kinematics_data


def apply_transform(kinematics_data):
    """
    Apply the PSM kinematics extraction transform to all data.
    
    Args:
        kinematics_data (dict): Dictionary with structure {user: {trial: numpy_array}}
        
    Returns:
        dict: Transformed data with same structure
    """
    transformed_data = defaultdict(dict)
    
    print("Applying PSM kinematics extraction transform...")
    for user in tqdm(sorted(kinematics_data.keys())):
        for trial in sorted(kinematics_data[user].keys()):
            data = kinematics_data[user][trial]
            
            # Convert to tensor and apply transform
            data_tensor = torch.from_numpy(data).float()
            transformed_tensor = extract_PSM_kinematics(data_tensor)
            
            # Convert back to numpy for storage
            transformed_data[user][trial] = transformed_tensor.numpy()
    
    return transformed_data


def compute_per_user_statistics(transformed_data, scaling_method='normalize'):
    """
    Compute statistics for each user across all their trials.
    
    Args:
        transformed_data (dict): Dictionary with structure {user: {trial: numpy_array}}
        scaling_method (str): Either 'normalize' (z-score) or 'minmax'
        
    Returns:
        dict: Dictionary with structure {user: {'mean': array, 'std': array, 'min': array, 'max': array}}
    """
    statistics = {}
    
    print(f"Computing per-user statistics for {scaling_method} scaling...")
    for user in tqdm(sorted(transformed_data.keys())):
        # Concatenate all trials for this user
        user_data = []
        for trial in sorted(transformed_data[user].keys()):
            user_data.append(transformed_data[user][trial])
        
        user_data = np.vstack(user_data)
        
        # Compute statistics for both methods
        mean = np.mean(user_data, axis=0)
        std = np.std(user_data, axis=0)
        min_val = np.min(user_data, axis=0)
        max_val = np.max(user_data, axis=0)
        
        # Avoid division by zero: set std to 1 where it's 0
        std[std == 0] = 1.0
        
        # Avoid division by zero for min-max: set range to 1 where min==max
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        statistics[user] = {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val
        }
    
    return statistics


def scale_data(transformed_data, statistics, scaling_method='normalize', minmax_range=(-1, 1)):
    """
    Scale data using per-user statistics.
    
    Args:
        transformed_data (dict): Dictionary with structure {user: {trial: numpy_array}}
        statistics (dict): Dictionary with structure {user: {'mean': array, 'std': array, 'min': array, 'max': array}}
        scaling_method (str): Either 'normalize' (z-score) or 'minmax'
        minmax_range (tuple): Target range for min-max scaling (min, max)
        
    Returns:
        dict: Scaled data with same structure
    """
    scaled_data = defaultdict(dict)
    
    print(f"Scaling data per user using {scaling_method} method...")
    for user in tqdm(sorted(transformed_data.keys())):
        for trial in sorted(transformed_data[user].keys()):
            data = transformed_data[user][trial]
            
            if scaling_method == 'normalize':
                # Z-score normalization: (x - mean) / std
                mean = statistics[user]['mean']
                std = statistics[user]['std']
                scaled = (data - mean) / std
            elif scaling_method == 'minmax':
                # Min-max scaling: ((x - min) / (max - min)) * (new_max - new_min) + new_min
                min_val = statistics[user]['min']
                range_val = statistics[user]['range']
                new_min, new_max = minmax_range
                new_range = new_max - new_min
                
                # First normalize to [0, 1]
                normalized = (data - min_val) / range_val
                # Then scale to desired range
                scaled = normalized * new_range + new_min
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}. Use 'normalize' or 'minmax'")
            
            scaled_data[user][trial] = scaled
    
    return scaled_data


def save_transformed_data(scaled_data, input_dir, output_dir):
    """
    Save the transformed and scaled data to the output directory.
    Preserves the original directory structure and filenames.
    
    Args:
        scaled_data (dict): Dictionary with structure {user: {trial: numpy_array}}
        input_dir (str): Original input directory (to get filenames)
        output_dir (str): Output directory where transformed data will be saved
    """
    # Create output directory structure
    output_kinematics_dir = os.path.join(output_dir, "kinematics", "AllGestures")
    os.makedirs(output_kinematics_dir, exist_ok=True)
    
    # Copy transcriptions directory if it exists
    input_transcriptions_dir = os.path.join(input_dir, "transcriptions")
    output_transcriptions_dir = os.path.join(output_dir, "transcriptions")
    
    if os.path.exists(input_transcriptions_dir):
        os.makedirs(output_transcriptions_dir, exist_ok=True)
        
        print("Copying transcription files...")
        for filename in os.listdir(input_transcriptions_dir):
            src = os.path.join(input_transcriptions_dir, filename)
            dst = os.path.join(output_transcriptions_dir, filename)
            
            # Read and write to preserve the file
            with open(src, 'r') as f:
                content = f.read()
            with open(dst, 'w') as f:
                f.write(content)
    
    # Get original filenames to preserve naming convention
    input_kinematics_dir = os.path.join(input_dir, "kinematics", "AllGestures")
    file_pattern = re.compile(r"(.*)_([B-I])(\d{3})\.txt")
    
    filename_map = {}
    for filename in os.listdir(input_kinematics_dir):
        match = file_pattern.match(filename)
        if match:
            prefix, user, trial_str = match.groups()
            trial = int(trial_str)
            filename_map[(user, trial)] = filename
    
    print("Saving transformed data...")
    for user in tqdm(sorted(scaled_data.keys())):
        for trial in sorted(scaled_data[user].keys()):
            data = scaled_data[user][trial]
            
            # Get original filename
            original_filename = filename_map.get((user, trial))
            if original_filename is None:
                print(f"Warning: Could not find original filename for user {user}, trial {trial}")
                continue
            
            output_path = os.path.join(output_kinematics_dir, original_filename)
            
            # Save as space-separated values to match original format
            np.savetxt(output_path, data, fmt='%.6f')
    
    print(f"\nTransformed data saved to: {output_dir}")


def save_statistics(statistics, output_dir, scaling_method='normalize', minmax_range=(-1, 1)):
    """
    Save the scaling statistics to a file for future reference.
    
    Args:
        statistics (dict): Dictionary with structure {user: {'mean': array, 'std': array, 'min': array, 'max': array}}
        output_dir (str): Output directory
        scaling_method (str): The scaling method used
        minmax_range (tuple): The range used for min-max scaling
    """
    stats_file = os.path.join(output_dir, "scaling_statistics.npz")
    
    # Prepare data for saving
    stats_dict = {
        'scaling_method': scaling_method,
        'minmax_range': np.array(minmax_range)
    }
    
    for user in statistics:
        stats_dict[f'{user}_mean'] = statistics[user]['mean']
        stats_dict[f'{user}_std'] = statistics[user]['std']
        stats_dict[f'{user}_min'] = statistics[user]['min']
        stats_dict[f'{user}_max'] = statistics[user]['max']
        stats_dict[f'{user}_range'] = statistics[user]['range']
    
    np.savez(stats_file, **stats_dict)
    print(f"Scaling statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JIGSAWS dataset with PSM transform and per-user scaling"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing the JIGSAWS dataset (e.g., dataset/Suturing)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where transformed data will be saved"
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="normalize",
        choices=["normalize", "minmax"],
        help="Scaling method: 'normalize' for z-score normalization, 'minmax' for min-max scaling (default: normalize)"
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=(-1, 1),
        metavar=("MIN", "MAX"),
        help="Target range for min-max scaling (default: -1 1)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task name for logging purposes"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Validate range for min-max scaling
    if args.scaling == "minmax" and args.range[0] >= args.range[1]:
        print("Error: Invalid range for min-max scaling. MIN must be less than MAX.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("JIGSAWS Dataset Preprocessing")
    print("=" * 70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Scaling method:   {args.scaling}")
    if args.scaling == "minmax":
        print(f"Target range:     [{args.range[0]}, {args.range[1]}]")
    if args.task:
        print(f"Task: {args.task}")
    print("=" * 70)
    print()
    
    # Step 1: Load kinematics data
    kinematics_data = load_kinematics_data(args.input_dir)
    print(f"Loaded data for {len(kinematics_data)} users\n")
    
    # Step 2: Apply PSM kinematics extraction transform
    transformed_data = apply_transform(kinematics_data)
    print(f"Transformed data from {kinematics_data[list(kinematics_data.keys())[0]][1].shape[1]} to {transformed_data[list(transformed_data.keys())[0]][1].shape[1]} features\n")
    
    # Step 3: Compute per-user statistics
    statistics = compute_per_user_statistics(transformed_data, args.scaling)
    print()
    
    # Step 4: Scale data
    scaled_data = scale_data(transformed_data, statistics, args.scaling, tuple(args.range))
    print()
    
    # Step 5: Save transformed data
    save_transformed_data(scaled_data, args.input_dir, args.output_dir)
    print()
    
    # Step 6: Save scaling statistics
    save_statistics(statistics, args.output_dir, args.scaling, tuple(args.range))
    
    print("\n" + "=" * 70)
    print("Preprocessing completed successfully!")
    print("=" * 70)
    print("\nYou can now use the transformed dataset by pointing the")
    print("KinematicsDataset to the output directory:")
    print(f"  dataset = KinematicsDataset(dir='{args.output_dir}', transform=None)")
    print("\nNote: Set transform=None since the data is already transformed.")
    if args.scaling == "minmax":
        print(f"Data has been scaled to range [{args.range[0]}, {args.range[1]}]")


if __name__ == "__main__":
    main()
