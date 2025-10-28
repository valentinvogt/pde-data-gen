#!/usr/bin/env python
"""
Consolidate individual NetCDF output files into a single dataset and inject metadata.

This script:
1. Finds all *_output.nc files in the dataset directory
2. Merges them into _dataset.nc using xarray
3. Injects metadata from JSON files into the merged dataset

Usage:
    python scripts/consolidate_nc.py <dataset_directory>

Example:
    python scripts/consolidate_nc.py data/bruss/my_dataset
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from src.metadata_store import MetadataStore


def find_output_files(dataset_dir: str) -> List[str]:
    """
    Find all *_output.nc files in the dataset directory.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        List of paths to output files
    """
    output_files = []
    for file in Path(dataset_dir).glob("*_output.nc"):
        output_files.append(str(file))

    return sorted(output_files)


def extract_input_file(output_file: str) -> str:
    """
    Extract the input file path from the output filename.

    Args:
        output_file: Path to output file (e.g., "path/to/uuid_output.nc")

    Returns:
        Input file path (e.g., "path/to/uuid.nc")
    """
    # Remove "_output.nc" suffix and replace with ".nc" to get the input file path
    input_file = output_file.replace("_output.nc", ".nc")
    return input_file


def merge_output_files(output_files: List[str], output_path: str):
    """
    Merge multiple output NetCDF files into a single dataset.

    Args:
        output_files: List of paths to output files
        output_path: Path for the merged output file
    """
    datasets = []
    input_files = []

    for i, file in enumerate(output_files):
        try:
            # Load the output file
            ds = xr.open_dataset(file)

            # Extract metadata from NetCDF attributes
            n_snapshots = ds.attrs.get("number_snapshots", None)
            n_x = ds.attrs.get("n_x", None)
            n_y = ds.attrs.get("n_y", None)
            n_coupled = ds.attrs.get("n_coupled", None)

            if n_snapshots is None:
                print(f"    Warning: No 'number_snapshots' attribute in {file}, skipping")
                continue

            # Get the data variable
            # Assuming shape: [member, snapshot, x_size_and_boundary, n_coupled_and_y_size_and_boundary]
            data = ds["data"].values

            # Take first member if there are multiple
            if data.ndim == 4:
                data = data[0]  # Shape: [snapshot, x, y*n_coupled]

            # Reshape to separate components and spatial dimensions
            # Current shape: [snapshot, x_size_and_boundary, n_coupled_and_y_size_and_boundary]
            # Target shape: [snapshot, component, x, y]
            if n_coupled and n_y:
                x_dim = data.shape[1]
                y_dim = (data.shape[2] // n_coupled)

                # Reshape: [snapshot, x, n_coupled, y]
                data_reshaped = data.reshape(n_snapshots, x_dim, n_coupled, y_dim)

                # Transpose to: [snapshot, n_coupled, x, y]
                data_reshaped = data_reshaped.transpose(0, 2, 1, 3)
            else:
                data_reshaped = data

            # Create a dataset for this trajectory
            traj_ds = xr.Dataset(
                {
                    "data": (["snapshot", "component", "x", "y"], data_reshaped)
                },
                coords={
                    "snapshot": np.arange(data_reshaped.shape[0]),
                    "component": ["u", "v"] if n_coupled == 2 else [f"c{i}" for i in range(n_coupled)],
                    "x": np.arange(data_reshaped.shape[2]),
                    "y": np.arange(data_reshaped.shape[3]),
                }
            )

            datasets.append(traj_ds)
            input_files.append(extract_input_file(file))

        except Exception as e:
            print(f"    Error loading {file}: {e}")
            continue

    if not datasets:
        print("Error: No valid output files found!")
        sys.exit(1)

    # Concatenate along a new 'trajectory' dimension
    merged_ds = xr.concat(datasets, dim="trajectory")

    # Add input_file variable for matching with metadata
    merged_ds["input_file"] = xr.DataArray(
        input_files,
        dims=["trajectory"]
    )

    # Add trajectory coordinate
    merged_ds["trajectory"] = np.arange(len(datasets))

    # Save merged dataset
    encoding = {"data": {"zlib": True, "complevel": 4}}
    merged_ds.to_netcdf(output_path, encoding=encoding)

    print(f"Merged dataset saved: {output_path}")


def inject_metadata_into_dataset(dataset_file: str, metadata_dir: str):
    """
    Inject metadata from JSON files into the merged dataset.

    Args:
        dataset_file: Path to the merged NetCDF file
        metadata_dir: Directory containing metadata JSON files
    """
    from scripts.inject_metadata import inject_metadata

    print(f"Injecting metadata from {metadata_dir}")
    inject_metadata(
        dataset_file=dataset_file,
        metadata_dir=metadata_dir,
        match_field="input_file"
    )

def consolidate_outputs(dataset_dir: str):
    """
    Main consolidation function: merge outputs and inject metadata.

    Args:
        dataset_dir: Path to the dataset directory
    """
    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist")
        sys.exit(1)

    print(f"Consolidating outputs in: {dataset_dir}")

    # Find output files
    output_files = find_output_files(dataset_dir)

    if not output_files:
        print(f"Error: No *_output.nc files found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(output_files)} output files")

    # Merge output files
    merged_file = os.path.join(dataset_dir, "_dataset.nc")
    merge_output_files(output_files, merged_file)

    # Inject metadata
    metadata_dir = os.path.join(dataset_dir, "metadata")
    if os.path.exists(metadata_dir):
        store = MetadataStore(os.path.dirname(metadata_dir), "metadata")
        metadata_count = store.get_trajectory_count()

        if metadata_count > 0:
            inject_metadata_into_dataset(merged_file, metadata_dir)
        else:
            print(f"\nWarning: No metadata files found in {metadata_dir}")
            print("Skipping metadata.")
    else:
        print(f"\nWarning: Metadata directory not found: {metadata_dir}")
        print("Skipping metadata.")

    print("Consolidation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate NetCDF output files and inject metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Consolidate outputs in a dataset directory
  python scripts/consolidate_nc.py data/bruss/my_dataset

  # The script will:
  # 1. Find all *_output.nc files
  # 2. Merge them into _dataset.nc
  # 3. Inject metadata from metadata/*.json files
        """
    )

    parser.add_argument(
        "dataset_dir",
        help="Path to the dataset directory containing *_output.nc files"
    )

    args = parser.parse_args()

    consolidate_outputs(args.dataset_dir)
