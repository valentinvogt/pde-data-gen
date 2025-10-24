#!/usr/bin/env python
"""
Inject trajectory metadata from JSON files into a merged NetCDF dataset.

This script reads metadata stored as individual JSON files and injects them
into the consolidated NetCDF file after simulation outputs have been merged.

Usage:
    python scripts/inject_metadata.py <dataset_nc_file> [--metadata-dir <path>]

Example:
    python scripts/inject_metadata.py data/bruss/my_dataset/_dataset.nc
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import xarray as xr
import numpy as np

from src.metadata_store import MetadataStore


def match_trajectories_to_metadata(
    ds: xr.Dataset,
    metadata_list: List[Dict[str, Any]],
    match_field: str = "input_file"
) -> List[Dict[str, Any]]:
    """
    Match trajectories in NetCDF to metadata files.

    Args:
        ds: xarray Dataset with trajectory dimension
        metadata_list: List of metadata dictionaries
        match_field: Field in both NetCDF and metadata to match on

    Returns:
        Ordered list of metadata dicts matching trajectory order in NetCDF
    """
    n_trajectories = len(ds.trajectory)

    # Check if we need to match or if we can use direct indexing
    if match_field in ds.variables:
        # Build lookup from match field to metadata
        metadata_lookup = {meta.get(match_field): meta for meta in metadata_list}

        # Match each trajectory
        ordered_metadata = []
        for i in range(n_trajectories):
            key = str(ds[match_field][i].values)
            if key in metadata_lookup:
                ordered_metadata.append(metadata_lookup[key])
            else:
                # Create empty metadata for unmatched trajectory
                print(f"Warning: No metadata found for trajectory {i} ({match_field}={key})")
                ordered_metadata.append({"traj_id": f"unknown_{i}"})
    else:
        # No match field in dataset, just use order
        print(f"Warning: Match field '{match_field}' not found in dataset. Using order.")
        ordered_metadata = metadata_list[:n_trajectories]

        # Warn if counts don't match
        if len(metadata_list) != n_trajectories:
            print(f"Warning: {n_trajectories} trajectories but {len(metadata_list)} metadata files")

    return ordered_metadata


def inject_metadata_into_dataset(
    ds: xr.Dataset,
    metadata_list: List[Dict[str, Any]]
) -> xr.Dataset:
    """
    Inject metadata from JSON files into xarray Dataset.

    Args:
        ds: Input xarray Dataset with trajectory dimension
        metadata_list: List of metadata dictionaries (must match trajectory order)

    Returns:
        Dataset with metadata added as variables
    """
    n_trajectories = len(ds.trajectory)

    if len(metadata_list) != n_trajectories:
        raise ValueError(
            f"Metadata count ({len(metadata_list)}) doesn't match "
            f"trajectory count ({n_trajectories})"
        )

    # Extract scalar fields from metadata
    scalar_fields = {
        "traj_id": "",
        "model": "",
        "A": 0.0,
        "B": 0.0,
        "Du": 0.0,
        "Dv": 0.0,
        "Nx": 0,
        "dx": 0.0,
        "Nt": 0,
        "dt": 0.0,
        "n_snapshots": 0,
        "output_file": "",
        "input_file": "",
        "random_seed": 0,
        "dataset_id": "",
        "ic_type": "",
    }

    # Complex fields to store as JSON
    json_fields = ["initial_condition", "original_point"]

    # Build arrays for each field
    for field, default in scalar_fields.items():
        values = []
        for meta in metadata_list:
            value = meta.get(field, default)

            # Handle special cases
            if field == "output_file" and "filename" in meta and field not in meta:
                value = meta["filename"]

            values.append(value)

        # Add as coordinate or data variable
        ds[field] = xr.DataArray(values, dims=["trajectory"])

    # Handle JSON fields
    for field in json_fields:
        values = []
        for meta in metadata_list:
            value = meta.get(field, {})
            values.append(json.dumps(value) if isinstance(value, dict) else value)

        ds[field] = xr.DataArray(values, dims=["trajectory"])

    return ds


def inject_metadata(
    dataset_file: str,
    metadata_dir: str = None,
    output_file: str = None,
    match_field: str = "input_file"
):
    """
    Inject metadata from JSON files into a NetCDF dataset.

    Args:
        dataset_file: Path to the merged NetCDF dataset
        metadata_dir: Directory containing metadata JSON files (default: same dir as dataset)
        output_file: Output file path (default: overwrite input)
        match_field: Field to use for matching trajectories to metadata
    """
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # Determine metadata directory
    if metadata_dir is None:
        dataset_dir = os.path.dirname(dataset_file)
        metadata_dir = os.path.join(dataset_dir, "metadata")

    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    print(f"Loading dataset from: {dataset_file}")
    print(f"Loading metadata from: {metadata_dir}")

    # Load metadata
    store = MetadataStore(os.path.dirname(metadata_dir), os.path.basename(metadata_dir))
    metadata_list = store.load_all_metadata()

    if not metadata_list:
        print("Warning: No metadata files found!")
        return

    print(f"Found {len(metadata_list)} metadata files")

    # Load dataset (load into memory to avoid conflicts)
    ds = xr.load_dataset(dataset_file)
    n_trajectories = len(ds.trajectory)
    print(f"Dataset has {n_trajectories} trajectories")

    # Match trajectories to metadata
    ordered_metadata = match_trajectories_to_metadata(ds, metadata_list, match_field)

    # Inject metadata
    print("Injecting metadata into dataset...")
    ds = inject_metadata_into_dataset(ds, ordered_metadata)

    # Save to output file
    output_path = output_file or dataset_file
    print(f"Saving dataset to: {output_path}")

    # Use compression for efficiency
    encoding = {}
    if "data" in ds.variables:
        encoding["data"] = {"zlib": True, "complevel": 4}

    ds.to_netcdf(output_path, encoding=encoding)

    print("Metadata injection complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Inject trajectory metadata from JSON files into NetCDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inject metadata into merged dataset (metadata dir auto-detected)
  python scripts/inject_metadata.py data/bruss/dataset1/_dataset.nc

  # Specify custom metadata directory
  python scripts/inject_metadata.py data/bruss/dataset1/_dataset.nc --metadata-dir data/bruss/dataset1/metadata

  # Save to a different output file
  python scripts/inject_metadata.py data/bruss/dataset1/_dataset.nc -o data/bruss/dataset1/_dataset_with_metadata.nc
        """
    )

    parser.add_argument(
        "dataset_file",
        help="Path to the merged NetCDF dataset file"
    )

    parser.add_argument(
        "--metadata-dir", "-m",
        help="Directory containing metadata JSON files (default: <dataset_dir>/metadata)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: overwrite input file)"
    )

    parser.add_argument(
        "--match-field",
        default="input_file",
        help="Field to use for matching trajectories (default: input_file)"
    )

    args = parser.parse_args()

    try:
        inject_metadata(
            args.dataset_file,
            args.metadata_dir,
            args.output,
            args.match_field
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
