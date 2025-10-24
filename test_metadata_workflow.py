#!/usr/bin/env python
"""
Simple test to verify the metadata workflow works correctly.

This script tests:
1. Creating metadata store
2. Saving trajectory metadata
3. Loading metadata back
4. Injecting metadata into a mock NetCDF dataset
"""

import os
import tempfile
import shutil
from pathlib import Path
import json

import numpy as np
import xarray as xr

from src.metadata_store import MetadataStore, save_trajectory_metadata, load_all_trajectory_metadata


def test_metadata_store():
    """Test basic metadata store operations."""
    print("=" * 60)
    print("TEST 1: Metadata Store Operations")
    print("=" * 60)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCreated temp directory: {tmpdir}")

        # Create metadata store
        store = MetadataStore.create(tmpdir)
        print(f"Created metadata store at: {store.metadata_dir}")

        # Save some metadata
        metadata1 = {
            "traj_id": "test-uuid-1",
            "A": 1.0,
            "B": 2.5,
            "Du": 0.1,
            "Dv": 0.05,
            "model": "bruss",
        }
        store.save_metadata("test-uuid-1", metadata1)
        print(f"Saved metadata for trajectory 1")

        metadata2 = {
            "traj_id": "test-uuid-2",
            "A": 1.5,
            "B": 3.0,
            "Du": 0.15,
            "Dv": 0.08,
            "model": "bruss",
        }
        store.save_metadata("test-uuid-2", metadata2)
        print(f"Saved metadata for trajectory 2")

        # Check count
        count = store.get_trajectory_count()
        print(f"\nTrajectory count: {count}")
        assert count == 2, f"Expected 2 trajectories, got {count}"

        # Load single metadata
        loaded = store.load_metadata("test-uuid-1")
        print(f"\nLoaded trajectory 1 metadata:")
        print(f"  A={loaded['A']}, B={loaded['B']}")
        assert loaded["A"] == 1.0

        # Load all metadata
        all_metadata = store.load_all_metadata()
        print(f"\nLoaded all metadata: {len(all_metadata)} trajectories")
        assert len(all_metadata) == 2

        print("\n✅ TEST 1 PASSED")


def test_convenience_functions():
    """Test convenience functions."""
    print("\n" + "=" * 60)
    print("TEST 2: Convenience Functions")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temp directory: {tmpdir}")

        # Save using convenience function
        metadata = {
            "traj_id": "test-uuid-3",
            "A": 2.0,
            "B": 4.0,
            "model": "gray_scott",
        }
        filepath = save_trajectory_metadata(tmpdir, "test-uuid-3", metadata)
        print(f"Saved metadata to: {filepath}")

        # Load using convenience function
        all_metadata = load_all_trajectory_metadata(tmpdir)
        print(f"Loaded {len(all_metadata)} trajectories")
        assert len(all_metadata) == 1
        assert all_metadata[0]["A"] == 2.0

        print("\n✅ TEST 2 PASSED")


def test_metadata_injection():
    """Test metadata injection into NetCDF."""
    print("\n" + "=" * 60)
    print("TEST 3: Metadata Injection into NetCDF")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temp directory: {tmpdir}")

        # Create mock NetCDF dataset with trajectory data
        n_traj = 3
        n_snap = 10
        nx = ny = 64

        # Create dataset with simulation data
        data = np.random.randn(n_traj, n_snap, 2, nx, ny)
        ds = xr.Dataset(
            data_vars={
                "data": (["trajectory", "snapshot", "component", "x", "y"], data),
                "input_file": (["trajectory"], [f"input_{i}.nc" for i in range(n_traj)]),
            },
            coords={
                "trajectory": np.arange(n_traj),
                "snapshot": np.arange(n_snap),
                "component": ["u", "v"],
                "x": np.arange(nx),
                "y": np.arange(ny),
            },
        )

        # Save dataset
        dataset_file = os.path.join(tmpdir, "_dataset.nc")
        ds.to_netcdf(dataset_file)
        print(f"Created mock dataset: {dataset_file}")
        print(f"  Shape: {n_traj} trajectories, {n_snap} snapshots, {nx}x{ny} grid")

        # Create metadata for each trajectory
        store = MetadataStore.create(tmpdir)
        for i in range(n_traj):
            metadata = {
                "traj_id": f"traj-{i}",
                "A": 1.0 + i * 0.5,
                "B": 2.0 + i * 0.5,
                "Du": 0.1,
                "Dv": 0.05,
                "Nx": nx,
                "dx": 0.5,
                "Nt": 1000,
                "dt": 0.01,
                "n_snapshots": n_snap,
                "model": "bruss",
                "dataset_id": "test_dataset",
                "random_seed": 42 + i,
                "input_file": f"input_{i}.nc",
                "input_file": f"input_{i}.nc",  # This field is used for matching
                "filename": f"output_{i}.nc",
                "ic_type": "random_uniform",
                "initial_condition": {"type": "random_uniform", "params": {}},
            }
            store.save_metadata(f"traj-{i}", metadata)

        print(f"Created {n_traj} metadata files")

        # Inject metadata
        from scripts.inject_metadata import inject_metadata

        inject_metadata(
            dataset_file=dataset_file,
            metadata_dir=str(store.metadata_dir),
            match_field="input_file",
        )
        print("\nInjected metadata into dataset")

        # Verify injection
        ds_with_meta = xr.open_dataset(dataset_file)
        print("\nVerifying injected metadata:")
        print(f"  Variables in dataset: {list(ds_with_meta.data_vars.keys())}")

        # Check that metadata variables exist
        assert "A" in ds_with_meta.variables
        assert "B" in ds_with_meta.variables
        assert "Du" in ds_with_meta.variables
        assert "model" in ds_with_meta.variables

        # Check values
        print(f"\n  A values: {ds_with_meta['A'].values}")
        print(f"  B values: {ds_with_meta['B'].values}")

        assert ds_with_meta["A"][0].values == 1.0
        assert ds_with_meta["A"][1].values == 1.5
        assert ds_with_meta["A"][2].values == 2.0

        print("\n✅ TEST 3 PASSED")


def test_trajectory_dataset_integration():
    """Test integration with TrajectoryDataset."""
    print("\n" + "=" * 60)
    print("TEST 4: TrajectoryDataset Integration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temp directory: {tmpdir}")

        # Create dataset with injected metadata
        n_traj = 2
        n_snap = 5
        nx = ny = 32

        data = np.random.randn(n_traj, n_snap, 2, nx, ny)
        ds = xr.Dataset(
            data_vars={
                "data": (["trajectory", "snapshot", "component", "x", "y"], data),
                "input_file": (["trajectory"], [f"input_{i}.nc" for i in range(n_traj)]),
                "A": (["trajectory"], [1.0, 1.5]),
                "B": (["trajectory"], [2.0, 2.5]),
                "Du": (["trajectory"], [0.1, 0.1]),
                "Dv": (["trajectory"], [0.05, 0.05]),
                "model": (["trajectory"], ["bruss", "bruss"]),
                "traj_id": (["trajectory"], ["traj-0", "traj-1"]),
            },
            coords={
                "trajectory": np.arange(n_traj),
                "snapshot": np.arange(n_snap),
                "component": ["u", "v"],
                "x": np.arange(nx),
                "y": np.arange(ny),
            },
        )

        dataset_file = os.path.join(tmpdir, "_dataset.nc")
        ds.to_netcdf(dataset_file)
        print(f"Created dataset with metadata: {dataset_file}")

        # Open with TrajectoryDataset
        from src.trajectory_dataset import TrajectoryDataset

        traj_ds = TrajectoryDataset.open(dataset_file, lazy=False)
        print("\nOpened with TrajectoryDataset")

        # Get DataFrame
        df = traj_ds.df
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Query
        subset = traj_ds.query(A=1.0)
        print(f"\nQueried for A=1.0: {len(subset)} trajectories")
        assert len(subset) == 1

        # Get data
        data_0 = traj_ds.get_data(0)
        print(f"\nData shape for trajectory 0: {data_0.shape}")
        assert data_0.shape == (n_snap, 2, nx, ny)

        print("\n✅ TEST 4 PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING METADATA WORKFLOW")
    print("=" * 60)

    try:
        test_metadata_store()
        test_convenience_functions()
        test_metadata_injection()
        test_trajectory_dataset_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe metadata workflow is working correctly.")
        print("You can now use it in your pipeline.")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
