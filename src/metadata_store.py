"""
Simple JSON-based metadata storage for trajectory inputs.

This module provides a lightweight alternative to storing metadata in NetCDF files
during input generation. Metadata is saved as individual JSON files that can be
optionally injected into the merged NetCDF file after simulation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4


class MetadataStore:
    """
    Manages metadata storage as JSON files in a directory.

    Each trajectory's metadata is stored as a separate JSON file named
    with the trajectory ID. This allows for easy inspection, editing, and
    optional injection into NetCDF files after simulation.

    Example usage:
        # During input generation:
        store = MetadataStore.create(output_dir)
        store.save_metadata(traj_id, metadata_dict)

        # Later, for reading/injection:
        store = MetadataStore(output_dir)
        all_metadata = store.load_all_metadata()
    """

    def __init__(self, output_dir: str, metadata_subdir: str = "metadata"):
        """
        Initialize metadata store.

        Args:
            output_dir: Base directory where metadata will be stored
            metadata_subdir: Subdirectory name for metadata files (default: "metadata")
        """
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir / metadata_subdir

    @classmethod
    def create(cls, output_dir: str, metadata_subdir: str = "metadata") -> "MetadataStore":
        """
        Create a new metadata store directory.

        Args:
            output_dir: Base directory where metadata will be stored
            metadata_subdir: Subdirectory name for metadata files

        Returns:
            MetadataStore instance
        """
        store = cls(output_dir, metadata_subdir)
        store.metadata_dir.mkdir(parents=True, exist_ok=True)
        return store

    def save_metadata(self, traj_id: str, metadata: Dict[str, Any]) -> str:
        """
        Save metadata for a single trajectory.

        Args:
            traj_id: Unique trajectory identifier
            metadata: Dictionary containing trajectory metadata

        Returns:
            Path to the saved metadata file
        """
        # Ensure metadata directory exists
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Add trajectory ID to metadata if not present
        if "traj_id" not in metadata:
            metadata["traj_id"] = traj_id

        # Save as JSON
        filepath = self.metadata_dir / f"{traj_id}.json"
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(filepath)

    def load_metadata(self, traj_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a specific trajectory.

        Args:
            traj_id: Trajectory identifier

        Returns:
            Metadata dictionary, or None if not found
        """
        filepath = self.metadata_dir / f"{traj_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def load_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Load metadata for all trajectories in the store.

        Returns:
            List of metadata dictionaries, sorted by filename
        """
        if not self.metadata_dir.exists():
            return []

        metadata_list = []
        for filepath in sorted(self.metadata_dir.glob("*.json")):
            with open(filepath, 'r') as f:
                metadata_list.append(json.load(f))

        return metadata_list

    def get_trajectory_count(self) -> int:
        """
        Get the number of trajectories with metadata.

        Returns:
            Count of JSON metadata files
        """
        if not self.metadata_dir.exists():
            return 0
        return len(list(self.metadata_dir.glob("*.json")))

    def delete_metadata(self, traj_id: str) -> bool:
        """
        Delete metadata for a specific trajectory.

        Args:
            traj_id: Trajectory identifier

        Returns:
            True if deleted, False if not found
        """
        filepath = self.metadata_dir / f"{traj_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def clear_all_metadata(self):
        """
        Delete all metadata files in the store.

        Warning: This cannot be undone!
        """
        if self.metadata_dir.exists():
            for filepath in self.metadata_dir.glob("*.json"):
                filepath.unlink()

    def exists(self) -> bool:
        """
        Check if the metadata directory exists.

        Returns:
            True if directory exists
        """
        return self.metadata_dir.exists()


# Convenience functions for quick operations

def save_trajectory_metadata(
    output_dir: str,
    traj_id: str,
    metadata: Dict[str, Any],
    metadata_subdir: str = "metadata"
) -> str:
    """
    Convenience function to save metadata without creating a store instance.

    Args:
        output_dir: Base directory
        traj_id: Trajectory identifier
        metadata: Metadata dictionary
        metadata_subdir: Subdirectory name

    Returns:
        Path to saved metadata file
    """
    store = MetadataStore.create(output_dir, metadata_subdir)
    return store.save_metadata(traj_id, metadata)


def load_all_trajectory_metadata(
    output_dir: str,
    metadata_subdir: str = "metadata"
) -> List[Dict[str, Any]]:
    """
    Convenience function to load all metadata without creating a store instance.

    Args:
        output_dir: Base directory
        metadata_subdir: Subdirectory name

    Returns:
        List of metadata dictionaries
    """
    store = MetadataStore(output_dir, metadata_subdir)
    return store.load_all_metadata()


def get_metadata_count(
    output_dir: str,
    metadata_subdir: str = "metadata"
) -> int:
    """
    Convenience function to count metadata files.

    Args:
        output_dir: Base directory
        metadata_subdir: Subdirectory name

    Returns:
        Number of metadata files
    """
    store = MetadataStore(output_dir, metadata_subdir)
    return store.get_trajectory_count()
