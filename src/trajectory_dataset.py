"""
Trajectory dataset management using xarray.
"""

import xarray as xr
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, Optional, Union


class TrajectoryDataset:
    """
    Interface for creating and reading trajectory datasets.

    Example usage:
        # Writing:
        ds = TrajectoryDataset.create("path/to/dataset.nc", config={...})
        ds.add_trajectory(metadata={...})
        ds.save()

        # Reading:
        ds = TrajectoryDataset.open("path/to/dataset.nc")
        df = ds.to_dataframe()
        data = ds.get_data(0)  # Get first trajectory
        filtered = ds.query(A=0.5, model="fhn")  # Filter trajectories
    """

    def __init__(self, dataset: xr.Dataset, filepath: str):
        """
        Internal constructor. Use TrajectoryDataset.create() or .open() instead.
        """
        self.ds = dataset
        self.filepath = filepath
        self._df_cache = None

    @classmethod
    def create(
        cls, filepath: str, config: Optional[Dict[str, Any]] = None
    ) -> "TrajectoryDataset":
        """
        Create a new trajectory dataset file.

        Args:
            filepath: Path where the dataset will be saved
            config: Optional configuration dictionary to store as global attributes

        Returns:
            TrajectoryDataset instance ready for adding trajectories
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Create empty dataset with trajectory dimension and all metadata variables pre-initialized
        # This ensures that when we later append trajectories, all coordinates are present
        ds = xr.Dataset(
            data_vars={
                'traj_id': (['trajectory'], []),
                'model': (['trajectory'], []),
                'output_file': (['trajectory'], []),
                'initial_condition': (['trajectory'], []),
                'original_point': (['trajectory'], []),
                'A': (['trajectory'], []),
                'B': (['trajectory'], []),
                'Du': (['trajectory'], []),
                'Dv': (['trajectory'], []),
                'dx': (['trajectory'], []),
                'dt': (['trajectory'], []),
                'Nx': (['trajectory'], []),
                'Nt': (['trajectory'], []),
                'n_snapshots': (['trajectory'], []),
                'random_seed': (['trajectory'], []),
            },
            coords={'trajectory': []}
        )

        # Store configuration as global attributes
        if config:
            for key, value in config.items():
                if isinstance(value, bool):
                    # NetCDF doesn't support booleans, convert to int
                    ds.attrs[key] = int(value)
                elif isinstance(value, (str, int, float)):
                    ds.attrs[key] = value
                else:
                    # Store complex types as JSON
                    ds.attrs[key] = json.dumps(value)

        return cls(ds, filepath)

    @classmethod
    def open(cls, filepath: str, lazy: bool = True) -> "TrajectoryDataset":
        """
        Open an existing trajectory dataset.

        Args:
            filepath: Path to the dataset file
            lazy: If True, use lazy loading (default). Set False for eager loading.

        Returns:
            TrajectoryDataset instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        if lazy:
            ds = xr.open_dataset(filepath, chunks={"trajectory": 16})
        else:
            ds = xr.load_dataset(filepath)

        return cls(ds, filepath)

    def add_trajectory(
        self, metadata: Dict[str, Any], data: Optional[np.ndarray] = None
    ):
        """
        Add a trajectory with its metadata (and optionally simulation data).

        Args:
            metadata: Dictionary containing trajectory metadata
                Required keys: traj_id, model, A, B, Du, Dv, Nx, dx, Nt, dt,
                              n_snapshots, output_file, random_seed
                Optional: initial_condition (dict), original_point (dict)
            data: Optional simulation data array of shape (n_snapshots, x, y, components)
        """
        # Define scalar metadata fields with default values
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
            "random_seed": 0,
        }

        # Append each scalar field
        for field, default in scalar_fields.items():
            value = metadata.get(field, default)

            # Initialize variable if it doesn't exist (for legacy file compatibility)
            if field not in self.ds.data_vars:
                self.ds[field] = xr.DataArray([], dims=["trajectory"])

            # Append value
            new_value = xr.DataArray([value], dims=["trajectory"])
            self.ds[field] = xr.concat([self.ds[field], new_value], dim="trajectory")

        # Handle complex fields (store as JSON)
        ic_json = json.dumps(metadata.get("initial_condition", {}))
        if "initial_condition" not in self.ds.data_vars:
            self.ds["initial_condition"] = xr.DataArray([], dims=["trajectory"])
        new_ic = xr.DataArray([ic_json], dims=["trajectory"])
        self.ds["initial_condition"] = xr.concat(
            [self.ds["initial_condition"], new_ic], dim="trajectory"
        )

        op_json = json.dumps(metadata.get("original_point", {}))
        if "original_point" not in self.ds.data_vars:
            self.ds["original_point"] = xr.DataArray([], dims=["trajectory"])
        new_op = xr.DataArray([op_json], dims=["trajectory"])
        self.ds["original_point"] = xr.concat(
            [self.ds["original_point"], new_op], dim="trajectory"
        )

        # Add simulation data if provided
        if data is not None:
            if "data" not in self.ds.data_vars:
                # Initialize data variable with appropriate dimensions
                n_snap, nx, ny, n_comp = data.shape
                self.ds = self.ds.assign_coords(
                    {
                        "snapshot": np.arange(n_snap),
                        "x": np.arange(nx),
                        "y": np.arange(ny),
                    }
                )

                # Create empty data array
                self.ds["data"] = xr.DataArray(
                    np.empty((0, n_snap, nx, ny, n_comp)),
                    dims=["trajectory", "snapshot", "x", "y", "component"],
                    attrs={"description": "Simulation data (u, v)"},
                )

            # Append data
            new_data = xr.DataArray(
                data[np.newaxis, ...],
                dims=["trajectory", "snapshot", "x", "y", "component"],
            )
            self.ds["data"] = xr.concat([self.ds["data"], new_data], dim="trajectory")

        # Update trajectory coordinate
        self.ds = self.ds.assign_coords(
            trajectory=np.arange(len(self.ds.trajectory) + 1)
        )

        # Clear dataframe cache
        self._df_cache = None

    def add_data(self, traj_index: int, data: np.ndarray):
        """
        Add simulation data for an existing trajectory.

        Args:
            traj_index: Index of the trajectory
            data: Simulation data array of shape (n_snapshots, x, y, components)
        """
        if "data" not in self.ds.data_vars:
            # Initialize data variable
            n_traj = len(self.ds.trajectory)
            n_snap, nx, ny, n_comp = data.shape

            self.ds = self.ds.assign_coords(
                {
                    "snapshot": np.arange(n_snap),
                    "x": np.arange(nx),
                    "y": np.arange(ny),
                }
            )

            # Create empty data array for all trajectories
            self.ds["data"] = xr.DataArray(
                np.empty((n_traj, n_snap, nx, ny, n_comp)),
                dims=["trajectory", "snapshot", "x", "y", "component"],
                attrs={"description": "Simulation data (u, v)"},
            )

        # Set data for this trajectory
        self.ds["data"][traj_index] = data

    def save(self, filepath: Optional[str] = None):
        """
        Save the dataset to disk.

        Args:
            filepath: Optional path. If not provided, uses the original filepath.
        """
        save_path = filepath or self.filepath
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Save with compression
        encoding = {}
        if "data" in self.ds.data_vars:
            encoding["data"] = {"zlib": True, "complevel": 4}

        self.ds.to_netcdf(save_path, encoding=encoding)

    def close(self):
        """Close the dataset (for lazy-loaded datasets)."""
        if hasattr(self.ds, "close"):
            self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save if this was a newly created dataset
        if exc_type is None and len(self.ds.trajectory) > 0:
            self.save()
        self.close()

    def to_dataframe(self, include_json: bool = False) -> pd.DataFrame:
        """
        Convert trajectory metadata to a pandas DataFrame.

        Args:
            include_json: If True, parse JSON fields into dict columns

        Returns:
            DataFrame with one row per trajectory
        """
        if self._df_cache is not None:
            return self._df_cache

        # Extract all scalar variables
        df_dict = {}
        for var in self.ds.data_vars:
            if var not in ["data", "time"] and len(self.ds[var].dims) == 1:
                df_dict[var] = self.ds[var].values

        df = pd.DataFrame(df_dict)
        df["idx"] = df.index

        # Parse JSON fields if requested
        if include_json:
            if "initial_condition" in df.columns:
                df["ic_dict"] = df["initial_condition"].apply(
                    lambda x: json.loads(x) if x and x != "{}" else {}
                )
            if "original_point" in df.columns:
                df["original_point_dict"] = df["original_point"].apply(
                    lambda x: json.loads(x) if x and x != "{}" else {}
                )

        self._df_cache = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Convenience property for to_dataframe()."""
        return self.to_dataframe()

    def get_data(self, trajectory: Union[int, pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Get simulation data for a specific trajectory.

        Args:
            trajectory: Can be:
                - int: trajectory index
                - pd.Series: row from dataframe
                - pd.DataFrame: single-row dataframe

        Returns:
            Simulation data array of shape (n_snapshots, x, y, components)
        """
        # Handle different input types
        if isinstance(trajectory, (int, np.integer)):
            idx = int(trajectory)
        elif isinstance(trajectory, pd.DataFrame):
            if len(trajectory) != 1:
                raise ValueError("DataFrame must have exactly one row")
            idx = int(trajectory.iloc[0]["idx"])
        elif isinstance(trajectory, pd.Series):
            idx = int(trajectory["idx"])
        else:
            raise TypeError(f"Unsupported trajectory type: {type(trajectory)}")

        if "data" not in self.ds.data_vars:
            raise ValueError("No simulation data in dataset")

        return self.ds["data"][idx].values

    def query(self, **conditions) -> pd.DataFrame:
        """
        Query trajectories by metadata conditions.

        Args:
            **conditions: Field-value pairs to filter by (e.g., A=0.5, model="fhn")

        Returns:
            Filtered DataFrame

        Example:
            df = ds.query(A=0.5, model="fhn")
        """
        df = self.to_dataframe()
        mask = np.ones(len(df), dtype=bool)

        for key, value in conditions.items():
            if key not in df.columns:
                raise KeyError(f"Field '{key}' not found in dataset")
            mask &= df[key] == value

        return df[mask]

    def get_trajectory_count(self) -> int:
        """Get the number of trajectories in the dataset."""
        return len(self.ds.trajectory)

    def add_column(self, name: str, values: np.ndarray, description: str = ""):
        """
        Add a new metadata column to the dataset.

        Args:
            name: Name of the new column
            values: Array of values (must match number of trajectories)
            description: Optional description for the column
        """
        if len(values) != len(self.ds.trajectory):
            raise ValueError(
                f"Length of values ({len(values)}) must match number of "
                f"trajectories ({len(self.ds.trajectory)})"
            )

        self.ds[name] = xr.DataArray(
            values, dims=["trajectory"], attrs={"description": description}
        )

        # Clear dataframe cache
        self._df_cache = None

    def filter(self, df: pd.DataFrame) -> "TrajectoryDataset":
        """
        Create a new dataset with only the trajectories in the given dataframe.

        Args:
            df: DataFrame with 'idx' column indicating which trajectories to keep

        Returns:
            New TrajectoryDataset with filtered trajectories
        """
        if "idx" not in df.columns:
            raise ValueError("DataFrame must have 'idx' column")

        indices = df["idx"].values
        filtered_ds = self.ds.isel(trajectory=indices)

        return TrajectoryDataset(filtered_ds, self.filepath)


# Backward compatibility functions


def create_metadata_file(output_dir: str, config: Dict[str, Any]) -> str:
    """
    Create a metadata file for a set of trajectories.

    This is a compatibility function for existing code.

    Args:
        output_dir: Directory where the file will be stored
        config: Configuration dictionary for the trajectories

    Returns:
        Path to the created file
    """
    filepath = os.path.join(output_dir, "_dataset.nc")
    ds = TrajectoryDataset.create(filepath, config)
    ds.save()
    return filepath


def get_dataset(
    model: str,
    ds_id: str,
    directory_var: str = "WORKDIR",
    dataset_file: str = "_dataset.nc",
) -> tuple:
    """
    Load a dataset based on model and dataset ID from the environment directory.

    This is a compatibility function for existing analysis code.

    Args:
        model: Model identifier string
        ds_id: Dataset identifier string
        directory_var: Name of the env variable containing the directory
        dataset_file: Name of the dataset file (default: "_dataset.nc")

    Returns:
        Tuple[TrajectoryDataset, str]: Dataset object and output directory path
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = os.getenv(directory_var)
    if not data_dir:
        raise ValueError(f"Environment variable {directory_var} not set")

    output_dir = os.path.join(data_dir, "out")
    dataset_path = os.path.join(data_dir, "data", model, ds_id, dataset_file)

    ds = TrajectoryDataset.open(dataset_path)
    return ds, output_dir


def expand_json_column(
    df: pd.DataFrame,
    column: str,
    short_name: Optional[str] = None,
    all_fields: bool = False,
) -> pd.DataFrame:
    """
    Expand a column containing JSON strings into dict or individual columns.

    Args:
        df: Input dataframe
        column: Name of column containing JSON strings
        short_name: Name for the expanded dict column (defaults to column name)
        all_fields: If True, create separate columns for each JSON field

    Returns:
        DataFrame with expanded column(s)
    """
    df = df.copy()
    if short_name is None:
        short_name = column

    df[short_name] = df[column].apply(json.loads)

    if all_fields and len(df) > 0:
        first_dict = df[short_name].iloc[0]
        for field in first_dict.keys():
            df[f"{short_name}_{field}"] = df[short_name].apply(lambda x: x.get(field))

    return df
