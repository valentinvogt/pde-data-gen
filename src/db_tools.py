import netCDF4 as nc
import numpy as np

from typing import Tuple
from dotenv import load_dotenv
import os
import pandas as pd
import json
import xarray as xr
############################################
#          Dataset Definition              #
############################################


def df_from_nc(ds) -> pd.DataFrame:
    """
    Convert a NetCDF dataset to a pandas DataFrame.
    
    Extracts all variables from the dataset except 'data', 'time', and 'input_file',
    and converts them to DataFrame columns.
    
    Args:
        ds: A NetCDF4 Dataset object containing variables to be converted
        
    Returns:
        pandas.DataFrame: DataFrame containing all extracted variables
    """
    df = pd.DataFrame()
    for var in ds.data_vars:
        if var not in ["data", "time"]: #, "time", "input_file", "component", "snapshot"]:
            df[var] = ds[var][:]
            
    if "n_snapshots" in ds.attrs:
        df["n_snapshots"] = ds.attrs["n_snapshots"]
    return df


class Dataset:
    """
    This is a wrapper to be used for analysis. Not
    to be confused with DatasetManager, which is used
    for the creation of the consolidated NetCDF file.
    """

    def __init__(self, data_dir, model, ds_id, ds_file="_dataset.nc"):
        self.data_dir = data_dir
        self.model = model
        self.ds_id = ds_id
        path = os.path.join(data_dir, model, ds_id)
        self.ds_file = os.path.join(path, ds_file)
        if not os.path.exists(self.ds_file):
            raise FileNotFoundError(self.ds_file)
        self.dataset = xr.open_dataset(self.ds_file)
        
        self.df_file = self.ds_file.replace(".nc", ".csv")
        if os.path.exists(self.df_file):
            self.df = pd.read_csv(self.df_file)
            self.df["idx"] = self.df.index
        else:
            self.df = df_from_nc(self.dataset)
            self.df["idx"] = self.df.index
            self.df.to_csv(self.df_file, index=False)
        
        # self.df["filename"] = self.df["output_file"]
        # self.df.drop(columns=["output_file"], inplace=True)

    def get_data(self, row):
        if isinstance(row, (int, np.int64)):
            idx = row
        else:
            if isinstance(row, pd.DataFrame):
                if len(row) == 1:
                    row = row.iloc[0]
                else:
                    raise ValueError("row should be Series or single-row DataFrame")
            idx = row.idx
        data = self.dataset.variables["data"][idx, :, :, :]
        return data

    def add_column(self, column_name, values, write_into_nc=False):
        self.df[column_name] = values
        if write_into_nc:
            if column_name not in self.dataset.variables:
                self.dataset.createVariable(column_name, "f8", ("trajectory",))
            self.dataset.variables[column_name][:] = values
        else:
            self.df.to_csv(self.df_file, index=False)
        print("created column", column_name)

def filter_dataset(dataset: Dataset, df) -> Dataset:
    """
    Create a new Dataset object from another one with a df that only contains certain rows.

    Args:
        dataset: The original Dataset object.

    Returns:
        A new Dataset object with the filtered DataFrame.
    """
    new_dataset = Dataset(dataset.data_dir, dataset.model, dataset.ds_id)
    if type(df) is pd.Series:
        df = df.to_frame().T

    new_dataset.df = df
    return new_dataset


def get_dataset(model, ds_id, directory_var="WORKDIR", dataset_file="_dataset.nc") -> Tuple[Dataset, str]:
    """
    Load a dataset based on model and dataset ID from the environment directory.
    
    Args:
        model: Model identifier string
        ds_id: Dataset identifier string
        directory_var: Name of the env. variable containing the directory
        
    Returns:
        Tuple[Dataset, str]: A tuple containing the loaded Dataset object and the output directory path
    """
    load_dotenv()
    data_dir = os.getenv(directory_var)
    output_dir = os.path.join(data_dir, "out")
    ds = Dataset(os.path.join(data_dir, "data"), model, ds_id, dataset_file)
    return ds, output_dir


def expand_json_column(df, column, short_name=None, all_fields=False):
    """
    Expand a column containing strings of JSON objects into multiple columns.
    """
    df = df.copy()
    if short_name is None:
        short_name = column
    df[short_name] = df[column].apply(json.loads)
    if all_fields:
        for field in df[short_name][0].keys():
            df[f"{short_name}_{field}"] = df[short_name].apply(lambda x: x.get(field))

    return df


##############################################
#         Tools for working with dfs         #
##############################################


def filter_df(df, **kwargs):
    """
    Filter a DataFrame based on multiple conditions.
    """
    mask = np.ones(len(df), dtype=bool)
    for key, value in kwargs.items():
        mask &= df[key] == value
    return df[mask]


def compute_metrics(row, data, start_frame, end_frame=-1, mode="old"):
    """
    end_frame: works like Python slicing
    Returns deviations, time_derivatives, spatial_derivatives, relative std, norm
    as 2D arrays of shape (num_frames, 2)
    """
    if end_frame < 0:
        end_frame = row["n_snapshots"] + end_frame

    num_frames = end_frame - start_frame + 1
    frame_dt = row["dt"] * row["Nt"] / row["n_snapshots"]
    deviations = np.zeros((num_frames, 2))
    time_derivatives = np.zeros((num_frames, 2))
    spatial_derivatives = np.zeros((num_frames, 2))
    relative_stds = np.zeros((num_frames, 2))
    norms = np.zeros((num_frames, 2))

    steady_state = np.zeros_like(data[0, :, :, :])

    if mode == "old":
        steady_state[:, 0::2] = row["A"]
        steady_state[:, 1::2] = row["B"] / row["A"]
        u = data[:, :, 0::2]
        v = data[:, :, 1::2]
    else:
        steady_state[0, :, :] = row["A"]
        steady_state[1, :, :] = row["B"] / row["A"]
        u = data[:, 0, :, :]
        v = data[:, 1, :, :]
        
    du_dt = np.gradient(u, frame_dt, axis=0)
    dv_dt = np.gradient(v, frame_dt, axis=0)

    for j in range(0, num_frames):
        snapshot = start_frame + j
        u_t = u[snapshot, :, :]
        v_t = v[snapshot, :, :]
        du_dx = np.gradient(u_t, row["dx"], axis=0)
        dv_dx = np.gradient(v_t, row["dx"], axis=0)
        deviations[j, 0] = np.linalg.norm(u_t - steady_state[0])
        deviations[j, 1] = np.linalg.norm(v_t - steady_state[1])
        time_derivatives[j, 0] = np.linalg.norm(du_dt[snapshot])
        time_derivatives[j, 1] = np.linalg.norm(dv_dt[snapshot])
        spatial_derivatives[j, 0] = np.linalg.norm(du_dx)
        spatial_derivatives[j, 1] = np.linalg.norm(dv_dx)
        relative_stds[j, 0] = np.std(u_t) / np.mean(u_t)
        relative_stds[j, 1] = np.std(v_t) / np.mean(v_t)
        norms[j, 0] = np.linalg.norm(u)
        norms[j, 1] = np.linalg.norm(v)

    return deviations, time_derivatives, spatial_derivatives, relative_stds, norms


def get_metrics_array(dataset: Dataset, start_frame=0, metric="dev", mode="old"):
    """
    Returns:
    all_metrics: num_trajectories x n_snapshots array of metric values
    title: for plotting
    """
    title = ""
    if metric not in ["dev", "dt", "dx", "std"]:
        raise ValueError("Not a valid metric!")
    df, get_data = dataset.df, dataset.get_data
    all_metrics = []
    for _, row in df.iterrows():
        metrics = compute_metrics(row, get_data(row), start_frame=start_frame, mode=mode)
        if metric == "dev":
            title = "Deviation"
            values = metrics[0]
        elif metric == "dt":
            title = "Time Derivative"
            values = metrics[1]
        elif metric == "dx":
            title = "Spatial Derivative"
            values = metrics[2]
        elif metric == "std":
            title = "Relative std"
            values = metrics[3]
        all_metrics.append(values)
    all_metrics = np.array(all_metrics)
    return all_metrics, title
