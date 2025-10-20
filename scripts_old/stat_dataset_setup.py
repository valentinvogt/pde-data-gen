import numpy as np
import xarray as xr
import pandas as pd

MODEL="gray_scott"
DATASET="/cluster/scratch/vogtva/data/gray_scott/gs_final/_dataset.nc"
CNO_PATH="/cluster/scratch/vogtva/models/gs_final/model400"
N_TEST=200
N_EXTRACT=10
DATASET_MODE="new"
indices = np.load(f"{CNO_PATH}/indices.npy")

test_indices = indices[-N_TEST:]
sampled_indices = np.random.choice(test_indices, N_EXTRACT, replace=False)

ds = xr.open_dataset(DATASET)
if DATASET_MODE == "old":
    ds_test = ds.isel(run=sampled_indices)
else:
    ds_test = ds.isel(trajectory=sampled_indices)
params = ds_test[["A", "B", "Du", "Dv"]].to_dataframe().reset_index(drop=True)


if MODEL == "bruss":
    initial_conditions = ds_test["data"][:, 0, 0]
    params["ic_type"] = "normal"
    ic_sigma = np.round(initial_conditions.values.std(axis=(1, 2)), 1)
    string_array = np.array([f"{{'sigma_u': {s:.1f}, 'sigma_v': {s:.1f}}}" for s in ic_sigma])
    params["initial_condition"] = string_array
elif MODEL == "gray_scott":
    initial_conditions = ds_test["data"][:, 0, 1]
    params["ic_type"] = "point_sources"
    ic_density = np.round(initial_conditions.values.sum(axis=(1,2)) / 128**2, 2)
    string_array = np.array([f'{{"density": {d:.2f}}}' for d in ic_density])
    params["initial_condition"] = string_array
else:
    raise ValueError(f"Invalid model {MODEL}")


params.to_csv(f"{MODEL}_stat.csv", index=False)
