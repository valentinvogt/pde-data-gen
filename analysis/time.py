import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_settling_times_batched(dataset, n_traj, batch_size, threshold=0.01):
    settling_times_all = []
    rel_dt_norm_all = []

    for start in range(0, n_traj, batch_size):
        end = min(start + batch_size, n_traj)
        data = dataset[start:end]  # shape: [batch_size, n_snap, Nx, 2Ny]

        # Compute discrete time derivative
        dt_data = np.diff(data, axis=1)  # shape: [B, n_snap-1, Nx, 2Ny]

        dt_norm = np.linalg.norm(dt_data, axis=(2, 3))  # [B, n_snap-1]
        max_dt = dt_norm.max(axis=1, keepdims=True)
        rel_dt_norm = dt_norm / max_dt  # relative norms

        def find_settling_time(norms, thresh):
            for t in range(len(norms)):
                if np.all(norms[t:] < thresh):
                    return t + 1
            return len(norms)

        settling_times = [find_settling_time(rel_dt_norm[i], threshold) for i in range(end - start)]

        settling_times_all.extend(settling_times)
        rel_dt_norm_all.extend(rel_dt_norm)

    return np.array(settling_times_all), np.array(rel_dt_norm_all)


if __name__ == "__main__":
    base_name = '/cluster/scratch/vogtva/data/bruss/final/_dataset'
    ds = xr.open_dataset(base_name + '.nc')

    data = ds["data"]
    settling_times, _ = compute_settling_times_batched(data, data.shape[0], 32, 0.1)

    df = pd.read_csv(base_name + '.csv')
    df["settling_time"] = settling_times
    df.to_csv(base_name + '_times.csv')
