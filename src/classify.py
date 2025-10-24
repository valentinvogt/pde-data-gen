import numpy as np
import sys
import pandas as pd
import argparse

from pde_data_gen.trajectory_dataset import TrajectoryDataset, get_dataset
from pde_data_gen.analysis import (
    classify_frames_pca,
    classify_pattern,
    classify_temporal_behavior,
    compute_metrics,
)


def compute_classification_metrics(ds: TrajectoryDataset, time_ratio=0.1, mode="old") -> pd.DataFrame:
    df = ds.df
    n = len(df)
    if n == 0:
        raise ValueError("Empty df provided!")

    # df["has_nans"] = False
    j = 0

    for i, row in df.iterrows():
        if i == j and n > 100:
            print(int(np.round(100 * j / n)), "%")
            j += int(np.round(0.1 * n))
            sys.stdout.flush()

        num_snapshots = row["n_snapshots"]
        data = ds.get_data(i)

        # if np.any(data.mask):
        #     df.at[i, "has_nans"] = True
        #     continue

        A, B = row["A"], row["B"]
        u_ss, v_ss = A, B / A
        starting_idx = int(num_snapshots * time_ratio)

        metrics = compute_metrics(data, u_ss, v_ss, starting_idx, mode=mode)
        for key, value in metrics.items():
            df.at[i, key] = value

    for col in df.columns:
        if col not in ds.df.columns:
            ds.add_column(col, df[col].values)

    ds.save()
    ds.close()
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bruss")
    parser.add_argument("--ds_id", default="")
    parser.add_argument("--time_ratio", default=0.1, type=float)
    parser.add_argument("--directory_var", default="WORKDIR", type=str)
    parser.add_argument("--mode", default="old", type=str, choices=["old", "new"])
    args = parser.parse_args()
    ds, _ = get_dataset(args.model, args.ds_id, args.directory_var)

    if args.mode == "old":
        final_frames_u = ds.ds["data"][:, -1, :, 0::2]
    else:
        final_frames_u = ds.ds["data"][:, -1, 0, :, :]
        if not isinstance(final_frames_u, np.ndarray):
            final_frames_u = final_frames_u.values
    compute_classification_metrics(ds, time_ratio=args.time_ratio, mode=args.mode)

    labels = classify_frames_pca(final_frames_u, n_components=10, n_clusters=3)
    ds.add_column("pca_label", labels)

    classes = classify_pattern(final_frames_u)
    ds.add_column("pattern_class", classes)

    ds.save()

    print(f"Added classification metrics to {ds.ds_file}")
