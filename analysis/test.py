import numpy as np
import sys
import pandas as pd
import argparse

from src.db_tools import Dataset, get_dataset
from src.analysis import (
    classify_frames_pca,
    classify_pattern,
    classify_temporal_behavior,
    compute_metrics,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bruss")
    parser.add_argument("--ds_id", default="")
    parser.add_argument("--time_ratio", default=0.1, type=float)
    parser.add_argument("--directory_var", default="WORKDIR", type=str)

    args = parser.parse_args()
    ds, _ = get_dataset(args.model, args.ds_id, args.directory_var)

    final_frames_u = ds.dataset["data"][:, -1, :, 0::2]

    for n_clusters in range(1, 10):
        _, intertia = classify_frames_pca(final_frames_u, n_components=10, n_clusters=3)
        print(f"n_clusters: {n_clusters}, inertia: {intertia}")

    print(f"Added classification metrics to {ds.ds_file}")
