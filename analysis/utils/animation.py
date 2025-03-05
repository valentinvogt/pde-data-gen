import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation

import os
import glob
import pandas as pd
import json
from dotenv import load_dotenv
from functools import partial
import argparse

from db_tools import get_db, get_data, filter_df, make_animation


def ab_grid_animation(
    df, component_idx=0, sigdigits=2, var1="A", var2="B", file="", fps=10, dpi=100
):
    if len(df) == 0:
        return None

    A_count = len(df[var1].unique())
    if var2 == "":
        df = df.sort_values(by=[var1])
        df[var2] = 0
        B_count = A_count
        A_count = 1
    else:
        df = df.sort_values(by=[var1, var2])
        B_count = int(len(df) / A_count)


    fig = plt.figure(figsize=(15, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(A_count, B_count), axes_pad=(0.1, 0.3))
    ims = []

    # Preload data and calculate global min/max for normalization
    global_min, global_max = float("inf"), float("-inf")
    for i, row in df.iterrows():
        data = get_data(row)
        ims.append((row, data))
        global_min = min(global_min, data[0, :, :, component_idx::2].min())
        global_max = max(global_max, data[0, :, :, component_idx::2].max())

    # Normalization parameters for consistent color mapping
    norm = plt.Normalize(vmin=global_min, vmax=global_max)

    def update(frame):
        fig.clear()
        grid = ImageGrid(fig, 111, nrows_ncols=(A_count, B_count), axes_pad=(0.1, 0.3))

        for ax, (row, data) in zip(grid, ims):
            im = data[0, frame, :, component_idx::2]
            ax.set_title(
                f"{var1}={row[var1]:.{sigdigits}f}\n{var2} = {row[var2]:.{sigdigits}f}",
                fontsize=6,
            )
            ax.imshow(im, cmap="viridis", norm=norm)
            ax.set_aspect("equal")
            ax.axis("off")

        row = df.iloc[0]
        time = row["dt"] * frame * row["Nt"] / row["n_snapshots"]
        fig.suptitle(
            f"{row['model'].capitalize()}, Nx={row['Nx']}, dx={row['dx']}, dt={row['dt']}, T={time:.2f}",
            fontsize=16,
        )

    anim = FuncAnimation(fig, update, frames=range(df.iloc[0]["n_snapshots"]), interval=1000/fps)

    if file:
        anim.save(file, fps=fps, dpi=dpi)
    else:
        plt.show()

    return anim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bruss")
    parser.add_argument("--subdir", type=str, default="")
    parser.add_argument("--dt", type=float, default=0)
    parser.add_argument("--Nt", type=int, default=0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--var1", type=str, default="A")
    parser.add_argument("--var2", type=str, default="B")
    parser.add_argument("--outfile", type=str, default="output.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    
    args = parser.parse_args()
    model = args.model
    subdir = args.subdir
    dt = args.dt
    Nt = args.Nt
    run_id = args.run_id
    var1 = args.var1
    var2 = args.var2
    outfile = args.outfile
    fps = args.fps
    dpi = args.dpi

    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    output_dir = os.getenv("OUT_DIR")
    output_dir = os.path.join(output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    if subdir != "":
        path = os.path.join(data_dir, model, subdir)
    else:
        path = os.path.join(data_dir, model)
    df0 = get_db(path)

    df = df0.copy()
    if run_id != "":
        df = df[df["run_id"] == run_id]
    if dt != 0:
        df = df[df["dt"] == dt]
    if Nt != 0:
        df = df[df["Nt"] == Nt]

    df = filter_df(df, A=0.5, B=1.25, Du=1.0, Dv=14.0)

    # for i, row in df.iterrows():
    #     data = get_data(row)
    #     A, B = row["A"], row["B"]
    #     make_animation(data, f"{model}-{A}-{B}", output_dir, coupled_idx=0)
    #     print(f"created ({A},{B})")

    # replace ending of outfile with _u.<ending> and _v.<ending>
    # should work for .gif and .mp4
    file_u = outfile.replace(".", "_u.")
    file_v = outfile.replace(".", "_v.")
    ab_grid_animation(df, 0, sigdigits=2, var1="random_seed", var2="", file=file_u, fps=fps, dpi=dpi)
    ab_grid_animation(df, 1, sigdigits=2, var1="random_seed", var2="", file=file_v, fps=fps, dpi=dpi)
        
if __name__ == "__main__":
    model = "bruss"
    run_id = "splitting"

    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    output_dir = os.getenv("OUT_DIR")
    output_dir = os.path.join(output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(data_dir, model, run_id)
    df = get_db(path)
    df = df[df.run_id == run_id]
    # A = 0.5
    # B = 1.25
    # Du = 1.0
    # Dv = 14.0
    # df = filter_df(df, A, B, Du, Dv)

    outfile = "anim-splitting.gif"
    file_u = outfile.replace(".", "_u.")
    file_v = outfile.replace(".", "_v.")
    ab_grid_animation(df, 0, sigdigits=2, var1="random_seed", var2="", file=file_u, fps=10, dpi=150)
    ab_grid_animation(df, 1, sigdigits=2, var1="random_seed", var2="", file=file_v, fps=10, dpi=150)
     