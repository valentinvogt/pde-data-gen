import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from typing import Tuple
from dotenv import load_dotenv
import os
import pandas as pd
import json
from functools import partial

############################################
#          Dataset Definition              #
############################################


def df_from_nc(ds):
    df = pd.DataFrame()
    for var in ds.variables:
        if var not in ["data", "time", "input_file"]:
            df[var] = ds[var][:]
    return df


class Dataset:
    """
    This is a wrapper to be used for analysis. Not
    to be confused with DatasetManager, which is used
    for the creation of the consolidated NetCDF file.
    """

    def __init__(self, data_dir, model, ds_id):
        self.data_dir = data_dir
        self.model = model
        self.ds_id = ds_id
        path = os.path.join(data_dir, model, ds_id)
        self.ds_file = os.path.join(path, "_dataset.nc")
        self.dataset = nc.Dataset(self.ds_file, "a")
        self.df = df_from_nc(self.dataset)
        self.df["filename"] = self.df["output_file"]
        # self.df.drop(columns=["output_file"], inplace=True)
        self.df["idx"] = self.df.index

    def get_data(self, row):
        if isinstance(row, int):
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

    def add_column(self, column_name, values, exclude_flag="has_nans"):
        self.df[column_name] = values
        if column_name not in self.dataset.variables:
            self.dataset.createVariable(column_name, "f8", ("run",))
        self.dataset.variables[column_name][:] = values


def filter_dataset(dataset: Dataset, df) -> Dataset:
    """
    Create a new Dataset object from another one with a df that only contains certain rows.

    Parameters:
    - dataset: The original Dataset object.
    - filter_func: A function that takes a DataFrame row and returns True if the row should be included.

    Returns:
    - A new Dataset object with the filtered DataFrame.
    """
    new_dataset = Dataset(dataset.data_dir, dataset.model, dataset.ds_id)
    new_dataset.df = df
    return new_dataset


def get_dataset(model, ds_id, directory_var="WORK_DIR") -> Tuple[Dataset, str]:
    load_dotenv()
    data_dir = os.getenv(directory_var)
    output_dir = os.path.join(data_dir, "out")
    ds = Dataset(os.path.join(data_dir, "data"), model, ds_id)
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


def compute_metrics(row, data, start_frame, end_frame=-1):
    """
    end_frame: works like Python slicing
    Returns deviations, time_derivatives, spatial_derivatives, relative std
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

    steady_state = np.zeros_like(data[0, :, :])

    steady_state[:, 0::2] = row["A"]
    steady_state[:, 1::2] = row["B"] / row["A"]

    u = data[:, :, 0::2]
    v = data[:, :, 1::2]
    du_dt = np.gradient(u, frame_dt, axis=0)
    dv_dt = np.gradient(v, frame_dt, axis=0)

    for j in range(0, num_frames):
        snapshot = start_frame + j
        u_t = u[snapshot, :, :]
        v_t = v[snapshot, :, :]
        du_dx = np.gradient(u_t, row["dx"], axis=0)
        dv_dx = np.gradient(v_t, row["dx"], axis=0)
        deviations[j, 0] = np.linalg.norm(u_t - steady_state[:, 0::2])
        deviations[j, 1] = np.linalg.norm(v_t - steady_state[:, 1::2])
        time_derivatives[j, 0] = np.linalg.norm(du_dt[snapshot])
        time_derivatives[j, 1] = np.linalg.norm(dv_dt[snapshot])
        spatial_derivatives[j, 0] = np.linalg.norm(du_dx)
        spatial_derivatives[j, 1] = np.linalg.norm(dv_dx)
        relative_stds[j, 0] = np.std(u_t) / np.mean(u_t)
        relative_stds[j, 1] = np.std(v_t) / np.mean(v_t)

    return deviations, time_derivatives, spatial_derivatives, relative_stds


def get_metrics_array(dataset: Dataset, start_frame=0, metric="dev"):
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
        metrics = compute_metrics(row, get_data(row), start_frame=start_frame)
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


###################################################
#         Tools for plotting and animating        #
###################################################


def plot(data, global_min, global_max):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.4})
    ims = []
    for coupled_idx, ax in enumerate(axes):
        matrix = data[0, :, 0::2]
        matrix /= np.max(matrix)
        im = ax.imshow(matrix, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"Snapshot 1, {'u' if coupled_idx == 0 else 'v'}")
        ims.append(im)
    return fig, axes, ims


def animate(snapshot, data, ims, axes):
    for coupled_idx, (ax, im) in enumerate(zip(axes, ims)):
        matrix = data[snapshot, :, coupled_idx::2]
        matrix /= matrix.max()  # Normalize
        im.set_array(matrix)
        name = "u" if coupled_idx == 0 else "v"
        ax.set_title(f"Snapshot {snapshot + 1}, {name}")
    return ims


def make_animation(data, filename_no_ext, out_dir):
    """
    Creates .gif animation of the data in the specified directory.
    """
    global_min = np.min(data)
    global_max = np.max(data)
    fig, axes, ims = plot(data, global_min, global_max)
    ani = animation.FuncAnimation(
        fig,
        partial(animate, data=data, ims=ims, axes=axes),
        frames=data.shape[0],
        interval=100,
        blit=True,
    )
    out_name = os.path.join(out_dir, f"{filename_no_ext}_output.gif")
    ani.save(out_name, writer="ffmpeg", dpi=150)
    plt.close(fig)


def plot_grid(
    dataset: Dataset,
    component_idx=0,
    frame=-1,
    sigdigits=3,
    var1="A",
    var2="B",
    filename="",
    scale=1
):
    df, get_data = dataset.df, dataset.get_data
    if len(df) == 0:
        return None

    if var1 == "":
        A_count = 1
        B_count = len(df)
    elif var2 == "":
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1])
        df[var2] = 0
        B_count = A_count
        A_count = 1
    else:
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1, var2])
        B_count = int(len(df) / A_count)

    fig = plt.figure(figsize=(scale * 3 * B_count + 1, scale * 5 * A_count))
    grid = ImageGrid(fig, 111, nrows_ncols=(A_count, B_count), axes_pad=(0.1, 0.3))
    ims = []

    for i, row in df.iterrows():
        data = get_data(row)
        f_min = data.min()
        f_max = data.max()
        ims.append((row, data[frame, :, component_idx::2], f_min, f_max))

    for ax, (row, im, f_min, f_max) in zip(grid, ims):
        if var1 == "":
            label = ""
        else:
            if isinstance(row[var1], float):
                label = f"{var1}={row[var1]:.{sigdigits}f}"
            else:
                label = f"{var1}={row[var1]}"
            if var2 != "":
                label += f"\n{var2} = {row[var2]:.{sigdigits}f}"
            ax.set_title(
                label,
                fontsize=6,
            )
        ax.imshow(im, cmap="viridis", vmin=f_min, vmax=f_max)
        ax.set_aspect("equal")
        ax.axis("off")

    row = df.iloc[0]
    if frame == -1:
        time = row["dt"] * row["Nt"]
    else:
        time = row["dt"] * frame * row["Nt"] / row["n_snapshots"]
    fig.suptitle(
        f"{row['model'].capitalize()}, Nx={row['Nx']}, dx={row['dx']}, dt={row['dt']}, T={time:.2f}",
        fontsize=16,
    )

    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=100)
        plt.close()
    return grid


def metrics_grid(
    dataset: Dataset,
    start_frame,
    sigdigits=3,
    joint=False,
    var1="A",
    var2="B",
    metric="dev",
    filename="",
    show_title=True,
    scale=1,
):
    if metric == "dev":
        text = "Deviation ||u(t) - u*||"
    elif metric == "dx":
        text = "Spatial Derivative ||∇u(t)||"
    elif metric == "dt":
        text = "Time Derivative ||du/dt||"
    elif metric == "std":
        text = "Relative Standard Deviation"
    else:
        raise ValueError("metric must be 'dev', 'dx', or 'dt'")

    df, get_data = dataset.df, dataset.get_data
    if len(df) == 0:
        return None

    if var1 == "":
        A_count = 1
        B_count = len(df)
    elif var2 == "":
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1])
        df[var2] = 0
        B_count = A_count
        A_count = 1
    else:
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1, var2])
        B_count = int(len(df) / A_count)

    df = df.reset_index(drop=True)
    fig, axes = plt.subplots(
        A_count, B_count, figsize=(scale * 3 * B_count + 1, scale * 5 * A_count)
    )

    axes = np.atleast_2d(axes)

    for i, row in df.iterrows():
        data = get_data(row)
        steady_state = np.zeros_like(data[0, :, :])

        steady_state[:, 0::2] = row["A"]
        steady_state[:, 1::2] = row["B"] / row["A"]

        metrics = compute_metrics(row, data, start_frame)
        if metric == "dev":
            values = metrics[0]
        elif metric == "dt":
            values = metrics[1]
        elif metric == "dx":
            values = metrics[2]
        elif metric == "std":
            values = metrics[3]

        row_idx = i // B_count if B_count > 1 else i
        col_idx = i % B_count if B_count > 1 else 0

        if not joint:
            axes[row_idx, col_idx].plot(
                np.arange(start_frame, row["n_snapshots"])
                * row["dt"]
                * row["Nt"]
                / row["n_snapshots"],
                values[:, 0],
                label="u",
            )
            axes[row_idx, col_idx].plot(
                np.arange(start_frame, row["n_snapshots"])
                * row["dt"]
                * row["Nt"]
                / row["n_snapshots"],
                values[:, 1],
                label="v",
            )
            if scale >= 1:
                axes[row_idx, col_idx].legend()
        else:
            values = np.linalg.norm(values, axis=1)
            axes[row_idx, col_idx].plot(
                np.arange(start_frame, row["n_snapshots"])
                * row["dt"]
                * row["Nt"]
                / row["n_snapshots"],
                values[:],
            )
            if var1 == "":
                label = ""
            else:
                if isinstance(row[var1], float):
                    label = f"{var1}={row[var1]:.{sigdigits}f}"
                else:
                    label = f"{var1}={row[var1]}"
                if var2 != "":
                    label += f"\n{var2} = {row[var2]:.{sigdigits}f}"
                axes[row_idx, col_idx].set_title(
                    label,
                    fontsize=6,
                )
        # axes[row_idx, col_idx].axis("off")

    row = df.iloc[0]
    time = row["dt"] * row["Nt"]
    if show_title:
        fig.suptitle(
            f"{row['model'].capitalize()}, Nx={row['Nx']}, dx={row['dx']}, dt={row['dt']}, T={time:.2f}, {text}",
            fontsize=4 * scale * B_count,
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=100)
        plt.close()

    return axes


def plot_ball_behavior(
    dataset: Dataset, start_frame=0, metric="dev", joint=False, fig=None, label=None
):
    """
    Plot the mean and mean + std of the given metric,
    as well as the trajectory with the minimum final value.
    joint: whether to average u and v to get a single time series
    fig: optional, if several plots are to be combined
    label: if there are several plots, identify which is which using this
    Returns a Plotly figure.
    """

    df = dataset.df
    all_metrics, title = get_metrics_array(
        dataset, start_frame=start_frame, metric=metric
    )
    all_metrics = np.array(all_metrics)
    row = df.iloc[0]
    dt = row["dt"] * row["Nt"] / row["n_snapshots"]
    t = np.linspace(
        start_frame * dt, row["n_snapshots"] * dt, row["n_snapshots"] - start_frame
    )
    # Compute mean and std
    avg_metric = np.mean(all_metrics, axis=0)
    std_metric = np.std(all_metrics, axis=0)

    ids = ["u", "v"]
    traj_count = 2
    if joint:
        avg_metric_uv = avg_metric
        avg_metric = np.mean(avg_metric_uv, axis=1)
        std_metric = np.linalg.norm(avg_metric_uv, axis=1)
        ids = ["u+v"]
        traj_count = 1

    for j in range(traj_count):
        id = ids[j]
        min_idx = np.argmin(all_metrics[:, -1, j])
        min_row = all_metrics[min_idx, :, j]

        avg_metric_loc = avg_metric[:, j]
        std_metric_loc = std_metric[:, j]
        # Create figure
        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate(
                    [avg_metric_loc + std_metric_loc, (avg_metric_loc)[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )

        text_avg = title
        text_std = f"Min({title})"
        if label is not None:
            text_avg += f"({label})"
            text_std += f"({label})"
        text_avg += f", {id}"
        text_std += f", {id}"

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=t,
                y=avg_metric_loc,
                mode="lines",
                name=text_avg,
                hovertemplate="Index: %{x}<br>Deviation: %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=min_row,
                mode="lines",
                name=text_std,
                hovertemplate="Index: %{x}<br>Min: %{y:.2f}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title="Deviation Metrics",
        xaxis_title="Time Step/Index",
        yaxis_title="Deviation Value",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def plot_all_trajectories(dataset, start_frame=0, metric="dev", fig=None, label_column="idx"):
    df = dataset.df
    t = np.linspace(0, 100, 100)
    title = ""

    # Create figure
    show = False
    if fig is None:
        show = True
        fig = go.Figure()
    all_metrics, title = get_metrics_array(dataset, start_frame, metric)
    for i, values in enumerate(all_metrics):
        # Add a trace for each row's metric values
        fig.add_trace(
            go.Scatter(
                x=t,
                y=values[:, 0],
                mode="lines",
                name=f"{label_column} = {df.iloc[i][label_column]}",
                hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{title} Metrics for All Rows",
        xaxis_title="Time Step/Index",
        yaxis_title=f"{title} Value",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"))

    if show:
        fig.show()
    else:
        return fig
