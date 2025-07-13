###################################################
#         Tools for plotting and animating        #
###################################################
import numpy as np
import os
from src.db_tools import Dataset, compute_metrics, get_metrics_array
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid
import plotly.graph_objects as go
from functools import partial


def plot(data, global_min, global_max, frame=-1, mode="old"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.4})
    ims = []
    for coupled_idx, ax in enumerate(axes):
        if mode == "old":
            matrix = data[frame, :, coupled_idx::2]
        else:
            if len(data.shape) == 4:
                matrix = data[frame, coupled_idx, :, :]
            else:
                matrix = data[frame, :, :]
        matrix /= np.max(matrix)
        im = ax.imshow(matrix, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"Snapshot {frame}, {'u' if coupled_idx == 0 else 'v'}")
        ims.append(im)
    return fig, axes, ims


def animate(snapshot, data, ims, axes, mode="old"):
    for coupled_idx, (ax, im) in enumerate(zip(axes, ims)):
        if mode == "old":
            matrix = data[snapshot, :, coupled_idx::2]
        else:
            if len(data.shape) == 4:
                matrix = data[snapshot, coupled_idx, :, :]
            else:
                matrix = data[snapshot, :, :]
        matrix /= matrix.max()  # Normalize
        im.set_array(matrix)
        name = "u" if coupled_idx == 0 else "v"
        ax.set_title(f"Snapshot {snapshot + 1}, {name}")
    return ims


def make_animation(data, filename_no_ext, out_dir, mode="old"):
    """
    Creates .gif animation of the data in the specified directory.
    """
    global_min = np.min(data)
    global_max = np.max(data)
    fig, axes, ims = plot(data, global_min, global_max, mode=mode)
    ani = animation.FuncAnimation(
        fig,
        partial(animate, data=data, ims=ims, axes=axes, mode=mode),
        frames=data.shape[0],
        interval=100,
        blit=True,
    )
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
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
    scale=1,
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
    start_frame=0,
    sigdigits=3,
    joint=False,
    var1="A",
    var2="B",
    metric="dev",
    filename="",
    show_title=True,
    scale=1,
):
    """
    Generates a grid of metrics plots for a given dataset.

    Args:
        dataset: The dataset containing trajectories to be plotted.
        start_frame: The frame at which to start computing metrics.
        sigdigits: Number of digits to show in labels for float variables.
        joint: Whether to average u and v to get a single time series.
        var1: First variable by which to organize the grid.
        var2: Second variable by which to organize the grid.
        metric: The metric to plot. Must be 'dev', 'dx', 'dt', or 'std'.
        filename: The name of the file to save the plot to.
        show_title: Whether to show the title of the plot.
        scale: Scaling factor for the plot.

    Returns:
        Axes of the plot.
    """
    if metric == "dev":
        text = "Deviation ||u(t) - u*||"
    elif metric == "dx":
        text = "Spatial Derivative ||∇u(t)||"
    elif metric == "dt":
        text = "Time Derivative ||du/dt||"
    elif metric == "std":
        text = "Relative Standard Deviation"
    elif metric == "norm":
        text = "Absolute Norm"
    else:
        raise ValueError("metric must be one of 'dev', 'dx', 'dt', 'std', 'norm'.")

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
        elif metric == "norm":
            values = metrics[4]

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


def plot_all_trajectories(
    dataset, start_frame=0, metric="dev", fig=None, label_column="idx"
):
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
