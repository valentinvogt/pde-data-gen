import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys


def final_frame(filename):
    file_path = filename
    dataset = nc.Dataset(file_path, mode="r")

    # Extract dimensions
    n_members = dataset.dimensions["n_members"].size
    n_snapshots = dataset.dimensions["n_snapshots"].size
    x_size_and_boundary = dataset.dimensions["x_size_and_boundary"].size
    n_coupled_and_y_size_and_boundary = dataset.dimensions[
        "n_coupled_and_x_size_and_boundary"
    ].size
    n_coupled = dataset.getncattr("n_coupled")
    x_size = dataset.getncattr("x_length")
    y_size = dataset.getncattr("y_length")

    # Extract the data variable
    data = dataset.variables["data"][:]

    # u = data[:, :, :, 0::n_coupled]
    # v = data[:, :, :, 1::n_coupled]
    # u_min = np.min(u)
    # u_max = np.max(u)
    # v_min = np.min(v)
    # v_max = np.max(v)
    # print(f"{u_min} <= u <= {u_max}")
    # print(f"{v_min} <= v <= {v_max}")
    global_min = np.min(data)
    global_max = np.max(data)
    # print(global_min)
    # print(global_max)

    u0 = 5
    v0 = 9

    logging = False

    # Function to initialize each frame for animation
    def animate(snapshot):
        for coupled_idx, ax in enumerate(axes):
            matrix = data[member, snapshot, :, coupled_idx::n_coupled]
            im = ims[coupled_idx]
            im.set_array(matrix)  # Update data for each coupled component
            ax.set_title(
                f"Member {member + 1}, Snapshot {snapshot + 1}, Coupled Index {coupled_idx + 1}"
            )
            if snapshot % 10 == 0 and logging:
                component_name = "u" if coupled_idx == 0 else "v"
                steady_state = u0 if coupled_idx == 0 else v0
                l1 = np.sum(np.abs(matrix - steady_state))
                print(f"{component_name} - {component_name}0 = {l1}")

        return ims

    member = 0
    # Initialize the figure and axes for the two components
    fig, axes = plt.subplots(
        nrows=1, ncols=n_coupled, figsize=(12, 6), gridspec_kw={"wspace": 0.4}
    )

    ims = []
    for coupled_idx, ax in enumerate(axes):
        # Display the first snapshot initially; this will be updated in the animation
        matrix = data[member, 0, :, coupled_idx::n_coupled]
        im = ax.imshow(
            matrix, cmap="viridis", aspect="equal", vmin=global_min, vmax=global_max
        )
        ims.append(im)
        ax.set_xlabel("y")
        ax.set_ylabel("x")

    # Add a colorbar
    cbar = fig.colorbar(
        ims[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("Data Value")

    # Save animation to file

    # remove prefix 'data/' from filename
    animate(n_snapshots - 1)

    dataset.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python final_frame.py <filename>")
        sys.exit

    filename = sys.argv[1]
    plt.show()
