import xarray as xr
import numpy as np
import argparse
import os
import shutil


def reshape_data(ds):
    x_size = ds.sizes["x_size_and_boundary"]
    ny_coupled = ds.sizes["n_coupled_and_y_size_and_boundary"]
    y_size = ny_coupled // 2

    original_data = ds["data"]

    u_data = original_data.isel(n_coupled_and_y_size_and_boundary=slice(0, None, 2))
    v_data = original_data.isel(n_coupled_and_y_size_and_boundary=slice(1, None, 2))

    u_data = u_data.isel(
        x_size_and_boundary=slice(1, x_size - 1),
        n_coupled_and_y_size_and_boundary=slice(1, y_size - 1),
    )
    v_data = v_data.isel(
        x_size_and_boundary=slice(1, x_size - 1),
        n_coupled_and_y_size_and_boundary=slice(1, y_size - 1),
    )

    x_coords = np.arange(x_size - 2)
    y_coords = np.arange(y_size - 2)

    u_data = u_data.assign_coords(
        Nx=("x_size_and_boundary", x_coords),
        Ny=("n_coupled_and_y_size_and_boundary", y_coords),
    ).rename({"x_size_and_boundary": "Nx", "n_coupled_and_y_size_and_boundary": "Ny"})
    v_data = v_data.assign_coords(
        Nx=("x_size_and_boundary", x_coords),
        Ny=("n_coupled_and_y_size_and_boundary", y_coords),
    ).rename({"x_size_and_boundary": "Nx", "n_coupled_and_y_size_and_boundary": "Ny"})

    combined_data = (
        xr.concat([u_data, v_data], dim="component")
        .assign_coords(component=["u", "v"])
        .transpose("run", "snapshot", "component", "Nx", "Ny")
    )

    return combined_data


# def create_parameter_component(ds_new):
#     # Get dimensions
#     n_traj = ds_new.sizes["run"]
#     nx = ds_new.sizes["Nx"]
#     ny = ds_new.sizes["Ny"]

#     # Create template data array with the right dimensions
#     template = ds_new["data"].isel(component=0)

#     # Start with zeros
#     param_array = xr.zeros_like(template)

#     # Half points for quadrants
#     half_x = nx // 2
#     half_y = ny // 2

#     # Create masks for each quadrant
#     top_left = (param_array["Nx"] < half_x) & (param_array["Ny"] < half_y)
#     top_right = (param_array["Nx"] >= half_x) & (param_array["Ny"] < half_y)
#     bottom_left = (param_array["Nx"] < half_x) & (param_array["Ny"] >= half_y)
#     bottom_right = (param_array["Nx"] >= half_x) & (param_array["Ny"] >= half_y)

#     # Fill each quadrant with the appropriate parameter value
#     for i in range(n_traj):
#         A_val = ds_new["A"].isel(run=i).values
#         B_val = ds_new["B"].isel(run=i).values
#         Du_val = ds_new["Du"].isel(run=i).values
#         Dv_val = ds_new["Dv"].isel(run=i).values

#         # Set parameter values in each quadrant
#         param_array = param_array.where(~(top_left & (param_array["run"] == i)), A_val)
#         param_array = param_array.where(~(top_right & (param_array["run"] == i)), B_val)
#         param_array = param_array.where(
#             ~(bottom_left & (param_array["run"] == i)), Du_val
#         )
#         param_array = param_array.where(
#             ~(bottom_right & (param_array["run"] == i)), Dv_val
#         )

#     return param_array


def create_parameter_component(ds_new):
    nx = ds_new.sizes["Nx"]
    ny = ds_new.sizes["Ny"]

    template = ds_new["data"].isel(component=0)

    half_x = nx // 2
    half_y = ny // 2

    # Create quadrant masks (broadcastable over `run`)
    x_coords, y_coords = np.meshgrid(template["Nx"], template["Ny"], indexing="ij")
    top_left = (x_coords < half_x) & (y_coords < half_y)
    top_right = (x_coords >= half_x) & (y_coords < half_y)
    bottom_left = (x_coords < half_x) & (y_coords >= half_y)
    bottom_right = (x_coords >= half_x) & (y_coords >= half_y)

    # Expand dimensions to (run, Nx, Ny)
    top_left = xr.DataArray(top_left, dims=("Nx", "Ny")).expand_dims(run=ds_new["run"])
    top_right = xr.DataArray(top_right, dims=("Nx", "Ny")).expand_dims(
        run=ds_new["run"]
    )
    bottom_left = xr.DataArray(bottom_left, dims=("Nx", "Ny")).expand_dims(
        run=ds_new["run"]
    )
    bottom_right = xr.DataArray(bottom_right, dims=("Nx", "Ny")).expand_dims(
        run=ds_new["run"]
    )

    # Extract parameter values with correct shape
    A_val = ds_new["A"].expand_dims({"Nx": nx, "Ny": ny})
    B_val = ds_new["B"].expand_dims({"Nx": nx, "Ny": ny})
    Du_val = ds_new["Du"].expand_dims({"Nx": nx, "Ny": ny})
    Dv_val = ds_new["Dv"].expand_dims({"Nx": nx, "Ny": ny})

    # Construct the parameter array
    param_array = (
        A_val.where(top_left, 0)
        + B_val.where(top_right, 0)
        + Du_val.where(bottom_left, 0)
        + Dv_val.where(bottom_right, 0)
    )
    param_array = param_array.expand_dims(snapshot=ds_new["snapshot"])
    return param_array


def downsample_in_time(ds, num_snapshots):
    snapshot_indices = np.linspace(
        0, ds.sizes["snapshot"] - 1, num_snapshots, dtype=int
    )
    return ds.isel(snapshot=snapshot_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to input file")
    parser.add_argument(
        "--num_snapshots",
        type=int,
        default=0,
        help="Number of snapshots to downsample to",
    )
    args = parser.parse_args()
    n_snapshots_target = args.num_snapshots
    outfile_nc = args.filename.replace(".nc", "_proc.nc")
    outfile_zarr = outfile_nc.replace(".nc", ".zarr")

    ds = xr.open_dataset(args.filename)
    ds = ds.chunk(
        {
            "run": 100,
            "snapshot": 10,
            "x_size_and_boundary": 10,
            "n_coupled_and_y_size_and_boundary": 10,
        }
    )
    # Extract dimensions
    n_runs = ds.sizes["run"]
    n_snapshots = ds.sizes["snapshot"]

    assert (
        n_snapshots_target < n_snapshots
    ), "The num_snapshots provided for downsampling is larger than the number of available snapshots!"
    # Assuming n_coupled_and_y_size_and_boundary is twice the y_size because
    # it contains the coupled variables u and v interleaved

    print("Reshaping")
    reshaped_data = reshape_data(ds)

    if n_snapshots_target > 0:
        print("Downsampling")
        reshaped_data = downsample_in_time(reshaped_data, n_snapshots_target)

    ds_new = xr.Dataset(
        data_vars={
            "data": reshaped_data,
            "run_id": ds["run_id"],
            "model": ds["model"],
            "A": ds["A"],
            "B": ds["B"],
            "Du": ds["Du"],
            "Dv": ds["Dv"],
        }
    )

    print("Creating parameter component")
    param_component = create_parameter_component(ds_new)
    param_component = param_component.expand_dims(dim="component").assign_coords(
        component=["param"]
    )

    data_with_param = xr.concat([ds_new["data"], param_component], dim="component")

    ds_final = xr.Dataset(
        data_vars={
            "data": data_with_param,
            "A": ds["A"],
            "B": ds["B"],
            "Du": ds["Du"],
            "Dv": ds["Dv"],
        }
    )
    ds_final.rename({"run": "trajectory"})
    ds_final = ds_final.chunk({"Nx": 10, "Ny": 10})
    # At this point, no computation has been done yet
    # To actually compute and save the result:
    # ds_final.to_netcdf(outfile)
    print("Writing to file")
    if os.path.exists(outfile_zarr):
        shutil.rmtree(outfile_zarr)

    ds_final.to_zarr(outfile_zarr)

    print("Done")
