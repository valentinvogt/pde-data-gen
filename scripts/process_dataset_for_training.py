import xarray as xr
import numpy as np
import dask.array as da
import argparse


def reshape_data(ds):
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


def create_parameter_component(ds_new):
    # Get dimensions
    n_traj = ds_new.sizes["run"]
    nx = ds_new.sizes["Nx"]
    ny = ds_new.sizes["Ny"]

    # Create template data array with the right dimensions
    template = ds_new["data"].isel(component=0)

    # Start with zeros
    param_array = xr.zeros_like(template)

    # Half points for quadrants
    half_x = nx // 2
    half_y = ny // 2

    # Create masks for each quadrant
    top_left = (param_array["Nx"] < half_x) & (param_array["Ny"] < half_y)
    top_right = (param_array["Nx"] >= half_x) & (param_array["Ny"] < half_y)
    bottom_left = (param_array["Nx"] < half_x) & (param_array["Ny"] >= half_y)
    bottom_right = (param_array["Nx"] >= half_x) & (param_array["Ny"] >= half_y)

    # Fill each quadrant with the appropriate parameter value
    for i in range(n_traj):
        A_val = ds_new["A"].isel(run=i).values
        B_val = ds_new["B"].isel(run=i).values
        Du_val = ds_new["Du"].isel(run=i).values
        Dv_val = ds_new["Dv"].isel(run=i).values

        # Set parameter values in each quadrant
        param_array = param_array.where(~(top_left & (param_array["run"] == i)), A_val)
        param_array = param_array.where(~(top_right & (param_array["run"] == i)), B_val)
        param_array = param_array.where(
            ~(bottom_left & (param_array["run"] == i)), Du_val
        )
        param_array = param_array.where(
            ~(bottom_right & (param_array["run"] == i)), Dv_val
        )

    return param_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to input file")
    args = parser.parse_args()
    ds = xr.open_dataset(args.filename)
    outfile = args.filename.replace(".nc", "_proc.nc")

    # Extract dimensions
    n_runs = ds.sizes["run"]
    n_snapshots = ds.sizes["snapshot"]
    x_size = ds.sizes["x_size_and_boundary"]
    ny_coupled = ds.sizes["n_coupled_and_y_size_and_boundary"]

    # Assuming n_coupled_and_y_size_and_boundary is twice the y_size because
    # it contains the coupled variables u and v interleaved
    y_size = ny_coupled // 2

    reshaped_data = reshape_data(ds)

    # 2. Create new dataset with renamed variables
    ds_new = xr.Dataset(
        data_vars={
            "data": reshaped_data,
            "run_id": ds["run_id"],
            "model": ds["model"],
            "A": ds["A"],
            "B": ds["B"],
            "Du": ds["Du"],
            "Dv": ds["Dv"],
            "time": ds["time"],
        }
    )

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
    # At this point, no computation has been done yet
    # To actually compute and save the result:
    ds_final.to_netcdf(outfile)
