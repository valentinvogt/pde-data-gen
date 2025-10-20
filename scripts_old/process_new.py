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

s

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

    return u_data, v_data


def downsample_in_time(ds, num_snapshots):
    snapshot_indices = np.linspace(5, 5 + 9 * num_snapshots, 10, dtype=int)
    return ds.isel(snapshot=snapshot_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to input file")
    parser.add_argument("--num_snapshots", type=int, default=0)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--without_param", action="store_true")
    args = parser.parse_args()

    out_file = args.output_file
    if out_file is None:
        outfile_nc = args.filename.replace(".nc", "_proc.nc")
        outfile_zarr = outfile_nc.replace(".nc", ".zarr")
    else:
        outfile_nc = out_file
        outfile_zarr = outfile_nc.replace(".nc", ".zarr")

    if os.path.exists(outfile_zarr):
        shutil.rmtree(outfile_zarr)

    ds = xr.open_dataset(args.filename)
    ds = ds.chunk({"run": 16})
    reshaped_data = reshape_data(ds)

    if args.num_snapshots > 0:
        reshaped_data = downsample_in_time(reshaped_data, args.num_snapshots)

    ds_new = xr.Dataset(
        data_vars={
            "data": reshaped_data,
            "A": ds["A"],
            "B": ds["B"],
            "Du": ds["Du"],
            "Dv": ds["Dv"],
        }
    )

    if not args.without_param:
        template = ds_new["data"].isel(component=0)
        data_with_param = ds_new["data"]

        for var in ["A", "B", "Du", "Dv"]:
            param_comp = ds_new[var].broadcast_like(template)
            param_comp = param_comp.expand_dims(component=[var])
            data_with_param = xr.concat([data_with_param, param_comp], dim="component")

        ds_final = xr.Dataset(
            data_vars={
                "data": data_with_param,
                "A": ds["A"],
                "B": ds["B"],
                "Du": ds["Du"],
                "Dv": ds["Dv"],
            }
        )
    else:
        ds_final = ds_new

    ds_final = ds_final.rename({"run": "trajectory"}).chunk(
        {"trajectory": 128, "snapshot": 1, "Nx": 32, "Ny": 32}
    )
    ds_final.to_netcdf(outfile_nc)
    # ds_final.to_zarr(outfile_zarr, mode="w")

    # if outfile_nc.endswith(".nc"):
    #     xr.open_zarr(outfile_zarr).to_netcdf(outfile_nc)

    print("Done.")
