import xarray as xr
import numpy as np
import argparse
import json

def reshape_interleaved_data(ds):
    x_size = ds.sizes["x_size_and_boundary"]
    ny_coupled = ds.sizes["n_coupled_and_y_size_and_boundary"]
    y_size = ny_coupled // 2

    original_data = ds["data"]
    u_data = original_data.isel(n_coupled_and_y_size_and_boundary=slice(0, None, 2))
    v_data = original_data.isel(n_coupled_and_y_size_and_boundary=slice(1, None, 2))

    x_coords = np.arange(x_size - 2)
    y_coords = np.arange(y_size - 2)

    u_data = u_data.assign_coords(
        Nx=("x_size_and_boundary", x_coords),
        Ny=("n_coupled_and_y_size_and_boundary", y_coords),
    ).rename({"x_size_and_boundary": "x", "n_coupled_and_y_size_and_boundary": "y"})

    v_data = v_data.assign_coords(
        Nx=("x_size_and_boundary", x_coords),
        Ny=("n_coupled_and_y_size_and_boundary", y_coords),
    ).rename({"x_size_and_boundary": "x", "n_coupled_and_y_size_and_boundary": "y"})

    return u_data, v_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Consolidated netcdf file")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--config_file")
    parser.add_argument("--num_snapshots", type=int, default=0)
    parser.add_argument("--interleaved_mode", action="store_true")
    
    args = parser.parse_args()
    ds = xr.open_dataset(args.filename).chunk({"trajectory": 16})

    if args.interleaved_mode:
        u_data, v_data = reshape_interleaved_data(ds)
    else:
        u_data, v_data = ds["data"][:, :, 0], ds["data"][:, :, 1]

    if args.config_file:
        with open(args.config_file, "r") as f:
            config = json.load(f)
        sim_params = config.get("sim_params")
        dx = sim_params.get("dx", 1.0)
        dt = sim_params.get("dt", 0.0025)
        Nt = sim_params.get("Nt", 60_000)
        n_snapshots = sim_params.get("n_snapshots", 100)
        time_step = dt * Nt / n_snapshots
    else:
        time_step = 1.5
        dx = 1.0

    n_member = ds.dims["trajectory"]
    n_time = u_data.sizes["snapshot"]
    Nx = u_data.sizes["x"]
    Ny = u_data.sizes["y"]

    # Coordinates
    member = np.arange(n_member)
    time = np.arange(n_time) * time_step
    x = np.linspace(0, 1, Nx) * dx
    y = np.linspace(0, 1, Ny) * dx

    # Assemble final dataset
    ds_new = xr.Dataset(
        {
            "u": (["member", "time", "x", "y"], u_data.data),
            "v": (["member", "time", "x", "y"], v_data.data),
            "A": (["member"], ds["A"].data),
            "B": (["member"], ds["B"].data),
            "Du": (["member"], ds["Du"].data),
            "Dv": (["member"], ds["Dv"].data),
        },
        coords={
            "member": member,
            "time": time,
            "x": x,
            "y": y,
        },
    )

    ds_new = ds_new.chunk({"member": 128, "time": 1, "x": 32, "y": 32})
    ds_new.to_netcdf(args.output_file)
    print("Done.")
