import xarray as xr
import numpy as np

# Load the dataset
in_file = "/cluster/work/math/vogtva/data/gray_scott/gs_final/_dataset.nc"
out_file = "/cluster/work/math/vogtva/data/bruss/clean/gray_scott_fmt.nc"

ds = xr.open_dataset(in_file)
ds = ds.chunk({"trajectory": 16})

n_member = ds.dims["trajectory"]
n_time = ds.dims["snapshot"]
Nx = ds.dims["x"]
Ny = ds.dims["y"]

# Create new coordinate arrays
member = np.arange(n_member)
time = np.arange(100) * 1.5 
x = np.linspace(0, 1, Nx)  # Replace with actual spatial coords if non-uniform
y = np.linspace(0, 1, Ny)

# Split the data variable
u = ds["data"][:, :-1, 0, :, :]
v = ds["data"][:, :-1, 1, :, :]


# Build the new dataset
ds_new = xr.Dataset(
    {
        "u": (["member", "time", "x", "y"], u.data),
        "v": (["member", "time", "x", "y"], v.data),
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
    }
)
ds_new = ds_new.chunk(
        {"member": 128, "time": 1, "x": 32, "y": 32}
    )
# Save the new dataset
ds_new.to_netcdf(out_file)
