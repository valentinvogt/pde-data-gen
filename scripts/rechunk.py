import xarray as xr
ds = xr.open_dataset(
    "/cluster/scratch/vogtva/data/bruss/final/_dataset_processed_new.nc"
)
ds = ds.chunk(
    {
        "run": 100,
        # "snapshot": 10,
        "Nx": 32,
        "Ny": 32,
    }
)
ds.to_netcdf("/cluster/scratch/vogtva/data/bruss/final/_dataset_processed_rch.nc")
