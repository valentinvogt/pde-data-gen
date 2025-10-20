import xarray as xr
ds = xr.open_dataset(
    "/cluster/scratch/vogtva/data/gray_scott/gs_new/_dataset_processed.nc"
)
ds = ds.chunk(
    {
        "trajectory": 100,
        # "snapshot": 10,
        "x": 32,
        "y": 32,
    }
)
ds.to_netcdf("/cluster/scratch/vogtva/data/gray_scott/gs_new/_dataset_processed_rch.nc")
