import numpy as np
import pandas as pd
import json
import netCDF4 as nc
import src.db_tools as db_tools
import matplotlib.pyplot as plt
import seaborn as sns
from src.db_tools import (
    get_dataset,
    expand_json_column,
    filter_df,
    filter_dataset,
)
from src.plotting import make_animation, plot
from src.analysis.classifiers import classify_temporal_behavior
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from skimage import measure
import xarray as xr

ds_final, output_dir = get_dataset("bruss", "final", "SCRATCHDIR")
df_fin = ds_final.df
df_fin["ratio_b_a"] = df_fin["B"] / df_fin["A"]
df_fin["ratio_d"] = df_fin["Dv"] / df_fin["Du"]

ds_sa, output_dir = get_dataset("bruss", "final_2", "SCRATCHDIR")
df_sa = ds_sa.df
df_sa["ratio_b_a"] = df_sa["B"] / df_sa["A"]
df_sa["ratio_d"] = df_sa["Dv"] / df_sa["Du"]

idx_fin = list(df_fin.sample(9000).idx)
idx_sa = list(df_sa[df_sa.mean_dt > 50].sample(1000).idx)
d0 = ds_final.dataset["data"].sel(run=idx_fin)
ds = ds_sa.dataset
if "trajectory" in ds.indexes:
    ds = ds.reset_index("trajectory")

ds = ds.reset_coords("trajectory", drop=True)
ds = ds.assign_coords(trajectory=np.arange(ds.dims["trajectory"]))
d1 = ds["data"].sel(trajectory=idx_sa)
d1 = d1.sel(snapshot=np.arange(0, 100))

# concatenate the two datasets
d = xr.concat([d0, d1], dim="trajectory")
d.to_netcdf("/cluster/scratch/vogtva/data/bruss/bruss_final_merged.nc")