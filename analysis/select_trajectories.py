import numpy as np
import pandas as pd
import json
import netCDF4 as nc
import src.db_tools as db_tools
from dotenv import load_dotenv
import xarray as xr

ds, _ = db_tools.get_dataset("bruss", "transfer", "SCRATCHDIR")
df = ds.df
df["ratio_b_a"] = df["B"] / df["A"]
df["ratio_d"] = df["Dv"] / df["Du"]

print(len(df[((df.ratio_d > 9) & (df.ratio_b_a > 1.5)) | ((df.ratio_d < 9) & (df.ratio_b_a < 1.5))]))