import pandas as pd
import numpy as np

df = pd.read_csv('data/final.csv')
df["Du"] = (2.5 + np.random.randn(len(df))) * df["Du"]
df["Dv"] = (2.5 + np.random.randn(len(df))) * df["Dv"]

df.to_csv('data/final_zoomed.csv', index=False)