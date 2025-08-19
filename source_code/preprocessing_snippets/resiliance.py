import pandas as pd, numpy as np

d=pd.read_csv("Resiliance.csv")   # load raw data

d=d.replace("..",np.nan)  # handle weird missing vals

# reshape wide yrs into long tidy format
dl=d.melt(id_vars=["Country Name","Country Code","Series Name","Series Code"],
var_name="Year",value_name="Val")

dl["Year"]=dl["Year"].str.extract(r"(\d{4})").astype(int)  # clean year col
dl["Val"]=pd.to_numeric(dl["Val"],errors="coerce")  # convert vals

dl=dl.sort_values(["Country Name","Series Name","Year"])  # keep in order
dl["miss"]=dl["Val"].isna()  # mark which ones were missing

# fill: interpolate tiny gaps, then forward/back fill edges
def fillf(s): return s.interpolate("linear",limit=1).ffill().bfill()

dl["Val"]=dl.groupby(["Country Name","Series Name"])["Val"].transform(fillf)
dl["miss"]=dl["miss"] & dl["Val"].notna()  # update imputed flag

dl=dl.reset_index(drop=True)
dl.to_csv("processed_timeseries_clean.csv",index=False)  # save cleaned data

print(dl.head(20))
