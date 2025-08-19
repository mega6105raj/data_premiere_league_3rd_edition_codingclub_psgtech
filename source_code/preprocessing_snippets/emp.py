import pandas as pd, numpy as np

d=pd.read_csv("Employment_Unemployment.csv")  # load data

d=d.replace("..",np.nan)  # fix missing vals

# reshape wide yrs to long
dl=d.melt(id_vars=["Country Name","Country Code","Series Name","Series Code"],
var_name="Year",value_name="Val")

dl["Year"]=dl["Year"].str.extract(r"(\d{4})").astype(int)  # keep only year num
dl["Val"]=pd.to_numeric(dl["Val"],errors="coerce")  # force numeric

dl=dl.sort_values(["Country Name","Series Name","Year"])  # order data
dl["miss"]=dl["Val"].isna()  # mark na

# fill small gaps only (linear, max 1 step)
def fillf(s): return s.interpolate("linear",limit=1)

dl["Val"]=dl.groupby(["Country Name","Series Name"])["Val"].transform(fillf)
dl["miss"]=dl["miss"] & dl["Val"].notna()  # update imputed flag

dl=dl.reset_index(drop=True)
dl.to_csv("processed_employment_unemployment.csv",index=False)  # save

print(dl.head(20))
