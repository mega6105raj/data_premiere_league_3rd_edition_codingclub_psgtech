import pandas as pd
import numpy as np

path="crop_and_livestock (1).csv"
df=pd.read_csv(path)
df.columns=df.columns.str.strip()

cols=["Domain Code","Domain","Area Code (M49)","Area","Element Code","Element","Item Code (CPC)","Item",
"Year Code","Year","Unit","Value","Flag","Flag Description","Note"]
df=df[[c for c in cols if c in df.columns]].copy()

df["Year"]=pd.to_numeric(df["Year"],errors="coerce").astype("Int64")
df["Value"]=pd.to_numeric(df["Value"],errors="coerce")

focus=["India","USA","Russia","France","Germany","Italy","China","Japan",
"Argentina","Portugal","Spain","Croatia","Belgium","Australia","Pakistan",
"Afghanistan","Israel","Iran","Iraq","Bangladesh","Sri Lanka","Canada",
"UK","Sweden","Saudi Arabia"]
df=df[df["Area"].isin(focus)]

# area harvested (make sure in ha)
a=df[df["Element"]=="Area harvested"].copy()
a["area_ha"]=a["Value"]
a.loc[a["Unit"].str.contains("1000",case=False,na=False),"area_ha"]=a["Value"]*1000
areas=a[["Area","Item","Year","area_ha"]]

# yields to kg/ha
y=df[df["Element"]=="Yield"].copy()
y["yield_kg_ha"]=np.nan
u=y["Unit"].str.lower()
y.loc[u.str.contains("kg/ha",na=False),"yield_kg_ha"]=y["Value"]
y.loc[u.str.contains("hg/ha",na=False),"yield_kg_ha"]=y["Value"]*0.1
y.loc[u.str.contains("t/ha",na=False)|u.str.contains("tonnes per hectare",na=False),"yield_kg_ha"]=y["Value"]*1000
yields=y[["Area","Item","Year","yield_kg_ha"]]

# production numbers directly
p=df[df["Element"]=="Production"].copy()
p2=p[["Area","Item","Year","Value"]].rename(columns={"Value":"reported_production_tonnes"})

# estimate prod by yield*area
base=pd.merge(yields,areas,on=["Area","Item","Year"],how="outer")
base["yield_kg_ha"]=base.groupby(["Area","Item"])["yield_kg_ha"].transform(lambda x:x.interpolate(method="linear"))
base["area_ha"]=base.groupby(["Area","Item"])["area_ha"].transform(lambda x:x.interpolate(method="linear"))
base["estimated_production_tonnes"]=(base["yield_kg_ha"]*base["area_ha"])/1000
prod=base[["Area","Item","Year","estimated_production_tonnes","yield_kg_ha","area_ha"]]

if not prod.empty:
    prod["_w"]=prod["area_ha"].fillna(0)
    prod["_y_w"]=prod["yield_kg_ha"].fillna(0)*prod["_w"]
    w=prod.groupby(["Area","Year"],as_index=False).agg(
        total_area_harvested_ha=("area_ha","sum"),
        est_ag_production_tonnes=("estimated_production_tonnes","sum"),
        _y_w_sum=("_y_w","sum"),
        _w_sum=("_w","sum")
    )
    w["avg_crop_yield_kg_ha"]=np.where(w["_w_sum"]>0,w["_y_w_sum"]/w["_w_sum"],np.nan)
    w=w.drop(columns=["_y_w_sum","_w_sum"])
else:
    w=pd.DataFrame(columns=["Area","Year","total_area_harvested_ha","est_ag_production_tonnes","avg_crop_yield_kg_ha"])

agg=pd.merge(w,p2.groupby(["Area","Year"],as_index=False).agg(
    reported_production_tonnes=("reported_production_tonnes","sum")
),on=["Area","Year"],how="outer")

agg=agg.rename(columns={"Area":"Country"})
agg=agg.sort_values(["Country","Year"]).reset_index(drop=True)

for c in ["total_area_harvested_ha","est_ag_production_tonnes","reported_production_tonnes"]:
    if c in agg.columns: agg[c]=agg[c].fillna(0)

# extra indicators
yagg=yields.groupby(["Area","Year"]).agg(yield_kg_ha=("yield_kg_ha","mean")).reset_index()
yagg=yagg.rename(columns={"Area":"Country"})
agg=pd.merge(agg,yagg,on=["Country","Year"],how="left")

agg["is_yield_outlier"]=(agg["yield_kg_ha"]>10000).astype(int)
agg["production_discrepancy_flag"]=np.where(
    (agg["est_ag_production_tonnes"]>0)&(agg["reported_production_tonnes"]>0)&
    (abs(agg["est_ag_production_tonnes"]-agg["reported_production_tonnes"])/agg["reported_production_tonnes"]>0.1),1,0
)

agg["food_production_index"]=np.where(
    agg["total_area_harvested_ha"]>0,
    agg["avg_crop_yield_kg_ha"]*agg["total_area_harvested_ha"]/1000,
    np.nan
)

agg["yield_variability"]=agg.groupby("Country")["yield_kg_ha"].transform(lambda x:x.rolling(window=3,min_periods=1).std())

# crop diversity
div=prod.groupby(["Area","Year"]).agg(crop_diversity_index=("Item","nunique")).reset_index()
div=div.rename(columns={"Area":"Country"})
agg=pd.merge(agg,div,on=["Country","Year"],how="left")

agg["agricultural_export_potential"]=np.nan

out="crop_livestock_processed.csv"
agg.to_csv(out,index=False)
print("Saved:",out)
print(agg.head(10))
