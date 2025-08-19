import pandas as pd
import numpy as np

# load file and drop extra cols
d = pd.read_csv("/content/drive/MyDrive/Datasets/disasters.csv")
dropc = ['DisNo.','External IDs','Classification Key','ISO','Subregion','Region',
'Origin',"Reconstruction Costs ('000 US$)","Insured Damage ('000 US$)",
"Total Damage ('000 US$)",'Entry Date','Last Update']
d = d.drop(columns=[c for c in dropc if c in d.columns])
c25 = ['India','USA','Russia','France','Germany','Italy','China','Japan','Argentina','Portugal',
'Spain','Croatia','Belgium','Australia','Pakistan','Afghanistan','Israel','Iran','Iraq',
'Bangladesh','Sri Lanka','Canada','UK','Sweden','Saudi Arabia']
d = d[d['Country'].isin(c25)]
d = d[d['Start Year']>=2000]
d = d[d['Historic']=='No']
d = d[d['Disaster Group']=='Natural']
numc = ["AID Contribution ('000 US$)",'Magnitude','Total Deaths','No. Injured','No. Affected',
'No. Homeless','Total Affected',"Reconstruction Costs, Adjusted ('000 US$)",
"Insured Damage, Adjusted ('000 US$)","Total Damage, Adjusted ('000 US$)"]
for c in numc:
    if c in d.columns:
        d[c] = pd.to_numeric(d[c],errors='coerce').fillna(0)
        if "000 US$" in c: d[c] *= 1000

# fill missing day/month with 1, calculate duration in days & years
for c in ['Start Month','Start Day','End Month','End Day']:
    if c in d.columns: d[c] = d[c].fillna(1).astype(int)
d['dur_days'] = (pd.to_datetime(d['End Year'].astype(str)+'-'+d['End Month'].astype(str)+'-'+d['End Day'].astype(str))
 - pd.to_datetime(d['Start Year'].astype(str)+'-'+d['Start Month'].astype(str)+'-'+d['Start Day'].astype(str))).dt.days.fillna(0)
d['dur_yrs'] = d['dur_days']/365.25

# tag disasters (climate vs drought)
ctype = ['Drought','Flood','Storm','Extreme temperature','Wildfire']
d['is_climate'] = d['Disaster Type'].isin(ctype)
d['is_drought'] = d['Disaster Type']=='Drought'

# aggregate country-year stats
d['yr'] = d['Start Year']
agg = d.groupby(['Country','yr']).agg(
    n_dis=('Disaster Type','count'),
    n_drought=('is_drought','sum'),
    n_climate=('is_climate','sum'),
    deaths=('Total Deaths','sum'),
    affected=('Total Affected','sum'),
    dmg_usd=("Total Damage, Adjusted ('000 US$)",'sum'),
    recon_usd=("Reconstruction Costs, Adjusted ('000 US$)",'sum'),
    ins_usd=("Insured Damage, Adjusted ('000 US$)",'sum'),
    aid_usd=("AID Contribution ('000 US$)",'sum'),
    mag=('Magnitude','mean'),
    dur=('dur_yrs','mean'),
    intl_appeal=('Appeal',lambda x:(x=='Yes').sum()),
    types=('Disaster Type',lambda x:', '.join(x.unique()))
).reset_index()

# new features for severity & recovery
agg['sev_idx'] = agg['mag'].fillna(0)+np.log(agg['dmg_usd']+1)
agg['recov_score'] = 1/(agg['dur'].fillna(1)+(agg['recon_usd']/(agg['dmg_usd']+1)))
agg = agg.fillna(0)

agg.to_csv("disaster_processed.csv",index=False)
print("disaster_processed.csv saved")
