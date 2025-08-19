import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

cr=pd.read_csv('/content/crop_livestock_processed.csv')
ds=pd.read_csv('/content/disaster_processed.csv')

cr['Country']=cr['Country'].astype(str)
ds['Country']=ds['Country'].astype(str)

cr=cr[['Country','Year','total_area_harvested_ha','avg_crop_yield_kg_ha','est_ag_production_tonnes']]
ds=ds[['Country','Year','num_droughts','severity_index']]

df=pd.merge(cr,ds,on=['Country','Year'],how='left')

# fill missing with ffill then mean
df=df.sort_values(['Country','Year'])
df[['num_droughts','severity_index']]=df.groupby('Country')[['num_droughts','severity_index']].fillna(method='ffill')
df[['num_droughts','severity_index']]=df[['num_droughts','severity_index']].fillna(df[['num_droughts','severity_index']].mean())

top=df.groupby('Country')['total_area_harvested_ha'].sum().nlargest(10).index
df=df[df['Country'].isin(top)]

# make lag features
df['yield_lag1']=df.groupby('Country')['avg_crop_yield_kg_ha'].shift(1)
df['production_lag1']=df.groupby('Country')['est_ag_production_tonnes'].shift(1)
fts=['avg_crop_yield_kg_ha','total_area_harvested_ha','num_droughts','severity_index','yield_lag1','production_lag1']

imp=SimpleImputer(strategy='mean')
df[fts]=imp.fit_transform(df[fts])

# create future rows for 2026-2030 with baseline and drought shocks
rows=[]
for c in top:
  sub=df[df['Country']==c]
  if len(sub)<3: continue
  ly=sub['Year'].max()
  lv=sub[sub['Year']==ly].iloc[0]
  mx=sub['severity_index'].max() if not sub['severity_index'].isna().all() else 1
  for y in range(2026,2031):
    r={'Country':c,'Year':y,'avg_crop_yield_kg_ha':lv['avg_crop_yield_kg_ha'],
       'total_area_harvested_ha':lv['total_area_harvested_ha'],
       'num_droughts':sub['num_droughts'].mean(),
       'severity_index':sub['severity_index'].mean(),
       'yield_lag1':lv['avg_crop_yield_kg_ha'] if y==2026 else np.nan,
       'production_lag1':lv['est_ag_production_tonnes'] if y==2026 else np.nan,
       'est_ag_production_tonnes':np.nan,'scenario':'Baseline'}
    rows.append(r)
    if y>=2028:
      d=r.copy()
      d['num_droughts']=1
      d['severity_index']=mx*2
      d['avg_crop_yield_kg_ha']*=0.8
      d['scenario']='Drought'
      rows.append(d)

fut=pd.DataFrame(rows)
fut[fts]=imp.transform(fut[fts])
df=pd.concat([df,fut],ignore_index=True)

res=[]
for c in top:
  sub=df[df['Country']==c].copy()
  tr=sub[sub['Year']<=ly]
  b=sub[(sub['Year']>=2026)&(sub['scenario']=='Baseline')]
  d=sub[(sub['Year']>=2028)&(sub['scenario']=='Drought')]
  if len(tr)<3: continue
  X=tr[fts]
  y=tr['est_ag_production_tonnes']
  m=RandomForestRegressor(n_estimators=100,random_state=42)
  m.fit(X,y)
  # predict baseline future
  b=b.sort_values('Year')
  for yr in range(2026,2031):
    X_t=b[b['Year']==yr][fts].copy()
    if not X_t.empty:
      X_t[fts]=imp.transform(X_t[fts])
      p=max(0,m.predict(X_t)[0])
      b.loc[b['Year']==yr,'est_ag_production_tonnes']=p
      if yr<2030:
        b.loc[b['Year']==yr+1,'production_lag1']=p
        b.loc[b['Year']==yr+1,'yield_lag1']=b[b['Year']==yr]['avg_crop_yield_kg_ha'].iloc[0]
  # predict drought
  d=d.sort_values('Year')
  for yr in range(2028,2031):
    X_t=d[d['Year']==yr][fts].copy()
    if not X_t.empty:
      X_t[fts]=imp.transform(X_t[fts])
      p=max(0,m.predict(X_t)[0])
      d.loc[d['Year']==yr,'est_ag_production_tonnes']=p
      if yr<2030:
        d.loc[d['Year']==yr+1,'production_lag1']=p
        d.loc[d['Year']==yr+1,'yield_lag1']=d[d['Year']==yr]['avg_crop_yield_kg_ha'].iloc[0]
  bb=b[b['Year']==2030]
  dd=d[d['Year']==2030]
  if not bb.empty and not dd.empty:
    pb=bb['est_ag_production_tonnes'].iloc[0]
    pd=dd['est_ag_production_tonnes'].iloc[0]
    res.append({'Country':c,
                'Baseline Production 2030 (tonnes)':round(pb,2),
                'Drought Production 2030 (tonnes)':round(pd,2),
                'Impact (% Change)':round((pb-pd)/pb*100,2) if pb!=0 else 0})

rf=pd.DataFrame(res)
print("Impact of 3 Consecutive Drought Years on Agricultural Production by 2030 (Random Forest):")
print(rf[['Country','Baseline Production 2030 (tonnes)','Drought Production 2030 (tonnes)','Impact (% Change)']])
rf.to_csv('drought_impact_predictions_2030_rf.csv',index=False)

# prophet check for validation
out=[]
for c in top:
  sub=df[df['Country']==c].copy()
  tr=sub[sub['Year']<=ly]
  b=sub[(sub['Year']>=2026)&(sub['scenario']=='Baseline')]
  d=sub[(sub['Year']>=2028)&(sub['scenario']=='Drought')]
  if len(tr)<3: continue
  pr=tr[['Year','est_ag_production_tonnes','avg_crop_yield_kg_ha','num_droughts','severity_index']].copy()
  pr['ds']=pd.to_datetime(pr['Year'].astype(str)+'-12-31')
  pr['y']=pr['est_ag_production_tonnes']
  m=Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False,
            n_changepoints=max(1,len(pr)//2))
  m.add_regressor('avg_crop_yield_kg_ha')
  m.add_regressor('num_droughts')
  m.add_regressor('severity_index')
  m.fit(pr)
  f_b=b[['Year','avg_crop_yield_kg_ha','num_droughts','severity_index']].copy()
  f_b['ds']=pd.to_datetime(f_b['Year'].astype(str)+'-12-31')
  f_d=d[['Year','avg_crop_yield_kg_ha','num_droughts','severity_index']].copy()
  f_d['ds']=pd.to_datetime(f_d['Year'].astype(str)+'-12-31')
  fb=m.predict(f_b)
  fd=m.predict(f_d)
  bb=fb[fb['ds'].dt.year==2030]
  dd=fd[fd['ds'].dt.year==2030]
  if not bb.empty and not dd.empty:
    pb=max(0,bb['yhat'].iloc[0])
    pd=max(0,dd['yhat'].iloc[0])
    out.append({'Country':c,
                'Baseline Production 2030 (tonnes, Prophet)':round(pb,2),
                'Drought Production 2030 (tonnes, Prophet)':round(pd,2),
                'Impact (% Change, Prophet)':round((pb-pd)/pb*100,2) if pb!=0 else 0})

pf=pd.DataFrame(out)
print("\nImpact of 3 Consecutive Drought Years on Agricultural Production by 2030 (Prophet):")
print(pf[['Country','Baseline Production 2030 (tonnes, Prophet)','Drought Production 2030 (tonnes, Prophet)','Impact (% Change, Prophet)']])
pf.to_csv('drought_impact_predictions_2030_prophet.csv',index=False)
