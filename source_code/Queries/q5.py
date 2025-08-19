import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

emp=pd.read_csv('/content/processed_employment_unemployment.csv')
eco=pd.read_csv('/content/cleaned_core_economic_indicators (1).csv')

# filter out youth unemployment rows
yu=emp[emp['Series Code']=='SL.UEM.1524.ZS'].copy()
yu=yu[['Country Code','Country Name','Year','Value']]
yu.rename(columns={'Value':'youth_unemployment'},inplace=True)

eco_f=eco[['Country Code','Year','gdp_growth_pct','inflation_consumer_prices_pct',
           'trade_balance_gdp_pct','gdp_per_capita_current_usd']].copy()

# merge both employment + economic features into one dataset
df=pd.merge(yu,eco_f,on=['Country Code','Year'],how='left')
df=df.sort_values(['Country Code','Year'])
df['unemployment_lag1']=df.groupby('Country Code')['youth_unemployment'].shift(1)
df['unemployment_lag2']=df.groupby('Country Code')['youth_unemployment'].shift(2)

fts=['gdp_growth_pct','inflation_consumer_prices_pct','trade_balance_gdp_pct',
     'gdp_per_capita_current_usd','unemployment_lag1','unemployment_lag2']
imp=SimpleImputer(strategy='mean')
df[fts]=imp.fit_transform(df[fts])

# here we generate fake "future years" (2026-2030) under a slowdown assumption
cnts=df['Country Code'].unique()
lst=[]
for c in cnts:
  sub=df[df['Country Code']==c]
  ly=sub['Year'].max()
  lv=sub[sub['Year']==ly].iloc[0]
  for y in range(2026,2031):
    row={'Country Code':c,'Country Name':lv['Country Name'],'Year':y,
         'gdp_growth_pct':lv['gdp_growth_pct']*0.5,
         'inflation_consumer_prices_pct':lv['inflation_consumer_prices_pct']*2,
         'trade_balance_gdp_pct':lv['trade_balance_gdp_pct'],
         'gdp_per_capita_current_usd':lv['gdp_per_capita_current_usd'],
         'unemployment_lag1':lv['youth_unemployment'] if y==2026 else np.nan,
         'unemployment_lag2':lv['unemployment_lag1'] if y==2026 else np.nan,
         'youth_unemployment':np.nan}
    lst.append(row)

fut=pd.DataFrame(lst)
df=pd.concat([df,fut],ignore_index=True)
preds=[]
for c in cnts:
  sub=df[df['Country Code']==c].copy()
  ly=sub['Year'].max()
  tr=sub[sub['Year']<=ly]
  tst=sub[sub['Year']>=2026]
  if len(tr)<3: continue

  X=tr[fts]
  y=tr['youth_unemployment']

  # train a simple RandomForest for each country
  m=RandomForestRegressor(n_estimators=100,random_state=42)
  m.fit(X,y)

  tst=tst.sort_values('Year')
  for yr in range(2026,2031):
    X_t=tst[tst['Year']==yr][fts].copy()
    if not X_t.empty:
      X_t[fts]=imp.transform(X_t[fts])
      p=max(0,min(100,m.predict(X_t)[0]))  # keep inside 0-100%
      tst.loc[tst['Year']==yr,'youth_unemployment']=p
      # updating lag values so next year prediction can chain properly
      if yr<2030:
        tst.loc[tst['Year']==yr+1,'unemployment_lag1']=p
        tst.loc[tst['Year']==yr+1,'unemployment_lag2']=tst[tst['Year']==yr]['unemployment_lag1'].iloc[0] if yr==2026 else tst[tst['Year']==yr]['youth_unemployment'].iloc[0]

  res=tst[tst['Year']==2030]
  if not res.empty:
    preds.append({'Country Name':res['Country Name'].iloc[0],
                  'Country Code':c,
                  'Youth Unemployment 2030':res['youth_unemployment'].iloc[0]})

out=pd.DataFrame(preds)
hi=out[out['Youth Unemployment 2030']>25]

print("Countries with predicted youth unemployment > 25% in 2030 under global slowdown scenario:")
print(hi[['Country Name','Youth Unemployment 2030']])
out.to_csv('youth_unemployment_predictions_2030.csv',index=False)
