import pandas as pd
import numpy as np

fp='Core_economic_indicators (1).csv'
d=pd.read_csv(fp)

# drop junk rows
d=d.dropna(subset=['Country Name'])
d=d[~d['Country Name'].str.contains('Data from database|Last Updated',na=False)]

yrs=[c for c in d.columns if '[YR' in c]
ymap={c:c.split()[0] for c in yrs}
d.rename(columns=ymap,inplace=True)

idv=['Country Name','Country Code','Series Name','Series Code']
valv=[str(y) for y in range(2000,2025)]
dl=pd.melt(d,id_vars=idv,value_vars=valv,var_name='Year',value_name='val')

dl['val']=dl['val'].replace('..',np.nan)  # replace weird ..
dl['val']=pd.to_numeric(dl['val'],errors='coerce')

# pivot to wide
dp=dl.pivot_table(index=['Country Name','Country Code','Year'],columns='Series Code',values='val').reset_index()

cr={'NE.IMP.GNFS.ZS':'imports_goods_services_gdp_pct','NE.EXP.GNFS.ZS':'exports_goods_services_gdp_pct',
'NE.TRD.GNFS.ZS':'trade_gdp_pct','FP.CPI.TOTL.ZG':'inflation_consumer_prices_pct','NY.GDP.MKTP.KD.ZG':'gdp_growth_pct',
'NY.GDP.PCAP.CD':'gdp_per_capita_current_usd','NY.GDP.MKTP.CD':'gdp_current_usd'}
dp.rename(columns=cr,inplace=True)

dp['Year']=dp['Year'].astype(int)
numc=list(cr.values())
dp[numc]=dp[numc].astype(float)

# focus only on some countries
cc=['India','USA','Russia','France','Germany','Italy','China','Japan','Argentina','Portugal','Spain','Croatia','Belgium',
'Australia','Pakistan','Afghanistan','Israel','Iran','Iraq','Bangladesh','Sri Lanka','Canada','UK','Sweden','Saudi Arabia']
dp=dp[dp['Country Name'].isin(cc)]

iso={'USA':'USA','RUS':'RUS','FRA':'FRA','DEU':'DEU','ITA':'ITA','CHN':'CHN','JPN':'JPN','ARG':'ARG','PRT':'PRT','ESP':'ESP',
'HRV':'HRV','BEL':'BEL','AUS':'AUS','PAK':'PAK','AFG':'AFG','ISR':'ISR','IRN':'IRN','IRQ':'IRQ','BGD':'BGD','LKA':'LKA','CAN':'CAN','GBR':'GBR','SWE':'SWE','SAU':'SAU'}
dp['ISO']=dp['Country Code'].map(iso).fillna(dp['Country Code'])

dp.drop_duplicates(subset=['Country Name','Year'],inplace=True)
dp['is_outlier']=((dp['gdp_growth_pct']>20)|(dp['gdp_growth_pct']<-20)).astype(int) # flag crazy values

dp.sort_values(['Country Name','Year'],inplace=True)
dp[numc]=dp.groupby('Country Name')[numc].transform(lambda x:x.interpolate(method='linear'))
dp[numc]=dp.groupby('Country Name')[numc].transform(lambda x:x.fillna(x.mean()))  # still missing? use mean

sl=['Country Name','Country Code','ISO','Year','imports_goods_services_gdp_pct','exports_goods_services_gdp_pct',
'trade_gdp_pct','inflation_consumer_prices_pct','gdp_growth_pct','gdp_per_capita_current_usd','gdp_current_usd']
dc=dp[sl].copy()

dc['trade_balance_gdp_pct']=dc['exports_goods_services_gdp_pct']-dc['imports_goods_services_gdp_pct']
dc['gdp_growth_stability']=dc.groupby('Country Name')['gdp_growth_pct'].transform(lambda x:x.rolling(window=3,min_periods=1).std())
dc['inflation_stability']=dc.groupby('Country Name')['inflation_consumer_prices_pct'].transform(lambda x:x.rolling(window=3,min_periods=1).std())
dc['log_gdp_current_usd']=np.log1p(dc['gdp_current_usd'])
dc['log_gdp_per_capita']=np.log1p(dc['gdp_per_capita_current_usd'])

# temp placeholders
dc['trade_vulnerability_index']=np.nan
dc['economic_shock_sensitivity']=0.4*dc['gdp_growth_stability']+0.4*dc['inflation_stability']+0.2*dc['trade_balance_gdp_pct']
dc['per_capita_trade_intensity']=np.nan

dc['data_source']='core_economic_indicators'

print("shp:",dc.shape)
print("nulls:\n",dc.isnull().sum()) # quick check
print("desc:\n",dc.describe())

op='cleaned_core_economic_indicators.csv'
dc.to_csv(op,index=False)
print("saved:",op)
