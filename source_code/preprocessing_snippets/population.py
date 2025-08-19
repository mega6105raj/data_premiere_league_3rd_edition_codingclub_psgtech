import pandas as pd

df=pd.read_csv("/content/drive/MyDrive/Datasets/population_and_demographics.csv")
df=df.drop(columns=["Domain Code","Domain","Area Code (M49)","Element Code","Item Code","Item","Year Code"],errors="ignore")

# only keep chosen 25 countries + a few elements
cts=['India','USA','Russia','France','Germany','Italy','China','Japan','Argentina','Portugal','Spain','Croatia',
     'Belgium','Australia','Pakistan','Afghanistan','Israel','Iran','Iraq','Bangladesh','Sri Lanka','Canada','UK','Sweden','Saudi Arabia']
els=['Total Population - Both sexes','Total Population - Male','Total Population - Female','Rural population','Urban population']
df=df[df["Area"].isin(cts)]
df=df[df["Element"].isin(els)]

# clean values (to numbers) and keep only reliable flagged X rows
df["Value"]=pd.to_numeric(df["Value"],errors="coerce")*1000
df=df.dropna(subset=["Value"])
df=df[df["Flag"]=="X"]

df["Country"]=df["Area"]
p=df.pivot_table(index=["Country","Year"],columns="Element",values="Value",aggfunc="first").reset_index()

p=p.rename(columns={
    "Total Population - Both sexes":"total_population",
    "Total Population - Male":"total_population_male",
    "Total Population - Female":"total_population_female",
    "Rural population":"rural_population",
    "Urban population":"urban_population"
})

# quick new features: urban%, gender ratio, growth, ageing
p["urbanization_pct"]=p["urban_population"]/p["total_population"]*100
p["gender_ratio_male_female"]=p["total_population_male"]/p["total_population_female"]
p=p.sort_values(["Country","Year"])
p["population_growth_pct"]=p.groupby("Country")["total_population"].pct_change()*100
p["ageing_index"]=p["population_growth_pct"].apply(lambda x:"High" if pd.notna(x) and x<0 else "Low")
p["population_growth_pct"]=p["population_growth_pct"].fillna(0)
p["ageing_index"]=p["ageing_index"].fillna("Neutral")

p.to_csv("population_demographics_processed.csv",index=False)
print("done, saved -> population_demographics_processed.csv")
