import pandas as pd, numpy as np, tensorflow as tf, optuna
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# load data
exp=pd.read_csv("export_processed.csv",encoding="latin1");integ=pd.read_csv("integrated_master.csv",encoding="latin1")
ctrys=["India","USA","Russia","France","Germany","Italy","China","Japan","Argentina","Portugal","Spain","Croatia","Belgium","Australia","Pakistan","Afghanistan","Israel","Iran","Iraq","Bangladesh","Sri Lanka","Canada","UK","Sweden","Saudi Arabia"]
exp=exp[exp["reporterDesc"].isin(ctrys)]

# find vuln
mx=exp.groupby(["reporterDesc","refYear"])["TradeDependencyIndex"].max().reset_index()
av=mx.groupby("reporterDesc")["TradeDependencyIndex"].mean().reset_index()
t3=av.sort_values(by="TradeDependencyIndex",ascending=False).head(3)
print("\nTop 3 vulnerable (by avg yearly max TDI):");print(t3)

# find top partners per country
tp=[]
for c in t3["reporterDesc"]:
    s=exp[exp["reporterDesc"]==c];yr=s["refYear"].max();sl=s[s["refYear"]==yr]
    p=sl.sort_values("TradeDependencyIndex",ascending=False).iloc[0]
    tp.append({"country":c,"partner":p["partnerDesc"],"partner_export_share":p["TradeDependencyIndex"],"year_basis":yr})
tpd=pd.DataFrame(tp)
print("\nTop partner & export share (latest year):");print(tpd)

# helper for sequence making
def mkseq(X,y,w=3):
    xx,yy=[],[]
    for i in range(len(X)-w): xx.append(X[i:i+w]);yy.append(y[i+w])
    return np.array(xx),np.array(yy)

# optuna obj (bad naming, minimal doc)
def obj(tr,Xtr,ytr,Xv,yv,inp):
    u=tr.suggest_int("units",32,128);l=tr.suggest_int("layers",1,3);d=tr.suggest_float("dropout",0.1,0.5);lr=tr.suggest_loguniform("lr",1e-4,1e-2)
    m=keras.Sequential()
    for i in range(l):
        rs=(i<l-1)
        m.add(layers.LSTM(u,activation="tanh",return_sequences=rs,input_shape=inp if i==0 else None))
        m.add(layers.Dropout(d))
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss="mse")
    es=keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    h=m.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=100,batch_size=tr.suggest_categorical("batch_size",[8,16,32]),verbose=0,callbacks=[es])
    return min(h.history["val_loss"])

# run model per country
smry=[]
feats=["imports_goods_services_gdp_pct","exports_goods_services_gdp_pct","trade_gdp_pct","inflation_consumer_prices_pct","gdp_growth_pct","trade_balance_gdp_pct","gdp_growth_stability","inflation_stability","HHI_export","Diversification_export","HHI_import","Diversification_import","Trade_Diversification_Index","Overall_Trade_Dependency"]
tgt="gdp_current_usd"
for _,r in tpd.iterrows():
    c=r["country"];p=r["partner"];shr=r["partner_export_share"]
    print(f"\n==============================");print(f"Country: {c} | Top partner: {p} ({shr:.2f})");print("==============================")
    df=integ[integ["Country Name"]==c].copy();df=df.dropna(subset=feats+[tgt])
    if df.empty or df["Year"].nunique()<10: print(f"Not enough data for {c}. Skipping.");continue
    sx=MinMaxScaler();sy=MinMaxScaler();X=sx.fit_transform(df[feats]);y=sy.fit_transform(df[[tgt]])
    Xs,ys=mkseq(X,y,3)

    if len(Xs)<10: 
        print(f"Too short sequence for {c}. Skipping.");
        continue
    yrs=df["Year"].values[3:]
    trcut=np.where(yrs<=2022)[0][-1]
    vcut=np.where(yrs<=2024)[0][-1]
    Xtr,Ytr=Xs[:trcut+1],ys[:trcut+1]
    Xv,Yv=Xs[trcut+1:vcut+1],ys[trcut+1:vcut+1]
    Xt,Yt=Xs[vcut+1:],ys[vcut+1:]

    st=optuna.create_study(direction="minimize");st.optimize(lambda tr: obj(tr,Xtr,Ytr,Xv,Yv,(Xtr.shape[1],Xtr.shape[2])),n_trials=20,timeout=300)
    bp=st.best_params
    print("Best params:",bp)
    bm=keras.Sequential()
    
    for i in range(bp["layers"]):
        rs=(i<bp["layers"]-1)
        bm.add(layers.LSTM(bp["units"],activation="tanh",return_sequences=rs,input_shape=(Xtr.shape[1],Xtr.shape[2]) if i==0 else None))
        bm.add(layers.Dropout(bp["dropout"]))
    bm.add(layers.Dense(1));bm.compile(optimizer=keras.optimizers.Adam(learning_rate=bp["lr"]),loss="mse")
    es=keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    bm.fit(np.concatenate([Xtr,Xv]),np.concatenate([Ytr,Yv]),epochs=200,batch_size=bp["batch_size"],verbose=0,callbacks=[es])
    ls=Xs[-1:];bl=bm.predict(ls,verbose=0)[0][0];blv=sy.inverse_transform([[bl]])[0,0]
    sh=ls.copy();idx=feats.index("Overall_Trade_Dependency");sh[0,-1,idx]*=(1-0.40*shr)
    shv=bm.predict(sh,verbose=0)[0][0];shv=sy.inverse_transform([[shv]])[0,0]
    loss=blv-shv;lp=loss/blv*100
    smry.append({"country":c,"top_partner":p,"baseline_gdp_2026":blv,"shock_gdp_2026":shv,"gdp_loss":loss,"gdp_loss_pct":lp})

# print results
if smry:
    sm=pd.DataFrame(smry).sort_values("gdp_loss_pct",ascending=False)
    pd.set_option("display.float_format",lambda v:f"{v:,.2f}")
    print("\n===== GDP Impact Summary (2026, 40% drop in top partner's imports, Tuned LSTM) =====");print(sm)
else: print("\n No valid countries had enough data for tuned LSTM modeling.")
