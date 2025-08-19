import pandas as pd
import numpy as np
import tensorflow as tf
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# load data & filter
e = pd.read_csv("export_processed.csv", encoding="latin1")
i = pd.read_csv("integrated_master.csv", encoding="latin1")
cs = ["India","USA","Russia","France","Germany","Italy","China","Japan","Argentina","Portugal","Spain","Croatia","Belgium","Australia","Pakistan","Afghanistan","Israel","Iran","Iraq","Bangladesh","Sri Lanka","Canada","UK","Sweden","Saudi Arabia"]
e = e[e["reporterDesc"].isin(cs)]

# vuln calc
mx = e.groupby(["reporterDesc","refYear"])["TradeDependencyIndex"].max().reset_index()
av = mx.groupby("reporterDesc")["TradeDependencyIndex"].mean().reset_index()
t3 = av.sort_values(by="TradeDependencyIndex", ascending=False).head(3)
print("\nTop 3 vulnerable (by avg yearly max TDI):")
print(t3)

# partner pick
tp = []
for c in t3["reporterDesc"]:
    s = e[e["reporterDesc"]==c]
    yr = s["refYear"].max()
    sl = s[s["refYear"]==yr]
    p = sl.sort_values("TradeDependencyIndex", ascending=False).iloc[0]
    tp.append({"c":c,"p":p["partnerDesc"],"shr":p["TradeDependencyIndex"],"y":yr})

tpd = pd.DataFrame(tp)
print("\nTop partner & export share (latest year):")
print(tpd)

# seq maker
def ms(X,y,w=3):
    xx,yy = [],[]
    for k in range(len(X)-w):
        xx.append(X[k:k+w])
        yy.append(y[k+w])
    return np.array(xx),np.array(yy)

# optuna obj
def o(tr,Xtr,ytr,Xv,yv,inp):
    u = tr.suggest_int("u",32,128)
    l = tr.suggest_int("l",1,3)
    d = tr.suggest_float("d",0.1,0.5)
    lr = tr.suggest_loguniform("lr",1e-4,1e-2)
    m = keras.Sequential()
    for j in range(l):
        rs = (j<l-1)
        if j==0:
            m.add(layers.LSTM(u,activation="tanh",return_sequences=rs,input_shape=inp))
        else:
            m.add(layers.LSTM(u,activation="tanh",return_sequences=rs))
        m.add(layers.Dropout(d))
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss="mse")
    es = keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    h = m.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=100,batch_size=tr.suggest_categorical("bs",[8,16,32]),verbose=0,callbacks=[es])
    return min(h.history["val_loss"])

# run per country
sm = []
fs = ["imports_goods_services_gdp_pct","exports_goods_services_gdp_pct","trade_gdp_pct","inflation_consumer_prices_pct","gdp_growth_pct","trade_balance_gdp_pct","gdp_growth_stability","inflation_stability","HHI_export","Diversification_export","HHI_import","Diversification_import","Trade_Diversification_Index","Overall_Trade_Dependency"]
tg = "gdp_current_usd"

for _,r in tpd.iterrows():
    c = r["c"]
    p = r["p"]
    shr = r["shr"]
    print(f"\n==============================")
    print(f"Country: {c} | Top partner: {p} ({shr:.2f})")
    print("==============================")
    d = i[i["Country Name"]==c].copy()
    d = d.dropna(subset=fs+[tg])
    if d.empty or d["Year"].nunique()<10:
        print(f"Not enough data for {c}. Skipping.")
        continue
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    X = sx.fit_transform(d[fs])
    y = sy.fit_transform(d[[tg]])
    Xs,ys = ms(X,y,3)
    if len(Xs)<10:
        print(f"Too short sequence for {c}. Skipping.")
        continue
    yrs = d["Year"].values[3:]
    trc = np.where(yrs<=2022)[0][-1]
    vc = np.where(yrs<=2024)[0][-1]
    Xtr,Ytr = Xs[:trc+1],ys[:trc+1]
    Xv,Yv = Xs[trc+1:vc+1],ys[trc+1:vc+1]
    Xt,Yt = Xs[vc+1:],ys[vc+1:]
    st = optuna.create_study(direction="minimize")
    st.optimize(lambda tr:o(tr,Xtr,Ytr,Xv,Yv,(Xtr.shape[1],Xtr.shape[2])),n_trials=20,timeout=300)
    bp = st.best_params
    print("Best params:",bp)
    bm = keras.Sequential()
    for j in range(bp["l"]):
        rs = (j<bp["l"]-1)
        if j==0:
            bm.add(layers.LSTM(bp["u"],activation="tanh",return_sequences=rs,input_shape=(Xtr.shape[1],Xtr.shape[2])))
        else:
            bm.add(layers.LSTM(bp["u"],activation="tanh",return_sequences=rs))
        bm.add(layers.Dropout(bp["d"]))
    bm.add(layers.Dense(1))
    bm.compile(optimizer=keras.optimizers.Adam(learning_rate=bp["lr"]),loss="mse")
    es = keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    bm.fit(np.concatenate([Xtr,Xv]),np.concatenate([Ytr,Yv]),epochs=200,batch_size=bp["bs"],verbose=0,callbacks=[es])
    ls = Xs[-1:]
    bl = bm.predict(ls,verbose=0)[0][0]
    blv = sy.inverse_transform([[bl]])[0,0]
    sh = ls.copy()
    idx = fs.index("Overall_Trade_Dependency")
    sh[0,-1,idx] *= (1-0.40*shr)
    shv = bm.predict(sh,verbose=0)[0][0]
    shv = sy.inverse_transform([[shv]])[0,0]
    loss = blv-shv
    lp = loss/blv*100
    sm.append({"c":c,"tp":p,"bl":blv,"sh":shv,"loss":loss,"lp":lp})

if sm:
    s = pd.DataFrame(sm).sort_values("lp",ascending=False)
    pd.set_option("display.float_format",lambda v:f"{v:,.2f}")
    print("\n===== GDP Impact Summary (2026, 40% drop in top partner's imports, Tuned LSTM) =====")
    print(s)
else:
    print("\n No valid countries had enough data for tuned LSTM modeling.")
