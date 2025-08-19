'''
    This code is kindof separate implementations of attempst for Q9, Q10, Q11, Q12. 
    Individual implementation caused more running time for each. Inorder to prevent that 
    an attempt was made to integrate. But the total execution caused even more running time even in T4 GPU.
    Attempts to lower the running time caused some errors to be raised. 
    But the implementation can be split to get he four parts running
'''
import pandas as pd
import numpy as np
import warnings, math
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna

SEQ_LEN = 5
N_TRIALS = 20
EPOCHS_TUNE = 10
EPOCHS_FINAL = 25
BATCH_SIZE = 32
RANDOM_STATE = 42
#Good standards maintained as in the column headers
TARGETS = {
    "gdp_current_usd": "GDP (current US$)",
    "poverty_rate_pct": "Poverty rate (%)",        
    "unemployment_total_pct": "Unemployment rate (%)"
}
PREFERRED_FEATURES = [
    'Trade_Diversification_Index', 'HHI_export', 'HHI_import',
    'Diversification_export', 'Diversification_import', 'Overall_Trade_Dependency',
    'trade_gdp_pct', 'imports_goods_services_gdp_pct', 'exports_goods_services_gdp_pct',
    'gdp_growth_pct', 'gdp_per_capita_current_usd', 'inflation_consumer_prices_pct',
    'log_gdp_current_usd', 'log_gdp_per_capita', 'economic_shock_sensitivity',
    'TradeDependencyIndex_mean', 'TradeDependencyIndex_max',
    'CountryTotalPrimaryValue', 'CountryTotalFOBValue',
    'Current account balance (% of GDP)',
    'disaster_severity' 
]
export_df = pd.read_csv('/content/export_processed.csv', encoding='latin1')
master_df = pd.read_csv('/content/integrated_master.csv', encoding='latin1')
res_df    = pd.read_csv('/content/processed_resiliance.csv', encoding='latin1')


#columns in export_df: reporterISO, refYear, TradeDependencyIndex, CountryTotalPrimaryValue, CountryTotalFOBValue
#aggregate to get country-year level signals
if 'reporterISO' in export_df.columns and 'refYear' in export_df.columns:
    exp = export_df.rename(columns={'reporterISO':'Country Code', 'refYear':'Year'})
else:
    raise ValueError("export_processed.csv must have 'reporterISO' and 'refYear'")

agg = {
    'TradeDependencyIndex': ['mean','max'],
}
for c in ['CountryTotalPrimaryValue','CountryTotalFOBValue']:
    if c in export_df.columns:
        agg[c] = 'sum'

export_country_year = export_df.groupby(['reporterISO','refYear']).agg(agg)
export_country_year.columns = ['_'.join([a,b]) if isinstance(b,str) else a for a,b in export_country_year.columns.ravel()]
export_country_year = export_country_year.reset_index().rename(columns={
    'reporterISO':'Country Code','refYear':'Year',
    'TradeDependencyIndex_mean':'TradeDependencyIndex_mean',
    'TradeDependencyIndex_max':'TradeDependencyIndex_max'
})

desired_series = ['Current account balance (% of GDP)']
res_keep = res_df[res_df['Series Name'].isin(desired_series)].copy()

if {'Country Code','Year','Series Name','Value'}.issubset(res_keep.columns):
    res_pivot = res_keep.pivot_table(index=['Country Code','Year'],
                                     columns='Series Name',
                                     values='Value',
                                     aggfunc='mean').reset_index()
else:
    res_pivot = pd.DataFrame(columns=['Country Code','Year'] + desired_series)
if not {'Country Code','Year'}.issubset(master_df.columns):
    if {'ISO','Year'}.issubset(master_df.columns):
        master_df = master_df.rename(columns={'ISO':'Country Code'})
    else:
        raise ValueError("integrated_master.csv must have 'Country Code' and 'Year' (or 'ISO' + 'Year').")
for col in ['poverty_rate_pct','unemployment_total_pct']:
    if col not in master_df.columns:
        master_df[col] = np.nan 
if 'disaster_severity' not in master_df.columns:
    master_df['disaster_severity'] = 0.0

#Merging logic 
df = master_df.merge(export_country_year, on=['Country Code','Year'], how='left') \
              .merge(res_pivot,          on=['Country Code','Year'], how='left')

feature_cols = [c for c in PREFERRED_FEATURES if c in df.columns]
if len(feature_cols) == 0:
    raise ValueError("No usable features.")
df = df.sort_values(['Country Code','Year']).reset_index(drop=True)

# Forward-fill time features by country for stability (As explained in the README)
for col in feature_cols:
    df[col] = df.groupby('Country Code')[col].ffill()
for col in feature_cols:
    if df[col].isna().any():
        # fill per-country median, then global median
        df[col] = df.groupby('Country Code')[col].apply(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

#Basic helper functions
def build_sequences(panel, seq_len, feature_cols, target_col):
    X_list, y_list = [], []
    for cc, cdf in panel.groupby('Country Code'):
        cdf = cdf.sort_values('Year')
        if target_col not in cdf.columns:
            continue
        vals = cdf[feature_cols].values
        tgt  = cdf[target_col].values
        if len(cdf) < seq_len + 1:
            continue
        for i in range(len(cdf) - seq_len):
            xt = vals[i:i+seq_len, :]
            yt = tgt[i+seq_len]
            if not np.isnan(yt):
                X_list.append(xt)
                y_list.append(yt)
    if len(X_list)==0:
        return np.empty((0,seq_len,len(feature_cols))), np.array([])
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y

def build_lstm(trial, seq_len: int, n_features: int):
    units = trial.suggest_int('units', 32, 128)
    layers = trial.suggest_int('layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=(layers>1), input_shape=(seq_len, n_features)))
    model.add(Dropout(dropout))
    for i in range(layers-1):
        model.add(LSTM(units=units, return_sequences=(i < layers-2)))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def fit_target_model(panel, feature_cols, target_col, seq_len=SEQ_LEN):
    # scale features globally (minmax)
    scaler = MinMaxScaler()
    scaled = panel.copy()
    scaled[feature_cols] = scaler.fit_transform(panel[feature_cols].astype(float))

    # build sequences
    X, y = build_sequences(scaled, seq_len, feature_cols, target_col)
    if X.size == 0 or y.size == 0:
        print(f"[WARN] No training sequences for target '{target_col}'. Skipping model.")
        return None, scaler

    #Train, test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    def objective(trial):
        model = build_lstm(trial, seq_len, len(feature_cols))
        model.fit(Xtr, ytr, epochs=EPOCHS_TUNE, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)
        loss = model.evaluate(Xte, yte, verbose=0)
        return loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial.params

    # final model
    class _FT(optuna.trial.Trial):
        def __init__(self, params): self.params = params
        def suggest_int(self, k,a,b): return self.params[k]
        def suggest_float(self, k,a,b,log=False): return self.params[k]
    final_model = build_lstm(_FT(best), seq_len, len(feature_cols))
    final_model.fit(Xtr, ytr, epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    test_loss = final_model.evaluate(Xte, yte, verbose=0)
    print(f"Trained '{target_col}' – test MSE: {test_loss:.4e}, best params: {best}")
    return final_model, scaler

models = {}
scalers = {}

for col, nice in TARGETS.items():
    if col in df.columns and (~df[col].isna()).sum() >= 100:  # need some signal
        m, s = fit_target_model(df, feature_cols, col, seq_len=SEQ_LEN)
        models[col] = m
        scalers[col] = s
    else:
        print(f"Target '{col}' not found or too sparse. It will be skipped.")
def clamp01(x): return max(0.0, min(1.0, float(x)))

def build_future_panel(panel, feature_cols, until_year=2030):
    fut_rows = []
    for cc, cdf in panel.groupby('Country Code'):
        cdf = cdf.sort_values('Year')
        last_year = int(cdf['Year'].max())
        last = cdf.iloc[-1].copy()
        if 'disaster_severity' not in cdf.columns:
            cdf['disaster_severity'] = 0.0
            last['disaster_severity'] = 0.0
        for y in range(last_year+1, until_year+1):
            row = last.copy()
            row['Year'] = y
            fut_rows.append(row)
    fut = pd.DataFrame(fut_rows)
    full = pd.concat([panel, fut], ignore_index=True)
    # Nan filling logic as present in other codes
    for c in feature_cols:
        full[c] = full.groupby('Country Code')[c].ffill()
        full[c] = full[c].fillna(full[c].median())
    return full

def apply_disaster_2026(future_df):
    if 'disaster_severity' in future_df.columns:
        future_df.loc[future_df['Year'] == 2026, 'disaster_severity'] = np.maximum(
            future_df.loc[future_df['Year'] == 2026, 'disaster_severity'].fillna(0), 9.0)
    if 'gdp_growth_pct' in future_df.columns:
        future_df.loc[future_df['Year'] >= 2026, 'gdp_growth_pct'] *= 0.9 # For small dagging
    return future_df

def apply_trade_war_2027(future_df):
    #-20% trade volume proxy
    for c in ['trade_gdp_pct','imports_goods_services_gdp_pct','exports_goods_services_gdp_pct']:
        if c in future_df.columns:
            future_df.loc[future_df['Year'] >= 2027, c] *= 0.8
    #increase concentration / reduce diversification 
    if 'Trade_Diversification_Index' in future_df.columns:
        future_df.loc[future_df['Year'] >= 2027, 'Trade_Diversification_Index'] *= 0.9
    for hh in ['HHI_export','HHI_import']:
        if hh in future_df.columns:
            future_df.loc[future_df['Year'] >= 2027, hh] *= 1.1
    return future_df

def apply_best_case_2030(future_df):
  #LOwers the concentaration for poor performance
    if 'Trade_Diversification_Index' in future_df.columns:
        future_df.loc[future_df['Year'] == 2030, 'Trade_Diversification_Index'] *= 1.15
    for hh in ['HHI_export','HHI_import']:
        if hh in future_df.columns:
            future_df.loc[future_df['Year'] == 2030, hh] *= 0.85
    if 'Current account balance (% of GDP)' in future_df.columns:
        future_df.loc[future_df['Year'] == 2030, 'Current account balance (% of GDP)'] *= 1.10
    if 'disaster_severity' in future_df.columns:
        future_df.loc[future_df['Year'] == 2030, 'disaster_severity'] = 0.0
    return future_df

def apply_worst_case_2030(future_df):
    # Recurring disasters + concentration
    if 'disaster_severity' in future_df.columns:
        future_df.loc[future_df['Year'].between(2026,2030), 'disaster_severity'] = 8.5
    if 'Trade_Diversification_Index' in future_df.columns:
        future_df.loc[future_df['Year'] == 2030, 'Trade_Diversification_Index'] *= 0.85
    for hh in ['HHI_export','HHI_import']:
        if hh in future_df.columns:
            future_df.loc[future_df['Year'] == 2030, hh] *= 1.15
    return future_df

def predict_2030(panel_base, models, scalers, feature_cols, seq_len=SEQ_LEN):
    out = {}
    for tgt, model in models.items():
        if model is None:
            continue
        res = []
        # scaling features with training scaler
        scl = scalers[tgt]
        scaled = panel_base.copy()
        scaled[feature_cols] = scl.transform(scaled[feature_cols].astype(float))
        for cc, cdf in scaled.groupby('Country Code'):
            cdf = cdf.sort_values('Year')
            # build last sequence ending at 2030-1 (years t-4..t for seq_len=5)
            hist = cdf[cdf['Year'] <= 2030]
            if len(hist) < seq_len:
                continue
            seq = hist[feature_cols].iloc[-seq_len:].values
            seq = np.expand_dims(seq, 0)
            pred = float(model.predict(seq, verbose=0).ravel()[0])
            res.append({'Country Code': cc, 'Year': 2030, f'pred_{tgt}': pred})
        out[tgt] = pd.DataFrame(res)
    return out

#Future forcasting
future_base = build_future_panel(df, feature_cols, until_year=2030)

#Q10a
future_disaster = apply_disaster_2026(future_base.copy())
pred_disaster = predict_2030(future_disaster, models, scalers, feature_cols)

#Q10b
future_tradewar = apply_trade_war_2027(future_base.copy())
pred_tradewar = predict_2030(future_tradewar, models, scalers, feature_cols)

#Q11a
future_best = apply_best_case_2030(future_base.copy())
pred_best = predict_2030(future_best, models, scalers, feature_cols)

#Q11b
future_worst = apply_worst_case_2030(future_base.copy())
pred_worst = predict_2030(future_worst, models, scalers, feature_cols)

#For Q10/11 collating teh outpts
def collect_outputs(label, dict_preds):
    # merge across targets if multiple were trained
    keys = list(dict_preds.keys())
    if not keys:
        return pd.DataFrame(columns=['Country Code','Year', 'Scenario'])
    merged = None
    for k in keys:
        dfk = dict_preds[k].copy()
        if merged is None: merged = dfk
        else: merged = merged.merge(dfk, on=['Country Code','Year'], how='outer')
    merged['Scenario'] = label
    return merged

q10a_df = collect_outputs('Disaster_2026', pred_disaster)
q10b_df = collect_outputs('TradeWar_2027', pred_tradewar)
q11a_df = collect_outputs('BestCase_2030',  pred_best)
q11b_df = collect_outputs('WorstCase_2030', pred_worst)

#Mapping with country names
if 'Country Name' in master_df.columns:
    for d in [q10a_df, q10b_df, q11a_df, q11b_df]:
        d['Country Name'] = d['Country Code'].map(master_df.set_index('Country Code')['Country Name'])

q10a_df.to_csv('/content/Q10a_disaster_2026_predictions_2030.csv', index=False)
q10b_df.to_csv('/content/Q10b_tradewar_2027_predictions_2030.csv', index=False)
q11a_df.to_csv('/content/Q11a_bestcase_2030_predictions.csv', index=False)
q11b_df.to_csv('/content/Q11b_worstcase_2030_predictions.csv', index=False)

print("Saved Q10/Q11 scenario outputs.")
# Q12
RES_MAP = {
    'Current account balance (% of GDP)': +1,
}

def zscore_by_year(df_, col):
    df_ = df_.copy()
    df_['__z__'] = df_.groupby('Year')[col].transform(
        lambda v: (v - v.mean())/v.std(ddof=0) if v.std(ddof=0) else 0.0
    )
    return df_['__z__']

def minmax_by_year(df_, col):
    df_ = df_.copy()
    vmin = df_.groupby('Year')[col].transform('min')
    vmax = df_.groupby('Year')[col].transform('max')
    out = (df_[col] - vmin) / (vmax - vmin)
    out[(vmax - vmin)==0] = 0.5
    return out

# Start from best-case future (investments)
res_2030_base = future_best[future_best['Year'] == 2030].copy()
if res_2030_base.empty:
    print("[WARN] No rows for 2030 in future panel; cannot compute resilience rankings.")
    top5 = pd.DataFrame(columns=['Country Code','Country Name','Resilience_Composite_2030'])
else:
    comp_list = []
    for series, direction in RES_MAP.items():
        if series in res_2030_base.columns:
            tmp = res_2030_base[['Country Code','Country Name','Year', series]].copy()
            tmp['z'] = zscore_by_year(tmp.rename(columns={series:'val'}), 'val') * direction
            comp_list.append(tmp[['Country Code','Year','z']].rename(columns={'z':f'Z_{series}'}))
    if comp_list:
        comp = comp_list[0]
        for c in comp_list[1:]:
            comp = comp.merge(c, on=['Country Code','Year'], how='outer')
        # average Z, then minmax to 0..1
        z_cols = [c for c in comp.columns if c.startswith('Z_')]
        comp['Z_mean'] = comp[z_cols].mean(axis=1)
        comp = comp.merge(res_2030_base[['Country Code','Country Name','Year']], on=['Country Code','Year'], how='left').drop_duplicates()
        comp['Resilience_Composite_2030'] = minmax_by_year(comp, 'Z_mean')
        comp = comp.sort_values('Resilience_Composite_2030', ascending=False)
        top5 = comp[['Country Code','Country Name','Resilience_Composite_2030']].head(5)
    else:
        print("No resilience series available to compute composite.")
        top5 = pd.DataFrame(columns=['Country Code','Country Name','Resilience_Composite_2030'])

#show which features most improved under bestCase vs wOrstcase in 2030 for the top countries
drivers = []
if not top5.empty:
    best_2030 = future_best[future_best['Year']==2030].set_index('Country Code')
    worst_2030 = future_worst[future_worst['Year']==2030].set_index('Country Code')
    for cc in top5['Country Code']:
        row = {'Country Code': cc}
        for f in ['Trade_Diversification_Index','HHI_export','HHI_import','Current account balance (% of GDP)','trade_gdp_pct','disaster_severity']:
            if f in best_2030.columns and f in worst_2030.columns and cc in best_2030.index and cc in worst_2030.index:
                try:
                    diff = float(best_2030.loc[cc,f]) - float(worst_2030.loc[cc,f])
                    row[f'driver_{f}'] = diff
                except Exception:
                    pass
        drivers.append(row)
drivers_df = pd.DataFrame(drivers)

# Save Q12
top5.to_csv('/content/Q12_top5_resilience_2030.csv', index=False)
drivers_df.to_csv('/content/Q12_top5_resilience_drivers.csv', index=False)

print("Saved Q12 resilience rankings and drivers.")
def pretty_targets(cols):
    return [TARGETS.get(c, c) for c in cols]

print("\nQ10: 2030 predictions under shocks")
print("Disaster 2026:")
print(q10a_df.head(10))
print("\nTrade War 2027 :")
print(q10b_df.head(10))

print("\nQ11: 2030 GDP & poverty")
print("Best Cas: ")
print(q11a_df.head(10))
print("\nWorst Case :")
print(q11b_df.head(10))

print("\nQ12: Top-5 resilience tier by 2030")
print(top5)
print("\nDrivers for those countries:")
print(drivers_df.head())