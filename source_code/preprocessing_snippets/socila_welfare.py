import pandas as pd, numpy as np

def proc_socwel(inp, outp):  
    df=pd.read_csv(inp)   # load file
    df.columns=df.columns.str.strip()
    df=df.replace("..",np.nan)

    idv=[c for c in ["Country Name","Country Code","Series Name","Series Code"] if c in df.columns]
    valv=[c for c in df.columns if c not in idv]

    # reshape wide yrs into long
    d=df.melt(id_vars=idv,value_vars=valv,var_name="Year",value_name="val")
    d["Year"]=d["Year"].astype(str).str.extract(r"(\d{4})").astype(int)
    d["val"]=pd.to_numeric(d["val"],errors="coerce")

    d=d.sort_values(["Country Name","Country Code","Series Code","Year"]).reset_index(drop=True)
    d["miss"]=False; d["how"]="orig"

    # rules for diff indicators (custom gap filling logic)
    def rulez(code):
        c=str(code)
        if c.startswith("SP.DYN.LE00"): return dict(limit=3,fill_edges=True)
        if c=="SP.URB.TOTL.IN.ZS": return dict(limit=2,fill_edges=True)
        if c=="SP.POP.GROW": return dict(limit=1,fill_edges=False)
        if c=="SL.UEM.TOTL.ZS": return dict(limit=1,fill_edges=False)
        if c=="SI.POV.GINI": return dict(limit=0,fill_edges=False)
        if c=="SI.POV.DDAY": return dict(limit=0,fill_edges=False)
        return dict(limit=1,fill_edges=False)

    # loop over each (country, series) and fix gaps
    out=[]
    for k,g in d.groupby(["Country Name","Country Code","Series Code","Series Name"],sort=False):
        r=rulez(g["Series Code"].iloc[0])
        s=g["val"].copy(); m=pd.Series("orig",index=g.index)

        if r["limit"]>0: 
            si=s.interpolate("linear",limit=r["limit"])
            m.loc[s.isna() & si.notna()]="interp"
        else: si=s

        if r["fill_edges"]:
            sf=si.ffill(); m.loc[si.isna() & sf.notna()]="ffill"
            sb=sf.bfill(); m.loc[sf.isna() & sb.notna()]="bfill"
            sf=sb
        else: sf=si

        g=g.copy()
        g["val"]=sf; g["miss"]=m.ne("orig"); g["how"]=m
        out.append(g)

    fin=pd.concat(out,axis=0).sort_values(["Country Name","Country Code","Series Code","Year"]).reset_index(drop=True)
    fin.to_csv(outp,index=False)
    return fin

if __name__=="__main__":
    res=proc_socwel("Social_and_welfare.csv","processed_social_and_welfare.csv")
    print(res.head(20))
