"""
HerbalSalt-AI | Step 2: Build the analytic dataset + feasibility report
Input : D:\\My research papers\\Harbal Salt\\01_raw_data
Output: D:\\My research papers\\Harbal Salt\\02_processed\\herbal_nhanes_analytic.csv
        D:\\My research papers\\Harbal Salt\\04_results\\feasibility_report.txt
Run:   python 02_build_dataset.py
"""
import os
import re
import numpy as np
import pandas as pd

ROOT = r"D:\My research papers\Harbal Salt"
RAW = os.path.join(ROOT, "01_raw_data")
PROC = os.path.join(ROOT, "02_processed")
RES = os.path.join(ROOT, "04_results")
os.makedirs(PROC, exist_ok=True)
os.makedirs(RES, exist_ok=True)

CYCLE_YEARS = {"2007-2008": 2.0, "2009-2010": 2.0, "2011-2012": 2.0,
               "2013-2014": 2.0, "2015-2016": 2.0, "2017-2020": 3.2}
TOTAL_YEARS = sum(CYCLE_YEARS.values())

# The 10 study botanicals ("herbal salts") -> label keywords (lower case)
BOTANICALS = {
    "Curcumin_S":        ["turmeric", "curcum"],
    "Gingerol_S":        ["ginger", "zingiber", "gingerol"],
    "Quercetin_S":       ["quercetin"],
    "Berberine_S":       ["berberine", "berberis"],
    "Ashwagandha_S":     ["ashwagandha", "withania"],
    "Boswellic_S":       ["boswell", "frankincense"],
    "Piperine_S":        ["piperine", "black pepper", "bioperine", "piper nigrum"],
    "Gallic_S":          ["amla", "emblica", "indian gooseberry", "amalaki"],
    "Andrographolide_S": ["andrograph"],
    "Bacosides_S":       ["bacopa", "brahmi", "bacoside"],
}
# Backup botanicals (used only if the 10 above have too few users)
BACKUP = {
    "Garlic":     ["garlic", "allium sativum"],
    "Cinnamon":   ["cinnamon", "cinnamomum"],
    "GreenTea":   ["green tea", "camellia sinensis", "egcg"],
    "Ginkgo":     ["ginkgo"],
    "Ginseng":    ["ginseng", "panax"],
    "Fenugreek":  ["fenugreek", "trigonella"],
    "MilkThistle": ["milk thistle", "silymarin", "silybum"],
}
ALL_HERBS = {**BOTANICALS, **BACKUP}


# ---------------------------------------------------------------- helpers
def read_xpt(path):
    if not os.path.exists(path):
        return None
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    return df


SUFFIX = {"2007-2008": "_E", "2009-2010": "_F", "2011-2012": "_G",
          "2013-2014": "_H", "2015-2016": "_I"}


def load(cycle, base):
    name = f"P_{base}" if cycle == "2017-2020" else f"{base}{SUFFIX[cycle]}"
    return read_xpt(os.path.join(RAW, "nhanes", cycle, f"{name}.xpt"))


def clean_codes(s, bad=(7, 9, 77, 99, 777, 999, 7777, 9999)):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin(bad))


def yes(s):
    s = clean_codes(s)
    return np.where(s == 1, 1, np.where(s == 2, 0, np.nan))


# ------------------------------------------- 1. map supplement IDs to herbs
def norm_id(s):
    """Make IDs comparable across files (numeric 123.0 and text '123' -> '123')."""
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).str.strip()
    ok = num.notna()
    out[ok] = num[ok].astype("int64").astype(str)
    return out


def text_cols(df, exclude):
    return [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]


def flag_herbs(df, key):
    df = df.groupby(key)["TEXT"].apply(" ".join).reset_index()
    for herb, kws in ALL_HERBS.items():
        pat = "|".join(re.escape(k) for k in kws)
        df[herb] = df["TEXT"].str.contains(pat, regex=True).astype(int)
    return df.drop(columns="TEXT")


def build_supplement_map():
    """Returns {key_column: herb-flag table}. Keys: DSDPID (product ID) and/or DSDSUPID."""
    dspi = read_xpt(os.path.join(RAW, "dsd", "DSPI.xpt"))
    dsii = read_xpt(os.path.join(RAW, "dsd", "DSII.xpt"))
    dsbi = read_xpt(os.path.join(RAW, "dsd", "DSBI.xpt"))
    if dsii is None:
        raise SystemExit("DSII.xpt missing - run step 1 first.")
    print("DSPI columns:", list(dspi.columns) if dspi is not None else "MISSING")

    pieces = []  # (DSDPID, TEXT)
    # Ingredient names + product names (DSII)
    t = dsii[["DSDPID"]].copy()
    t["DSDPID"] = norm_id(t["DSDPID"])
    t["TEXT"] = dsii[text_cols(dsii, ["DSDPID", "DSDIID"])].astype(str).agg(" ".join, axis=1).str.lower()
    pieces.append(t)
    # Blend components (DSBI) -> linked to products through DSDIID in DSII
    if dsbi is not None and "DSDIID" in dsbi.columns:
        link = pd.DataFrame({"DSDIID": norm_id(dsii["DSDIID"]), "DSDPID": norm_id(dsii["DSDPID"])}).drop_duplicates()
        b = pd.DataFrame({"DSDIID": norm_id(dsbi["DSDIID"])})
        b["TEXT"] = dsbi[text_cols(dsbi, ["DSDIID", "DSDBID", "DSDBCID"])].astype(str).agg(" ".join, axis=1).str.lower()
        pieces.append(b.merge(link, on="DSDIID", how="inner")[["DSDPID", "TEXT"]])
    # Product names (DSPI)
    if dspi is not None and "DSDPID" in dspi.columns:
        p = pd.DataFrame({"DSDPID": norm_id(dspi["DSDPID"])})
        p["TEXT"] = dspi[text_cols(dspi, ["DSDPID", "DSDSUPID"])].astype(str).agg(" ".join, axis=1).str.lower()
        pieces.append(p)

    maps = {"DSDPID": flag_herbs(pd.concat(pieces, ignore_index=True), "DSDPID")}

    # Older supplement IDs (DSDSUPID) -> product ID crosswalk, if DSPI has both
    if dspi is not None and {"DSDPID", "DSDSUPID"} <= set(dspi.columns):
        cw = pd.DataFrame({"DSDSUPID": norm_id(dspi["DSDSUPID"]), "DSDPID": norm_id(dspi["DSDPID"])})
        m = cw.merge(maps["DSDPID"], on="DSDPID", how="inner").drop(columns="DSDPID")
        maps["DSDSUPID"] = m.groupby("DSDSUPID").max().reset_index()

    pm = maps["DSDPID"]
    print(f"Products in database: {len(pm):,}")
    for h in BOTANICALS:
        print(f"  products containing {h:<18}: {int(pm[h].sum()):,}")
    return maps


# ------------------------------------------- 2. per-cycle participant data
def build_cycle(cycle, supmap):
    demo = load(cycle, "DEMO")
    if demo is None:
        print(f"skip {cycle}: DEMO missing")
        return None
    wcol = "WTMECPRP" if cycle == "2017-2020" else "WTMEC2YR"
    d = pd.DataFrame({
        "SEQN": demo["SEQN"],
        "cycle": cycle,
        "age": demo["RIDAGEYR"],
        "sex_female": (demo["RIAGENDR"] == 2).astype(int),
        "race_eth": demo["RIDRETH1"],
        "education": clean_codes(demo.get("DMDEDUC2")),
        "income_pir": demo.get("INDFMPIR"),
        "psu": demo["SDMVPSU"],
        "strata": demo["SDMVSTRA"],
        "weight": demo[wcol] * CYCLE_YEARS[cycle] / TOTAL_YEARS,
    })

    def merge(df, cols):
        nonlocal d
        if df is not None:
            keep = ["SEQN"] + [c for c in cols if c in df.columns]
            d = d.merge(df[keep], on="SEQN", how="left")

    merge(load(cycle, "BMX"), ["BMXBMI", "BMXWAIST"])
    merge(load(cycle, "GHB"), ["LBXGH"])
    merge(load(cycle, "DIQ"), ["DIQ010", "DIQ050", "DIQ070"])
    merge(load(cycle, "BPQ"), ["BPQ020", "BPQ050A"])
    merge(load(cycle, "MCQ"), ["MCQ160A"])
    merge(load(cycle, "SMQ"), ["SMQ020", "SMQ040"])

    # Blood pressure (mean of available readings)
    if cycle == "2017-2020":
        bp = load(cycle, "BPXO")
        sy, di = ["BPXOSY1", "BPXOSY2", "BPXOSY3"], ["BPXODI1", "BPXODI2", "BPXODI3"]
    else:
        bp = load(cycle, "BPX")
        sy, di = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]
    if bp is not None:
        sy = [c for c in sy if c in bp.columns]
        di = [c for c in di if c in bp.columns]
        b = pd.DataFrame({"SEQN": bp["SEQN"],
                          "sbp": bp[sy].mean(axis=1),
                          "dbp": bp[di].replace(0, np.nan).mean(axis=1)})
        d = d.merge(b, on="SEQN", how="left")

    # CRP (mg/L): CRP files report mg/dL -> x10
    crp = load(cycle, "HSCRP")
    if crp is not None and "LBXHSCRP" in crp.columns:
        d = d.merge(crp[["SEQN", "LBXHSCRP"]].rename(columns={"LBXHSCRP": "crp_mgL"}), on="SEQN", how="left")
    else:
        crp = load(cycle, "CRP")
        if crp is not None and "LBXCRP" in crp.columns:
            c = crp[["SEQN", "LBXCRP"]].copy()
            c["crp_mgL"] = c.pop("LBXCRP") * 10
            d = d.merge(c, on="SEQN", how="left")

    # PHQ-9 depression score
    dpq = load(cycle, "DPQ")
    if dpq is not None:
        items = [f"DPQ0{i}0" for i in range(1, 10)]
        items = [c for c in items if c in dpq.columns]
        sc = dpq[items].apply(lambda s: s.where(s.between(0, 3)))
        p = pd.DataFrame({"SEQN": dpq["SEQN"], "phq9": sc.sum(axis=1, min_count=len(items))})
        d = d.merge(p, on="SEQN", how="left")

    # Supplements -> herb exposure
    ds = load(cycle, "DSQIDS")
    herbs = list(ALL_HERBS)
    if ds is None:
        print(f"  {cycle}: DSQIDS missing")
        for h in herbs:
            d[h] = 0
        d["n_supplements"] = 0
        return d
    flags = pd.DataFrame(0, index=ds.index, columns=herbs)
    matched = pd.Series(False, index=ds.index)
    for key, mp in supmap.items():
        if key in ds.columns:
            k = norm_id(ds[key])
            m = pd.DataFrame({key: k}).merge(mp, on=key, how="left")
            m.index = ds.index
            matched |= m[herbs[0]].notna()
            flags = np.maximum(flags, m[herbs].fillna(0).astype(int))
    ds = pd.concat([ds, flags], axis=1)
    id_col = "DSDPID" if "DSDPID" in ds.columns else ("DSDSUPID" if "DSDSUPID" in ds.columns else None)
    print(f"  {cycle}: {len(ds):,} supplement records, {matched.mean()*100:.1f}% matched to database "
          f"(ID columns present: {[c for c in ('DSDPID','DSDSUPID') if c in ds.columns]})")

    days = clean_codes(ds["DSD103"]) if "DSD103" in ds.columns else pd.Series(np.nan, index=ds.index)
    dur = clean_codes(ds["DSD090"], bad=(77777, 99999)) if "DSD090" in ds.columns else pd.Series(np.nan, index=ds.index)
    agg = {"n_supplements": ds.groupby("SEQN")[id_col].nunique() if id_col else ds.groupby("SEQN").size()}
    for h in herbs:
        msk = ds[h] == 1
        agg[h] = ds[msk].groupby("SEQN")[h].max()
        agg[h + "_days30"] = days[msk].groupby(ds.loc[msk, "SEQN"]).max()
        agg[h + "_duration_days"] = dur[msk].groupby(ds.loc[msk, "SEQN"]).max()
    a_ = pd.DataFrame(agg)
    a_.index.name = "SEQN"
    d = d.merge(a_.reset_index(), on="SEQN", how="left")
    d[herbs + ["n_supplements"]] = d[herbs + ["n_supplements"]].fillna(0).astype(int)
    return d


# ------------------------------------------------------------------- main
def main():
    supmap = build_supplement_map()
    parts = [build_cycle(c, supmap) for c in CYCLE_YEARS]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)

    # Clean / derive
    df["diabetes"] = yes(df.get("DIQ010"))
    df["diabetes_med"] = np.where((yes(df.get("DIQ050")) == 1) | (yes(df.get("DIQ070")) == 1), 1, 0)
    df["hypertension"] = yes(df.get("BPQ020"))
    df["bp_med"] = yes(df.get("BPQ050A"))
    df["arthritis"] = yes(df.get("MCQ160A"))
    df["ever_smoker"] = yes(df.get("SMQ020"))
    df["n_study_herbs"] = df[list(BOTANICALS)].sum(axis=1)
    df["any_study_herb"] = (df["n_study_herbs"] > 0).astype(int)
    df = df[df["age"] >= 18].copy()

    out_csv = os.path.join(PROC, "herbal_nhanes_analytic.csv")
    df.to_csv(out_csv, index=False)

    # ---------------- feasibility report
    groups = {
        "All adults": df,
        "Diabetes": df[df["diabetes"] == 1],
        "Hypertension": df[df["hypertension"] == 1],
        "Arthritis": df[df["arthritis"] == 1],
        "Depressive sx (PHQ9>=10)": df[df["phq9"] >= 10] if "phq9" in df else df.iloc[0:0],
    }
    lines = ["HerbalSalt-AI | NHANES feasibility report", "=" * 60,
             f"Adults (18+): {len(df):,}", f"Cycles: {', '.join(df['cycle'].unique())}", ""]
    for gname, g in groups.items():
        lines.append(f"--- {gname} (n = {len(g):,}) ---")
        for h in ALL_HERBS:
            tag = "" if h in BOTANICALS else "  [backup]"
            lines.append(f"  {h:<20}{int(g[h].sum()):>6}{tag}")
        lines.append(f"  {'ANY of 10 study herbs':<20}{int(g['any_study_herb'].sum()):>6}")
        lines.append(f"  {'2+ study herbs':<20}{int((g['n_study_herbs'] >= 2).sum()):>6}")
        lines.append("")
    lines.append("Outcome availability (non-missing):")
    for c in ["LBXGH", "sbp", "dbp", "crp_mgL", "phq9", "BMXBMI"]:
        if c in df:
            lines.append(f"  {c:<10}{df[c].notna().sum():>8,}")
    report = "\n".join(lines)
    with open(os.path.join(RES, "feasibility_report.txt"), "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()