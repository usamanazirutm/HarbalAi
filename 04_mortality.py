"""
HerbalSalt-AI | Step 4: Prospective mortality follow-up (NHANES 2007-2016 -> deaths to 31 Dec 2019)
Downloads NCHS public-use Linked Mortality Files, then runs:
  - survey-weighted Cox models (all-cause, cardiovascular, cancer death) per botanical
  - sensitivity: exclude deaths in first 2 years; supplement users only
  - weighted Kaplan-Meier curves
  - XGBoost survival model + C-index (with vs without botanicals) + SHAP
Run:  pip install lifelines
      python 04_mortality.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
ROOT = r"D:\My research papers\Harbal Salt"
RAW = os.path.join(ROOT, "01_raw_data", "mortality")
DATA = os.path.join(ROOT, "02_processed", "herbal_nhanes_analytic.csv")
TAB = os.path.join(ROOT, "04_results", "tables")
FIG = os.path.join(ROOT, "04_results", "figures")
for p in (RAW, TAB, FIG):
    os.makedirs(p, exist_ok=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight"})

HERBS = {
    "Curcumin_S": "Turmeric/curcumin", "Gingerol_S": "Ginger", "Quercetin_S": "Quercetin",
    "Boswellic_S": "Boswellia", "Piperine_S": "Black pepper/piperine", "Garlic": "Garlic",
    "Cinnamon": "Cinnamon", "GreenTea": "Green tea", "Ginkgo": "Ginkgo",
    "Ginseng": "Ginseng", "MilkThistle": "Milk thistle",
}
H = list(HERBS)
CYCLES = ["2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


# ============================================================ 1. download
BASES = ["https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/",
         "https://ftp.cdc.gov/pub/health_statistics/NCHs/datalinkage/linked_mortality/"]
frames = []
for c in CYCLES:
    fn = f"NHANES_{c.replace('-', '_')}_MORT_2019_PUBLIC.dat"
    path = os.path.join(RAW, fn)
    if not os.path.exists(path):
        for b in BASES:
            try:
                r = requests.get(b + fn, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.content) > 10000:
                    open(path, "wb").write(r.content)
                    print(f"[ok] {fn}")
                    break
            except requests.RequestException as e:
                print("  error:", e)
        else:
            raise SystemExit(f"Download failed: {fn}\nGet it manually from {BASES[0]}{fn} and save into {RAW}")
    m = pd.read_fwf(path, colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)],
                    names=["SEQN", "eligstat", "mortstat", "ucod", "mort_diab", "mort_htn", "permth_int", "permth_exm"],
                    na_values=["."], dtype=str)
    frames.append(m)
mort = pd.concat(frames, ignore_index=True)
for c in mort.columns:
    mort[c] = pd.to_numeric(mort[c], errors="coerce")
print(f"Mortality records: {len(mort):,}")

# ============================================================ 2. dataset
df = pd.read_csv(DATA, low_memory=False)
df = df[df["cycle"].isin(CYCLES) & (df["age"] >= 18) & (df["weight"] > 0)].copy()
df["weight"] = df["weight"] * 13.2 / 10.0           # re-scale pooled weight to 5 two-year cycles
df = df.merge(mort, on="SEQN", how="inner")
df = df[(df["eligstat"] == 1) & df["permth_exm"].notna() & df["mortstat"].notna()].copy()
df["time_y"] = df["permth_exm"] / 12.0
df = df[df["time_y"] > 0]
df["death_all"] = (df["mortstat"] == 1).astype(int)
df["death_cvd"] = ((df["mortstat"] == 1) & df["ucod"].isin([1, 5])).astype(int)
df["death_cancer"] = ((df["mortstat"] == 1) & (df["ucod"] == 2)).astype(int)

df["any_herb"] = (df[H].sum(axis=1) > 0).astype(int)
df["n_herbs"] = df[H].sum(axis=1)
df["n_other_supp"] = (df["n_supplements"] - df["n_herbs"]).clip(lower=0)
df["cluster"] = df["strata"].astype(int).astype(str) + "_" + df["psu"].astype(int).astype(str)
for c in ["income_pir", "BMXBMI"]:
    df[c + "_miss"] = df[c].isna().astype(int)
    df[c] = df[c].fillna(df[c].median())
df["smoker_f"] = df["ever_smoker"].fillna(-1)
for c in ["diabetes", "hypertension"]:
    df[c + "_f"] = (df[c] == 1).astype(int)
cat = pd.get_dummies(df[["race_eth", "education", "smoker_f", "cycle"]].astype(str), drop_first=True, dtype=float)
cat.columns = [c.replace(".0", "").replace(" ", "") for c in cat.columns]
df = pd.concat([df, cat], axis=1)
COV = ["age", "sex_female", "income_pir", "income_pir_miss", "BMXBMI", "BMXBMI_miss", "n_other_supp",
       "diabetes_f", "hypertension_f"] + list(cat.columns)
df["w_norm"] = df["weight"] / df["weight"].mean()

print(f"Analytic cohort: {len(df):,} adults, {df['death_all'].sum():,} deaths, "
      f"median follow-up {df['time_y'].median():.1f} y")
summary = pd.DataFrame({
    "Item": ["Participants", "Botanical users", "All-cause deaths", "CVD deaths", "Cancer deaths",
             "Median follow-up (years)", "Person-years"],
    "Value": [len(df), int(df["any_herb"].sum()), int(df["death_all"].sum()), int(df["death_cvd"].sum()),
              int(df["death_cancer"].sum()), round(df["time_y"].median(), 2), round(df["time_y"].sum())]})
summary.to_csv(os.path.join(TAB, "Table8_mortality_cohort_summary.csv"), index=False)


# ============================================================ 3. Cox models
def cox(d, event, xvars):
    cols = list(dict.fromkeys(xvars))
    x = d[cols + ["time_y", event, "w_norm", "cluster"]].copy()
    x = x.loc[:, (x.nunique() > 1) | x.columns.isin(["time_y", event, "w_norm", "cluster"])]
    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(x, duration_col="time_y", event_col=event, weights_col="w_norm",
            cluster_col="cluster", robust=True)
    return cph


EVENTS = {"death_all": "All-cause mortality", "death_cvd": "Cardiovascular mortality",
          "death_cancer": "Cancer mortality"}
rows = []
analyses = {
    "Main": df,
    "Excluding deaths in first 2 years": df[~((df["death_all"] == 1) & (df["time_y"] < 2))],
    "Supplement users only": df[df["n_supplements"] > 0],
}
for aname, d in analyses.items():
    for ev, elab in EVENTS.items():
        for h in H + ["any_herb"]:
            n_ev = int(d.loc[d[h] == 1, ev].sum())
            if n_ev < 5:
                continue
            for mname, xv in {"Model 1 (age, sex)": [h, "age", "sex_female"],
                              "Model 2 (fully adjusted)": [h] + COV}.items():
                try:
                    m = cox(d, ev, xv)
                    s = m.summary.loc[h]
                    rows.append([aname, elab, HERBS.get(h, "Any botanical"), mname, len(d), int(d[h].sum()), n_ev,
                                 s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], s["p"]])
                except Exception as e:
                    print("  Cox failed:", aname, ev, h, e)
        print(f"Cox done: {aname} / {elab}")
res = pd.DataFrame(rows, columns=["Analysis", "Outcome", "Botanical", "Model", "N", "N_users",
                                  "Deaths_in_users", "HR", "CI_low", "CI_high", "p"])
for (a, mo), g in res.groupby(["Analysis", "Model"]):
    res.loc[g.index, "q_FDR"] = multipletests(g["p"], method="fdr_bh")[1]
res.to_csv(os.path.join(TAB, "Table9_Cox_mortality_HRs.csv"), index=False)

# Forest plot (main, fully adjusted)
fp = res[(res["Analysis"] == "Main") & res["Model"].str.startswith("Model 2")]
order = [HERBS[h] for h in H] + ["Any botanical"]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharey=True)
for ax, (ev, elab) in zip(axes, EVENTS.items()):
    s = fp[fp["Outcome"] == elab].set_index("Botanical").reindex(order)
    yy = np.arange(len(order))[::-1]
    ax.errorbar(s["HR"], yy, xerr=[s["HR"] - s["CI_low"], s["CI_high"] - s["HR"]], fmt="none", ecolor="grey", lw=1)
    ax.scatter(s["HR"], yy, c=["#c0392b" if q < 0.05 else "#34495e" for q in s["q_FDR"].fillna(1)], s=22, zorder=3)
    ax.axvline(1, ls="--", c="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(elab)
    ax.set_xlabel("Adjusted HR (95% CI)")
    ax.set_yticks(yy)
    ax.set_yticklabels(order)
fig.suptitle("Botanical supplement use and mortality, NHANES 2007-2016 (follow-up to 2019; red = FDR q < 0.05)", fontsize=9)
save(fig, "Fig7_mortality_forest")

# Weighted Kaplan-Meier
fig, ax = plt.subplots(figsize=(4.5, 3.6))
for v, lab, col in [(0, "Non-users", "#7f8c8d"), (1, "Botanical users", "#16a085")]:
    g = df[df["any_herb"] == v]
    km = KaplanMeierFitter().fit(g["time_y"], g["death_all"], weights=g["w_norm"], label=lab)
    km.plot_survival_function(ax=ax, ci_show=True, color=col)
ax.set_xlabel("Years since examination")
ax.set_ylabel("Survival probability (weighted)")
ax.set_title("All-cause survival by botanical use")
save(fig, "Fig8_KM_survival")

# ============================================================ 4. ML survival
ML_HERB = H + [h + "_days30" for h in H] + ["n_herbs"]
for c in ML_HERB:
    df[c] = df[c].fillna(0)
y_lower = df["time_y"].values
y_upper = np.where(df["death_all"] == 1, df["time_y"], np.inf)
params = {"objective": "survival:cox", "eval_metric": "cox-nloglik", "tree_method": "hist",
          "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": SEED}
oof = {"full": np.zeros(len(df)), "base": np.zeros(len(df))}
kf = KFold(5, shuffle=True, random_state=SEED)
for tr, te in kf.split(df):
    for name, feats in [("full", COV + ML_HERB), ("base", COV)]:
        lab = np.where(df["death_all"].values[tr] == 1, y_lower[tr], -y_lower[tr])
        dtr = xgb.DMatrix(df.iloc[tr][feats].astype(float), label=lab)
        bst = xgb.train(params, dtr, num_boost_round=400)
        oof[name][te] = bst.predict(xgb.DMatrix(df.iloc[te][feats].astype(float)), output_margin=True)
T, E = df["time_y"].values, df["death_all"].values
ci = {k: concordance_index(T, -v, E) for k, v in oof.items()}
deltas = []
idx = np.arange(len(df))
for _ in range(500):
    b = rng.choice(idx, len(idx))
    deltas.append(concordance_index(T[b], -oof["full"][b], E[b]) - concordance_index(T[b], -oof["base"][b], E[b]))
deltas = np.array(deltas)
pd.DataFrame({
    "Model": ["XGBoost-Cox covariates + botanicals", "XGBoost-Cox covariates only", "Delta C-index"],
    "Value": [ci["full"], ci["base"], ci["full"] - ci["base"]],
    "CI_low": [np.nan, np.nan, np.percentile(deltas, 2.5)],
    "CI_high": [np.nan, np.nan, np.percentile(deltas, 97.5)],
}).to_csv(os.path.join(TAB, "Table10_ML_survival_Cindex.csv"), index=False)

# SHAP on full model
lab = np.where(E == 1, T, -T)
bst = xgb.train(params, xgb.DMatrix(df[COV + ML_HERB].astype(float), label=lab), num_boost_round=400)
contrib = bst.predict(xgb.DMatrix(df[COV + ML_HERB].astype(float)), pred_contribs=True)[:, :-1]
feats = COV + ML_HERB
imp = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
dirs = []
for h in H:
    u = df[h].values == 1
    dirs.append([HERBS[h], int(u.sum()), contrib[u, feats.index(h)].mean() if u.any() else np.nan])
pd.DataFrame({"Feature": imp.index, "Mean_abs_SHAP": imp.values}).to_csv(
    os.path.join(TAB, "Table11_SHAP_mortality.csv"), index=False)
pd.DataFrame(dirs, columns=["Botanical", "N_users", "Mean_SHAP_in_users (log-hazard; <0 = protective)"]).to_csv(
    os.path.join(TAB, "Table12_SHAP_mortality_direction.csv"), index=False)
top = imp.head(15)[::-1]
pretty = {**HERBS, "age": "Age", "sex_female": "Female sex", "income_pir": "Income-to-poverty ratio",
          "BMXBMI": "BMI", "n_other_supp": "No. of other supplements", "diabetes_f": "Diabetes",
          "hypertension_f": "Hypertension", "n_herbs": "No. of botanicals"}
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.barh([pretty.get(f, f) for f in top.index], top.values,
        color=["#c0392b" if f in ML_HERB else "#7f8c8d" for f in top.index])
ax.set_xlabel("Mean |SHAP| (log-hazard)")
ax.set_title(f"All-cause mortality predictors (C-index {ci['full']:.3f})")
save(fig, "Fig9_SHAP_mortality")

print("\nALL DONE. New tables 8-12 and figures 7-9 saved in 04_results.")
