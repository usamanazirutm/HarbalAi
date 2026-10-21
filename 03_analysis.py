"""
HerbalSalt-AI | Step 3: Full analysis
Input : 02_processed\\herbal_nhanes_analytic.csv
Output: 04_results\\tables\\*.csv   and   04_results\\figures\\*.png (600 dpi)
Run:    python 03_analysis.py            (full run, ~20-40 min)
        python 03_analysis.py --fast     (quick check, ~3 min)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

warnings.filterwarnings("ignore")
FAST = "--fast" in sys.argv
SEED = 42
rng = np.random.default_rng(SEED)

ROOT = r"D:\My research papers\Harbal Salt"
DATA = os.path.join(ROOT, "02_processed", "herbal_nhanes_analytic.csv")
TAB = os.path.join(ROOT, "04_results", "tables")
FIG = os.path.join(ROOT, "04_results", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

HERBS = {
    "Curcumin_S": "Turmeric/curcumin", "Gingerol_S": "Ginger", "Quercetin_S": "Quercetin",
    "Boswellic_S": "Boswellia", "Piperine_S": "Black pepper/piperine", "Garlic": "Garlic",
    "Cinnamon": "Cinnamon", "GreenTea": "Green tea", "Ginkgo": "Ginkgo",
    "Ginseng": "Ginseng", "MilkThistle": "Milk thistle",
}
H = list(HERBS)

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


# =============================================================== 1. data
df = pd.read_csv(DATA, low_memory=False)
df = df[(df["age"] >= 18) & (df["weight"] > 0)].copy()
df["any_herb"] = (df[H].sum(axis=1) > 0).astype(int)
df["n_herbs"] = df[H].sum(axis=1)
df["supp_user"] = (df["n_supplements"] > 0).astype(int)
df["n_other_supp"] = (df["n_supplements"] - df["n_herbs"]).clip(lower=0)
df["cluster"] = df["strata"].astype(int).astype(str) + "_" + df["psu"].astype(int).astype(str)

# Outcomes
crp_ok = df["crp_mgL"].notna() & (df["crp_mgL"] <= 10)          # exclude acute inflammation
df["y_high_crp"] = np.where(crp_ok, (df["crp_mgL"] > 3).astype(float), np.nan)
df["y_depression"] = np.where(df["phq9"].notna(), (df["phq9"] >= 10).astype(float), np.nan)
dm = (df["diabetes"] == 1) & df["LBXGH"].notna()
df["y_uncontrolled_hba1c"] = np.where(dm, (df["LBXGH"] >= 7.0).astype(float), np.nan)
ht = (df["hypertension"] == 1) & df["sbp"].notna() & df["dbp"].notna()
df["y_uncontrolled_bp"] = np.where(ht, ((df["sbp"] >= 140) | (df["dbp"] >= 90)).astype(float), np.nan)

OUTCOMES = {
    "y_high_crp": ("High hs-CRP (>3 mg/L)", []),
    "y_depression": ("Depression (PHQ-9 >= 10)", []),
    "y_uncontrolled_hba1c": ("Uncontrolled HbA1c (>= 7%) in diabetes", ["diabetes_med"]),
    "y_uncontrolled_bp": ("Uncontrolled BP (>= 140/90) in hypertension", ["bp_med_f"]),
}

# Covariates (with missing-indicator handling)
for c in ["income_pir", "BMXBMI"]:
    df[c + "_miss"] = df[c].isna().astype(int)
    df[c] = df[c].fillna(df[c].median())
df["smoker_f"] = df["ever_smoker"].fillna(-1)
df["bp_med_f"] = df["bp_med"].fillna(0)
df["diabetes_med"] = df["diabetes_med"].fillna(0)
cat = pd.get_dummies(df[["race_eth", "education", "smoker_f", "cycle"]].astype("str"),
                     drop_first=True, dtype=float)
cat.columns = [c.replace(".0", "").replace(" ", "") for c in cat.columns]
df = pd.concat([df, cat], axis=1)
COV = (["age", "sex_female", "income_pir", "income_pir_miss", "BMXBMI", "BMXBMI_miss",
        "n_other_supp"] + list(cat.columns))

# =============================================================== 2. Table 1
def wmean(x, w):
    m = x.notna()
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


def wsd(x, w):
    m = x.notna()
    mu = np.average(x[m], weights=w[m])
    return np.sqrt(np.average((x[m] - mu) ** 2, weights=w[m]))


rows = []
groups = {"Non-users": df[df["any_herb"] == 0], "Botanical users": df[df["any_herb"] == 1], "Total": df}
cont = {"Age, years": "age", "BMI, kg/m2": "BMXBMI", "Income-to-poverty ratio": "income_pir",
        "HbA1c, %": "LBXGH", "Systolic BP, mmHg": "sbp", "hs-CRP, mg/L": "crp_mgL", "PHQ-9 score": "phq9",
        "Number of supplements": "n_supplements"}
binv = {"Female": "sex_female", "Ever smoker": "ever_smoker", "Diabetes": "diabetes",
        "Hypertension": "hypertension", "Arthritis": "arthritis"}
rows.append(["n (unweighted)"] + [f"{len(g):,}" for g in groups.values()])
for lab, c in cont.items():
    rows.append([f"{lab}, mean (SD)"] + [f"{wmean(g[c], g['weight']):.1f} ({wsd(g[c], g['weight']):.1f})" for g in groups.values()])
for lab, c in binv.items():
    rows.append([f"{lab}, %"] + [f"{100*wmean((g[c]==1).astype(float).where(g[c].notna()), g['weight']):.1f}" for g in groups.values()])
for h, lab in HERBS.items():
    rows.append([f"Uses {lab}, n"] + [f"{int(g[h].sum()):,}" for g in groups.values()])
pd.DataFrame(rows, columns=["Characteristic"] + list(groups)).to_csv(os.path.join(TAB, "Table1_characteristics.csv"), index=False)
print("Table 1 done")


# =================================== 3. Survey-weighted logistic regression
def svy_logit(d, y, xvars):
    d = d.dropna(subset=[y] + xvars)
    X = sm.add_constant(d[xvars].astype(float), has_constant="add")
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    w = d["weight"] / d["weight"].mean()
    m = sm.GLM(d[y], X, family=sm.families.Binomial(), var_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cluster"])[0]})
    return m, len(d)


def herb_models(data, tag):
    res = []
    for y, (ylab, extra) in OUTCOMES.items():
        for h in H + ["any_herb"]:
            sub = data[data[y].notna()]
            n_exp = int(sub[h].sum())
            if n_exp < 20:
                continue
            for model, xv in {"Model 1 (age, sex)": [h, "age", "sex_female"],
                              "Model 2 (fully adjusted)": [h] + COV + extra}.items():
                try:
                    m, n = svy_logit(sub, y, xv)
                    b, se = m.params[h], m.bse[h]
                    res.append([tag, ylab, HERBS.get(h, "Any botanical"), model, n, n_exp,
                                np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[h]])
                except Exception as e:
                    print("  model failed:", y, h, e)
    r = pd.DataFrame(res, columns=["Sample", "Outcome", "Botanical", "Model", "N", "N_users",
                                   "OR", "CI_low", "CI_high", "p"])
    for mod in r["Model"].unique():
        k = r["Model"] == mod
        r.loc[k, "q_FDR"] = multipletests(r.loc[k, "p"], method="fdr_bh")[1]
    return r


main = herb_models(df, "All adults")
sens = herb_models(df[df["supp_user"] == 1], "Supplement users only (healthy-user control)")
pd.concat([main, sens]).to_csv(os.path.join(TAB, "Table2_survey_logistic_ORs.csv"), index=False)
print("Regression done")

# Forest plot (Model 2, all adults)
fp = main[main["Model"].str.startswith("Model 2")]
outs = list(fp["Outcome"].unique())
fig, axes = plt.subplots(1, len(outs), figsize=(3.6 * len(outs), 4.8), sharey=True)
axes = np.atleast_1d(axes)
order = [HERBS[h] for h in H] + ["Any botanical"]
for ax, o in zip(axes, outs):
    s = fp[fp["Outcome"] == o].set_index("Botanical").reindex(order)
    yy = np.arange(len(order))[::-1]
    col = ["#c0392b" if q < 0.05 else "#34495e" for q in s["q_FDR"].fillna(1)]
    ax.errorbar(s["OR"], yy, xerr=[s["OR"] - s["CI_low"], s["CI_high"] - s["OR"]],
                fmt="none", ecolor="grey", lw=1)
    ax.scatter(s["OR"], yy, c=col, s=22, zorder=3)
    ax.axvline(1, ls="--", c="k", lw=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_title(o.replace(" (", "\n("), fontsize=8.5)
    ax.set_xlabel("Adjusted OR (95% CI)")
    ax.set_yticks(yy)
    ax.set_yticklabels(order)
fig.suptitle("Botanical supplement use and clinical outcomes (survey-weighted, fully adjusted; red = FDR q < 0.05)", fontsize=9)
save(fig, "Fig2_forest_plot")

# ================================================== 4. Co-use network
co = df.loc[df["any_herb"] == 1, H]
pair = co.T.dot(co)
pv = pair.to_numpy(dtype=float)
jv = pv / (np.add.outer(np.diag(pv), np.diag(pv)) - pv)
np.fill_diagonal(jv, np.nan)
jac = pd.DataFrame(jv, index=H, columns=H)
fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(jac, cmap="viridis")
lbl = [HERBS[h] for h in H]
ax.set_xticks(range(len(H)))
ax.set_xticklabels(lbl, rotation=60, ha="right")
ax.set_yticks(range(len(H)))
ax.set_yticklabels(lbl)
for i in range(len(H)):
    for j in range(len(H)):
        if i != j and pair.iloc[i, j] >= 20:
            ax.text(j, i, int(pair.iloc[i, j]), ha="center", va="center", fontsize=6, color="w")
fig.colorbar(im, ax=ax, label="Jaccard co-use index")
ax.set_title("Botanical co-use (numbers = co-users, shown if >= 20)")
save(fig, "Fig3_couse_heatmap")

# Top combinations vs outcomes
pairs = [(H[i], H[j]) for i in range(len(H)) for j in range(i + 1, len(H)) if pair.iloc[i, j] >= 50]
crow = []
for a, b in pairs:
    df["_combo"] = ((df[a] == 1) & (df[b] == 1)).astype(int)
    for y, (ylab, extra) in OUTCOMES.items():
        sub = df[df[y].notna()]
        if sub["_combo"].sum() < 20:
            continue
        try:
            m, n = svy_logit(sub, y, ["_combo"] + COV + extra)
            bb, se = m.params["_combo"], m.bse["_combo"]
            crow.append([f"{HERBS[a]} + {HERBS[b]}", ylab, int(sub["_combo"].sum()),
                         np.exp(bb), np.exp(bb - 1.96 * se), np.exp(bb + 1.96 * se), m.pvalues["_combo"]])
        except Exception:
            pass
comb = pd.DataFrame(crow, columns=["Combination", "Outcome", "N_users", "OR", "CI_low", "CI_high", "p"])
if len(comb):
    comb["q_FDR"] = multipletests(comb["p"], method="fdr_bh")[1]
comb.to_csv(os.path.join(TAB, "Table3_combinations.csv"), index=False)
print("Combinations done")

# ============================================ 5. Duration of use (tertiles)
dur_rows = []
for y, (ylab, extra) in OUTCOMES.items():
    d = df[df[y].notna()].copy()
    durs = d[[h + "_duration_days" for h in H]].max(axis=1)
    users = d["any_herb"] == 1
    if durs[users].notna().sum() < 60:
        continue
    t = pd.qcut(durs[users], 3, labels=["T1 (shortest)", "T2", "T3 (longest)"], duplicates="drop")
    d["dur_cat"] = "Non-user"
    d.loc[t.index, "dur_cat"] = t.astype(str)
    d = d[d["dur_cat"] != "nan"]
    dd = pd.get_dummies(d["dur_cat"], dtype=float).drop(columns="Non-user")
    d = pd.concat([d, dd], axis=1)
    try:
        m, n = svy_logit(d, y, list(dd.columns) + COV + extra)
        for c in dd.columns:
            dur_rows.append([ylab, c, int(dd[c].sum()), np.exp(m.params[c]),
                             np.exp(m.params[c] - 1.96 * m.bse[c]), np.exp(m.params[c] + 1.96 * m.bse[c]), m.pvalues[c]])
    except Exception as e:
        print("  duration model failed:", y, e)
dur = pd.DataFrame(dur_rows, columns=["Outcome", "Duration tertile", "N", "OR", "CI_low", "CI_high", "p"])
dur.to_csv(os.path.join(TAB, "Table4_duration_response.csv"), index=False)
if len(dur):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, (o, s) in enumerate(dur.groupby("Outcome", sort=False)):
        x = np.arange(len(s)) + k * 0.12
        ax.errorbar(x, s["OR"], yerr=[s["OR"] - s["CI_low"], s["CI_high"] - s["OR"]], fmt="o-", capsize=3, label=o, ms=4)
    ax.axhline(1, ls="--", c="k", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["T1 (shortest)", "T2", "T3 (longest)"])
    ax.set_ylabel("Adjusted OR vs non-users")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Duration of botanical use and outcomes")
    save(fig, "Fig6_duration_response")
print("Duration done")

# ================================================= 6. Machine learning
ML_HERB = H + [h + "_days30" for h in H] + ["n_herbs"]
for c in ML_HERB:
    if c.endswith("_days30"):
        df[c] = df[c].fillna(0)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
param_space = {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 5, 6],
               "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
               "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10]}
n_iter = 3 if FAST else 12


def boot_ci(y, p, B=300 if FAST else 1000):
    s = []
    idx = np.arange(len(y))
    for _ in range(B):
        b = rng.choice(idx, len(idx))
        if y[b].min() != y[b].max():
            s.append(roc_auc_score(y[b], p[b]))
    return np.percentile(s, [2.5, 97.5])


def boot_delta(y, p1, p0, B=300 if FAST else 1000):
    s = []
    idx = np.arange(len(y))
    for _ in range(B):
        b = rng.choice(idx, len(idx))
        if y[b].min() != y[b].max():
            s.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]))
    s = np.array(s)
    return s.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5), 2 * min((s <= 0).mean(), (s >= 0).mean())


ml_rows, roc_store, cal_store, shap_store = [], {}, {}, {}
for y, (ylab, extra) in OUTCOMES.items():
    d = df[df[y].notna()].reset_index(drop=True)
    if FAST and len(d) > 8000:
        d = d.sample(8000, random_state=SEED).reset_index(drop=True)
    Y = d[y].astype(int).values
    feats_full = COV + extra + ML_HERB
    feats_base = COV + extra
    oof = {k: np.zeros(len(d)) for k in ["XGB_full", "XGB_base", "LR_full"]}
    for tr, te in outer.split(d, Y):
        for name, feats in [("XGB_full", feats_full), ("XGB_base", feats_base)]:
            spw = (Y[tr] == 0).sum() / max((Y[tr] == 1).sum(), 1)
            search = RandomizedSearchCV(
                XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=SEED,
                              n_jobs=-1, scale_pos_weight=spw),
                param_space, n_iter=n_iter, cv=3, scoring="roc_auc", random_state=SEED, n_jobs=1)
            search.fit(d.loc[tr, feats].astype(float), Y[tr])
            oof[name][te] = search.predict_proba(d.loc[te, feats].astype(float))[:, 1]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        lr.fit(d.loc[tr, feats_full].astype(float), Y[tr])
        oof["LR_full"][te] = lr.predict_proba(d.loc[te, feats_full].astype(float))[:, 1]

    for name, p in oof.items():
        auc = roc_auc_score(Y, p)
        lo, hi = boot_ci(Y, p)
        ml_rows.append([ylab, name, len(d), int(Y.sum()), auc, lo, hi, brier_score_loss(Y, p)])
    dm_, dlo, dhi, dp = boot_delta(Y, oof["XGB_full"], oof["XGB_base"])
    ml_rows.append([ylab, "Delta AUC (botanicals added)", len(d), int(Y.sum()), dm_, dlo, dhi, dp])
    roc_store[ylab] = (Y, oof)
    cal_store[ylab] = (Y, oof["XGB_full"])

    # Final model on all data -> SHAP (native XGBoost TreeSHAP)
    spw = (Y == 0).sum() / max((Y == 1).sum(), 1)
    search = RandomizedSearchCV(
        XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1, scale_pos_weight=spw),
        param_space, n_iter=n_iter, cv=3, scoring="roc_auc", random_state=SEED, n_jobs=1)
    Xf = d[feats_full].astype(float)
    search.fit(Xf, Y)
    import xgboost as xgb
    contrib = search.best_estimator_.get_booster().predict(xgb.DMatrix(Xf), pred_contribs=True)[:, :-1]
    shap_store[ylab] = (contrib, Xf)
    print(f"ML done: {ylab}")

ml = pd.DataFrame(ml_rows, columns=["Outcome", "Model", "N", "N_events", "AUC_or_delta", "CI_low", "CI_high", "Brier_or_p"])
ml.to_csv(os.path.join(TAB, "Table5_ML_performance.csv"), index=False)

# ROC figure
fig, axes = plt.subplots(1, len(roc_store), figsize=(3.2 * len(roc_store), 3.2))
for ax, (o, (Y, oof)) in zip(np.atleast_1d(axes), roc_store.items()):
    for name, lab in [("XGB_full", "XGBoost + botanicals"), ("XGB_base", "XGBoost covariates only"), ("LR_full", "Logistic + botanicals")]:
        f, t, _ = roc_curve(Y, oof[name])
        ax.plot(f, t, lw=1.2, label=f"{lab} ({roc_auc_score(Y, oof[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_title(o.replace(" (", "\n("), fontsize=8.5)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
save(fig, "Fig4_ROC_curves")

# Calibration figure
fig, ax = plt.subplots(figsize=(4, 4))
for o, (Y, p) in cal_store.items():
    pt, pp = calibration_curve(Y, p, n_bins=10, strategy="quantile")
    ax.plot(pp, pt, "o-", ms=3, lw=1, label=o)
ax.plot([0, 1], [0, 1], "k--", lw=0.7)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed proportion")
ax.legend(fontsize=6, frameon=False)
ax.set_title("Calibration (out-of-fold, XGBoost + botanicals)")
save(fig, "FigS1_calibration")

# SHAP figures + table
pretty = {**HERBS, **{h + "_days30": HERBS[h] + " (days/30)" for h in H}, "n_herbs": "No. of botanicals",
          "age": "Age", "sex_female": "Female sex", "income_pir": "Income-to-poverty ratio", "BMXBMI": "BMI",
          "n_other_supp": "No. of other supplements", "diabetes_med": "Diabetes medication", "bp_med_f": "BP medication"}
shap_rows = []
fig, axes = plt.subplots(1, len(shap_store), figsize=(4.2 * len(shap_store), 4.4))
for ax, (o, (cv, Xf)) in zip(np.atleast_1d(axes), shap_store.items()):
    imp = pd.Series(np.abs(cv).mean(0), index=Xf.columns).sort_values(ascending=False)
    for f, v in imp.items():
        shap_rows.append([o, pretty.get(f, f), v, f in ML_HERB])
    top = imp.head(15)[::-1]
    ax.barh([pretty.get(f, f) for f in top.index], top.values,
            color=["#c0392b" if f in ML_HERB else "#7f8c8d" for f in top.index])
    ax.set_title(o.replace(" (", "\n("), fontsize=8.5)
    ax.set_xlabel("Mean |SHAP|")
fig.suptitle("Top-15 predictors by SHAP (red = botanical features)", fontsize=9)
save(fig, "Fig5_SHAP_importance")
pd.DataFrame(shap_rows, columns=["Outcome", "Feature", "Mean_abs_SHAP", "Is_botanical"]).to_csv(
    os.path.join(TAB, "Table6_SHAP_importance.csv"), index=False)

# SHAP direction for botanicals (users vs non-users)
dir_rows = []
for o, (cv, Xf) in shap_store.items():
    for h in H:
        j = list(Xf.columns).index(h)
        u = Xf[h] == 1
        if u.sum() >= 20:
            dir_rows.append([o, HERBS[h], int(u.sum()), cv[u.values, j].mean()])
pd.DataFrame(dir_rows, columns=["Outcome", "Botanical", "N_users", "Mean_SHAP_in_users (log-odds; <0 = protective)"]).to_csv(
    os.path.join(TAB, "Table7_SHAP_direction_botanicals.csv"), index=False)

# ====================================================== 7. Fig 1 overview
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
cnt = pd.Series({HERBS[h]: int(df[h].sum()) for h in H}).sort_values()
axes[0].barh(cnt.index, cnt.values, color="#16a085")
axes[0].set_xlabel("Number of users (unweighted)")
axes[0].set_title("a  Botanical supplement users, NHANES 2007-2020")
yr = df.groupby("cycle").apply(lambda g: 100 * np.average(g["any_herb"], weights=g["weight"]))
axes[1].plot(yr.index, yr.values, "o-", c="#16a085")
axes[1].set_ylabel("Weighted prevalence (%)")
axes[1].set_title("b  Prevalence of any botanical use by cycle")
axes[1].tick_params(axis="x", rotation=30)
save(fig, "Fig1_overview")

print("\nALL DONE. Tables ->", TAB, "\nFigures ->", FIG)
