"""
HerbalSalt-AI | Step 7: Main figures formatted for European Journal of Nutrition
- No titles inside figures (journal rule), Arial lettering 8-10 pt, panel letters a, b, c
- Width 174 mm (single-column text area), 600 dpi TIFF (LZW) + vector PDF + PNG (for Overleaf)
Output: 05_manuscript\\EJN\\Fig1 ... Fig5  (numbered in order of first citation in the text)
Run:    python 07_ejn_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from lifelines import KaplanMeierFitter

ROOT = r"D:\My research papers\Harbal Salt"
TAB = os.path.join(ROOT, "04_results", "tables")
DATA = os.path.join(ROOT, "02_processed", "herbal_nhanes_analytic.csv")
MORT = os.path.join(ROOT, "01_raw_data", "mortality")
OUT = os.path.join(ROOT, "05_manuscript", "EJN")
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
W = 174 * MM
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight"})

HERBS = {"Curcumin_S": "Turmeric/curcumin", "Gingerol_S": "Ginger", "Quercetin_S": "Quercetin",
         "Boswellic_S": "Boswellia", "Piperine_S": "Black pepper/piperine", "Garlic": "Garlic",
         "Cinnamon": "Cinnamon", "GreenTea": "Green tea", "Ginkgo": "Ginkgo",
         "Ginseng": "Ginseng", "MilkThistle": "Milk thistle"}
H = list(HERBS)
ORDER = [HERBS[h] for h in H] + ["Any botanical"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


def letter(ax, s):
    ax.text(-0.04, 1.03, s, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")


def forest(df, est, panels, xlabel, name, width=W):
    n = len(panels)
    nr, nc = (2, 2) if n == 4 else (1, n)
    fig, axes = plt.subplots(nr, nc, figsize=(width, (165 if nr == 2 else 92) * MM), sharey=True, squeeze=False)
    yy = np.arange(len(ORDER))[::-1]
    for i, (key, lab) in enumerate(panels):
        ax = axes[i // nc][i % nc]
        s = df[df["Outcome"] == key].set_index("Botanical").reindex(ORDER)
        sig = s["q_FDR"].fillna(1) < 0.05
        ax.errorbar(s[est], yy, xerr=[s[est] - s["CI_low"], s["CI_high"] - s[est]],
                    fmt="none", ecolor="0.45", lw=0.8)
        ax.scatter(s.loc[~sig, est], yy[~sig.values], marker="o", s=14, color="0.2", zorder=3)
        ax.scatter(s.loc[sig, est], yy[sig.values], marker="D", s=20, color="#b03a2e", zorder=4)
        ax.axvline(1, ls="--", c="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 8)
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_title(lab, fontsize=7.5, pad=4)
        letter(ax, "abcd"[i])
        ax.set_yticks(yy)
        ax.set_yticklabels(ORDER)
    fig.tight_layout()
    save(fig, name)


# ----------------------------------------------------------- Fig 1 overview
df = pd.read_csv(DATA, low_memory=False)
df = df[(df["age"] >= 18) & (df["weight"] > 0)].copy()
df["any"] = (df[H].sum(axis=1) > 0).astype(int)
fig, axes = plt.subplots(1, 2, figsize=(W, 70 * MM), gridspec_kw={"width_ratios": [1.15, 1]})
cnt = pd.Series({HERBS[h]: int(df[h].sum()) for h in H}).sort_values()
axes[0].barh(cnt.index, cnt.values, color="0.35")
axes[0].set_xlabel("Number of users (unweighted)")
letter(axes[0], "a")
prev = df.groupby("cycle").apply(lambda g: 100 * np.average(g["any"], weights=g["weight"]))
axes[1].plot(range(len(prev)), prev.values, "o-", color="0.2", ms=4, lw=1)
axes[1].set_xticks(range(len(prev)))
axes[1].set_xticklabels([c.replace("-", "–") for c in prev.index], rotation=35, ha="right")
axes[1].set_ylabel("Weighted prevalence of any botanical use (%)")
letter(axes[1], "b")
fig.tight_layout()
save(fig, "Fig1")

# ------------------------------------------------- Fig 2 cross-sectional forest
t2 = pd.read_csv(os.path.join(TAB, "Table2_survey_logistic_ORs.csv"))
t2 = t2[(t2["Sample"] == "All adults") & t2["Model"].str.startswith("Model 2")]
outs = list(t2["Outcome"].unique())
nice = {"High hs-CRP (>3 mg/L)": "High hs-CRP (>3 mg/L)", "Depression (PHQ-9 >= 10)": "Depressive symptoms (PHQ-9 ≥10)",
        "Uncontrolled HbA1c (>= 7%) in diabetes": "Uncontrolled HbA1c (≥7%) in diabetes",
        "Uncontrolled BP (>= 140/90) in hypertension": "Uncontrolled BP (≥140/90 mmHg) in hypertension"}
forest(t2, "OR", [(o, nice.get(o, o)) for o in outs], "Adjusted OR (95% CI)", "Fig2")

# ------------------------------------------------------------- Fig 3 KM curve
frames = []
for c in ["2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016"]:
    p = os.path.join(MORT, f"NHANES_{c.replace('-', '_')}_MORT_2019_PUBLIC.dat")
    frames.append(pd.read_fwf(p, colspecs=[(0, 6), (14, 15), (15, 16), (45, 48)],
                              names=["SEQN", "eligstat", "mortstat", "permth_exm"], na_values=["."], dtype=str))
m = pd.concat(frames).apply(pd.to_numeric, errors="coerce")
d = df[df["cycle"].isin(["2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016"])].merge(m, on="SEQN")
d = d[(d["eligstat"] == 1) & d["permth_exm"].notna() & d["mortstat"].notna() & (d["permth_exm"] > 0)]
d["t"] = d["permth_exm"] / 12
d["w"] = d["weight"] / d["weight"].mean()
fig, ax = plt.subplots(figsize=(110 * MM, 80 * MM))
for v, lab, col, ls in [(0, "Non-users", "0.55", "--"), (1, "Botanical users", "0.1", "-")]:
    g = d[d["any"] == v]
    KaplanMeierFitter().fit(g["t"], g["mortstat"] == 1, weights=g["w"], label=lab).plot_survival_function(
        ax=ax, ci_show=True, color=col, linestyle=ls, lw=1.2)
ax.set_xlabel("Years since examination")
ax.set_ylabel("Survival probability (weighted)")
ax.legend(frameon=False, loc="lower left")
save(fig, "Fig3")

# ----------------------------------------------------- Fig 4 mortality forest
t9 = pd.read_csv(os.path.join(TAB, "Table9_Cox_mortality_HRs.csv"))
t9 = t9[(t9["Analysis"] == "Main") & t9["Model"].str.startswith("Model 2")]
forest(t9, "HR", [("All-cause mortality", "All-cause mortality"), ("Cardiovascular mortality", "Cardiovascular mortality"),
                  ("Cancer mortality", "Cancer mortality")], "Adjusted HR (95% CI)", "Fig4")

# ---------------------------------------------------- Fig 5 SHAP (mortality)
RACE = {"2": "Other Hispanic", "3": "Non-Hispanic White", "4": "Non-Hispanic Black", "5": "Other/multiracial"}
EDU = {"2": "Education: 9–11th grade", "3": "Education: high school/GED", "4": "Education: some college",
       "5": "Education: college graduate", "nan": "Education: missing"}
BASE = {"age": "Age", "sex_female": "Female sex", "income_pir": "Income-to-poverty ratio", "income_pir_miss": "Income missing",
        "BMXBMI": "BMI", "BMXBMI_miss": "BMI missing", "n_other_supp": "No. of other supplements",
        "diabetes_f": "Diabetes", "hypertension_f": "Hypertension", "n_herbs": "No. of botanicals",
        "smoker_f_0": "Never smoker", "smoker_f_1": "Ever smoker"}


def pretty(f):
    if f in HERBS: return HERBS[f]
    if f.endswith("_days30") and f[:-7] in HERBS: return HERBS[f[:-7]] + " (days/month)"
    if f in BASE: return BASE[f]
    if f.startswith("race_eth_"): return RACE.get(f[9:], f)
    if f.startswith("education_"): return EDU.get(f[10:], f)
    if f.startswith("cycle_"): return "Survey cycle " + f[6:].replace("-", "–")
    return f


bot = set(H) | {h + "_days30" for h in H} | {"n_herbs"}
s = pd.read_csv(os.path.join(TAB, "Table11_SHAP_mortality.csv")).sort_values("Mean_abs_SHAP", ascending=False).head(15)[::-1]
fig, ax = plt.subplots(figsize=(110 * MM, 95 * MM))
ax.barh([pretty(f) for f in s["Feature"]], s["Mean_abs_SHAP"],
        color=["#b03a2e" if f in bot else "0.6" for f in s["Feature"]],
        hatch=None, edgecolor=["k" if f in bot else "none" for f in s["Feature"]], linewidth=0.6)
for p, f in zip(ax.patches, s["Feature"]):
    if f in bot:
        p.set_hatch("////")
ax.set_xlabel("Mean |SHAP value| (log-hazard)")
save(fig, "Fig5")
print("\nAll figures saved to", OUT)
