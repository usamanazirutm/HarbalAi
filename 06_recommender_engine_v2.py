"""
HerbalSalt-AI | Track 2, Step 1: Personalised recommender engine (v2: gated personalisation + data-support checks)
--------------------------------------------------------------------------
For every botanical ("salt") and outcome, fits a survey-weighted model with
botanical x patient-characteristic interactions (effect modifiers):
    age (per 10 y, centred at 50), female sex, diabetes, hypertension
From the coefficients and their robust covariance, a personalised effect
(HR or OR) with 95% CI can be computed for ANY patient profile.

Outputs (04_results\\recommender\\):
    recommender_model.json          -> loaded by the web app (next step)
    heterogeneity_tests.csv         -> does the effect differ between patients?
    example_patient_profiles.csv    -> personalised estimates for 8 example patients

To retrain on clinic data later: edit the CONFIG block only.
Run:  python 06_recommender_engine.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

# ============================== CONFIG ==================================
ROOT = r"D:\My research papers\Harbal Salt"
DATA = os.path.join(ROOT, "02_processed", "herbal_nhanes_analytic.csv")
MORT_DIR = os.path.join(ROOT, "01_raw_data", "mortality")
OUT = os.path.join(ROOT, "04_results", "recommender")
os.makedirs(OUT, exist_ok=True)

SALTS = {
    "Curcumin_S": "Turmeric/curcumin", "Gingerol_S": "Ginger", "Quercetin_S": "Quercetin",
    "Boswellic_S": "Boswellia", "Piperine_S": "Black pepper/piperine", "Garlic": "Garlic",
    "Cinnamon": "Cinnamon", "GreenTea": "Green tea", "Ginkgo": "Ginkgo",
    "Ginseng": "Ginseng", "MilkThistle": "Milk thistle",
}
# effect modifiers: name -> how to compute from the data frame
MODIFIERS = ["age10", "female", "diabetes", "hypertension"]
AGE_CENTRE = 50.0
MIN_USERS, MIN_EVENTS_IN_USERS = 100, 10        # below this -> "insufficient data"
HET_Q = 0.05                                    # personalise only if heterogeneity FDR q < this
MIN_BAND_EVENTS = 5                             # events among users in patient's age band for support
AGE_BANDS = [(18, 39), (40, 59), (60, 79), (80, 200)]
DATA_SOURCE = "NHANES 2007-2020 (observational, US adults)"
# ========================================================================

MORT_CYCLES = ["2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016"]


def prepare(df):
    df = df[(df["age"] >= 18) & (df["weight"] > 0)].copy()
    df["age10"] = (df["age"] - AGE_CENTRE) / 10.0
    df["female"] = df["sex_female"].astype(int)
    df["diabetes"] = (df["diabetes"] == 1).astype(int)
    df["hypertension"] = (df["hypertension"] == 1).astype(int)
    df["n_herbs"] = df[list(SALTS)].sum(axis=1)
    df["n_other_supp"] = (df["n_supplements"] - df["n_herbs"]).clip(lower=0)
    df["cluster"] = df["strata"].astype(int).astype(str) + "_" + df["psu"].astype(int).astype(str)
    for c in ["income_pir", "BMXBMI"]:
        df[c + "_miss"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(df[c].median())
    df["smoker_f"] = df["ever_smoker"].fillna(-1)
    cat = pd.get_dummies(df[["race_eth", "education", "smoker_f", "cycle"]].astype(str), drop_first=True, dtype=float)
    cat.columns = [c.replace(".0", "").replace(" ", "") for c in cat.columns]
    df = pd.concat([df, cat], axis=1)
    cov = ["age10", "female", "diabetes", "hypertension", "income_pir", "income_pir_miss",
           "BMXBMI", "BMXBMI_miss", "n_other_supp"] + list(cat.columns)
    return df, cov


def load_mortality(df):
    frames = []
    for c in MORT_CYCLES:
        p = os.path.join(MORT_DIR, f"NHANES_{c.replace('-', '_')}_MORT_2019_PUBLIC.dat")
        m = pd.read_fwf(p, colspecs=[(0, 6), (14, 15), (15, 16), (45, 48)],
                        names=["SEQN", "eligstat", "mortstat", "permth_exm"], na_values=["."], dtype=str)
        frames.append(m)
    m = pd.concat(frames).apply(pd.to_numeric, errors="coerce")
    d = df[df["cycle"].isin(MORT_CYCLES)].merge(m, on="SEQN")
    d = d[(d["eligstat"] == 1) & d["permth_exm"].notna() & d["mortstat"].notna()].copy()
    d["time_y"] = d["permth_exm"] / 12.0
    d = d[d["time_y"] > 0]
    d["event"] = (d["mortstat"] == 1).astype(int)
    d["weight"] = d["weight"] * 13.2 / 10.0
    return d


def interaction_terms(d, salt):
    names = [salt] + [f"{salt}:{m}" for m in MODIFIERS]
    for m in MODIFIERS:
        d[f"{salt}:{m}"] = d[salt] * d[m]
    return d, names


def fit_cox(d, salt, cov):
    d, tnames = interaction_terms(d.copy(), salt)
    cols = tnames + cov
    x = d[cols + ["time_y", "event", "cluster"]].copy()
    x = x.loc[:, (x.nunique() > 1) | x.columns.isin(["time_y", "event", "cluster"])]
    x["w"] = d["weight"] / d["weight"].mean()
    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(x, duration_col="time_y", event_col="event", weights_col="w", cluster_col="cluster", robust=True)
    beta = cph.params_.reindex(tnames).fillna(0).values
    V = cph.variance_matrix_.reindex(index=tnames, columns=tnames).fillna(0).values
    return beta, V, int(d[salt].sum()), int(d.loc[d[salt] == 1, "event"].sum())


def fit_logit(d, salt, cov, y):
    d, tnames = interaction_terms(d.copy(), salt)
    d = d.dropna(subset=[y])
    X = sm.add_constant(d[tnames + cov].astype(float), has_constant="add")
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
    w = d["weight"] / d["weight"].mean()
    m = sm.GLM(d[y], X, family=sm.families.Binomial(), var_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cluster"])[0]})
    beta = m.params.reindex(tnames).fillna(0).values
    V = m.cov_params().reindex(index=tnames, columns=tnames).fillna(0).values
    return beta, V, int(d[salt].sum()), int(d.loc[d[salt] == 1, y].sum())


def fit_main(o, s, cov):
    d = o["data"].copy()
    d["_none"] = 0
    if o["kind"] == "cox":
        x = d[[s] + cov + ["time_y", "event", "cluster"]].copy()
        x = x.loc[:, (x.nunique() > 1) | x.columns.isin(["time_y", "event", "cluster"])]
        x["w"] = d["weight"] / d["weight"].mean()
        cph = CoxPHFitter(penalizer=1e-4)
        cph.fit(x, duration_col="time_y", event_col="event", weights_col="w", cluster_col="cluster", robust=True)
        return float(cph.params_[s]), float(cph.variance_matrix_.loc[s, s])
    d = d.dropna(subset=["y"])
    X = sm.add_constant(d[[s] + cov].astype(float), has_constant="add")
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
    w = d["weight"] / d["weight"].mean()
    m = sm.GLM(d["y"], X, family=sm.families.Binomial(), var_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cluster"])[0]})
    return float(m.params[s]), float(m.cov_params().loc[s, s])


def support(o, s):
    d = o["data"]
    ev = "event" if o["kind"] == "cox" else "y"
    bands = []
    for lo, hi in AGE_BANDS:
        u = (d[s] == 1) & d["age"].between(lo, hi)
        bands.append({"lo": lo, "hi": hi, "users": int(u.sum()), "events": int(d.loc[u, ev].sum())})
    return bands


def personal_effect(beta, V, profile):
    c = np.array([1.0] + [profile[m] for m in MODIFIERS])
    est = c @ beta
    se = np.sqrt(max(c @ V @ c, 0))
    return np.exp(est), np.exp(est - 1.96 * se), np.exp(est + 1.96 * se)


def wald_interactions(beta, V):
    b, v = beta[1:], V[1:, 1:]
    try:
        chi2 = float(b @ np.linalg.pinv(v) @ b)
        return chi2, float(1 - stats.chi2.cdf(chi2, len(b)))
    except Exception:
        return np.nan, np.nan


# =============================================================== main
raw = pd.read_csv(DATA, low_memory=False)
df, COV = prepare(raw)
COV_ADJ = [c for c in COV if c not in MODIFIERS]           # modifiers enter as main effects separately

OUTCOMES = {}
mort = load_mortality(df)
OUTCOMES["mortality"] = {"label": "All-cause mortality", "measure": "HR", "data": mort, "kind": "cox"}
crp = df[df["crp_mgL"].notna() & (df["crp_mgL"] <= 10)].copy()
crp["y"] = (crp["crp_mgL"] > 3).astype(int)
OUTCOMES["inflammation"] = {"label": "High hs-CRP (>3 mg/L)", "measure": "OR", "data": crp, "kind": "logit"}

model = {"data_source": DATA_SOURCE, "modifiers": MODIFIERS, "age_centre": AGE_CENTRE,
         "min_users": MIN_USERS, "min_events": MIN_EVENTS_IN_USERS,
         "salts": SALTS, "outcomes": {}}
het_rows = []
for key, o in OUTCOMES.items():
    model["outcomes"][key] = {"label": o["label"], "measure": o["measure"], "salts": {}}
    for s in SALTS:
        try:
            if o["kind"] == "cox":
                beta, V, n, ev = fit_cox(o["data"], s, COV_ADJ + MODIFIERS)
            else:
                beta, V, n, ev = fit_logit(o["data"], s, COV_ADJ + MODIFIERS, "y")
        except Exception as e:
            print(f"  failed {key}/{s}: {e}")
            continue
        chi2, p = wald_interactions(beta, V)
        b0, v0 = fit_main(o, s, COV_ADJ + MODIFIERS)
        model["outcomes"][key]["salts"][s] = {"beta": beta.tolist(), "cov": V.tolist(),
                                              "beta_main": b0, "var_main": v0,
                                              "n_users": n, "events_in_users": ev, "het_p": p,
                                              "support": support(o, s)}
        het_rows.append([o["label"], SALTS[s], n, ev, chi2, p] + list(np.round(beta, 3)))
        print(f"{o['label']:<24} {SALTS[s]:<22} users={n:<5} events={ev:<5} heterogeneity p={p:.3f}")

het = pd.DataFrame(het_rows, columns=["Outcome", "Botanical", "Users", "Events_in_users", "Wald_chi2_4df",
                                      "Heterogeneity_p", "beta_main"] + [f"beta_x_{m}" for m in MODIFIERS])
for o, g in het.groupby("Outcome"):
    het.loc[g.index, "Heterogeneity_q_FDR"] = multipletests(g["Heterogeneity_p"].fillna(1), method="fdr_bh")[1]
het.to_csv(os.path.join(OUT, "heterogeneity_tests.csv"), index=False)
lab2key = {o["label"]: k for k, o in model["outcomes"].items()}
name2salt = {v: k for k, v in SALTS.items()}
for _, r in het.iterrows():
    m = model["outcomes"][lab2key[r["Outcome"]]]["salts"][name2salt[r["Botanical"]]]
    m["het_q"] = float(r["Heterogeneity_q_FDR"])
    m["mode"] = "personalised" if r["Heterogeneity_q_FDR"] < HET_Q else "average"
model["het_q_threshold"] = HET_Q
model["min_band_events"] = MIN_BAND_EVENTS
model["version"] = 2

with open(os.path.join(OUT, "recommender_model.json"), "w") as f:
    json.dump(model, f)

# Example patients
profiles = [
    ("35y woman, healthy", 35, 1, 0, 0), ("35y man, healthy", 35, 0, 0, 0),
    ("55y woman, hypertension", 55, 1, 0, 1), ("55y man, diabetes", 55, 0, 1, 0),
    ("65y man, diabetes + hypertension", 65, 0, 1, 1), ("65y woman, diabetes + hypertension", 65, 1, 1, 1),
    ("75y woman, hypertension", 75, 1, 0, 1), ("75y man, healthy", 75, 0, 0, 0),
]
rows = []
for name, age, fem, dia, htn in profiles:
    prof = {"age10": (age - AGE_CENTRE) / 10, "female": fem, "diabetes": dia, "hypertension": htn}
    for key, o in model["outcomes"].items():
        for s, m in o["salts"].items():
            if m["mode"] == "personalised":
                e, lo, hi = personal_effect(np.array(m["beta"]), np.array(m["cov"]), prof)
            else:
                se = np.sqrt(m["var_main"])
                e, lo, hi = np.exp(m["beta_main"]), np.exp(m["beta_main"] - 1.96 * se), np.exp(m["beta_main"] + 1.96 * se)
            band = next(b for b in m["support"] if b["lo"] <= age <= b["hi"])
            enough = m["n_users"] >= MIN_USERS and m["events_in_users"] >= MIN_EVENTS_IN_USERS
            status = ("insufficient data" if not enough else
                      "limited data for this profile" if band["events"] < MIN_BAND_EVENTS else
                      "associated benefit" if hi < 1 else "associated harm" if lo > 1 else "uncertain")
            rows.append([name, o["label"], SALTS[s], m["mode"], round(e, 2), round(lo, 2), round(hi, 2), status])
ex = pd.DataFrame(rows, columns=["Patient", "Outcome", "Botanical", "Mode", "Estimate", "CI_low", "CI_high", "Status"])
ex.to_csv(os.path.join(OUT, "example_patient_profiles.csv"), index=False)

print("\nPersonalised models:", [(k, s) for k, o in model["outcomes"].items() for s, m in o["salts"].items() if m["mode"] == "personalised"])
print("\nStatus counts across example patients:")
print(ex.groupby(["Outcome", "Status"]).size().to_string())
print("\nSaved to", OUT)
