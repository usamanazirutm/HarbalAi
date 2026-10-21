"""
HerbalSalt-AI | Step 5: Build supplementary material + collect manuscript figures
Output: 05_manuscript\\Supplementary_Tables.xlsx
        05_manuscript\\Supplementary_Figures.pdf
        05_manuscript\\Fig*.png  (main-text figures, copied next to manuscript.tex)
Run:    pip install openpyxl
        python 05_supplementary.py
"""
import os
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT = r"D:\My research papers\Harbal Salt"
TAB = os.path.join(ROOT, "04_results", "tables")
FIG = os.path.join(ROOT, "04_results", "figures")
OUT = os.path.join(ROOT, "05_manuscript")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------- main figures
for f in ["Fig1_overview", "Fig2_forest_plot", "Fig7_mortality_forest", "Fig8_KM_survival", "Fig9_SHAP_mortality"]:
    for ext in (".png", ".pdf"):
        src = os.path.join(FIG, f + ext)
        if os.path.exists(src):
            shutil.copy(src, OUT)
print("Main figures copied to", OUT)

# ------------------------------------------------------ supplementary tables
keywords = pd.DataFrame([
    ["Turmeric/curcumin", "turmeric; curcum (curcumin, Curcuma longa)"],
    ["Ginger", "ginger; Zingiber; gingerol"],
    ["Quercetin", "quercetin"],
    ["Boswellia", "boswell (Boswellia serrata); frankincense"],
    ["Black pepper/piperine", "piperine; black pepper; BioPerine; Piper nigrum"],
    ["Garlic", "garlic; Allium sativum"],
    ["Cinnamon", "cinnamon; Cinnamomum"],
    ["Green tea", "green tea; Camellia sinensis; EGCG"],
    ["Ginkgo", "ginkgo"],
    ["Ginseng", "ginseng; Panax"],
    ["Milk thistle", "milk thistle; silymarin; Silybum"],
    ["Excluded (<150 users)", "berberine; ashwagandha; amla/Phyllanthus emblica; andrographis; bacopa; fenugreek"],
], columns=["Botanical", "Case-insensitive text patterns matched in product names, ingredient names and blend-component names"])


def r(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    return df


sheets = {
    "S1_botanical_definitions": keywords,
    "S2_crosssectional_ORs": r(pd.read_csv(os.path.join(TAB, "Table2_survey_logistic_ORs.csv"))),
    "S3_combinations": r(pd.read_csv(os.path.join(TAB, "Table3_combinations.csv"))),
    "S4_ML_performance": r(pd.read_csv(os.path.join(TAB, "Table5_ML_performance.csv"))),
    "S5_mortality_HRs": r(pd.read_csv(os.path.join(TAB, "Table9_Cox_mortality_HRs.csv"))),
    "S6_duration_response": r(pd.read_csv(os.path.join(TAB, "Table4_duration_response.csv"))),
    "S7_SHAP_crosssectional": r(pd.read_csv(os.path.join(TAB, "Table6_SHAP_importance.csv"))),
    "S8_SHAP_mortality": r(pd.read_csv(os.path.join(TAB, "Table11_SHAP_mortality.csv"))),
    "S9_mortality_cohort": pd.read_csv(os.path.join(TAB, "Table8_mortality_cohort_summary.csv")),
    "S10_survival_Cindex": r(pd.read_csv(os.path.join(TAB, "Table10_ML_survival_Cindex.csv"))),
}
xlsx = os.path.join(OUT, "Supplementary_Tables.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    for name, d in sheets.items():
        d.to_excel(w, sheet_name=name[:31], index=False)
        ws = w.sheets[name[:31]]
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
print("Saved", xlsx)

# ----------------------------------------------------- supplementary figures
supp = [
    ("Fig3_couse_heatmap", "Supplementary Fig. S1. Co-use of the eleven study botanicals. Colour shows the Jaccard index; "
                           "numbers show co-users (shown if at least 20)."),
    ("Fig4_ROC_curves", "Supplementary Fig. S2. Out-of-fold ROC curves from nested cross-validation for XGBoost with and "
                        "without botanical features and for logistic regression."),
    ("Fig6_duration_response", "Supplementary Fig. S3. Duration of botanical use (tertiles among users) and outcomes; "
                               "survey-weighted, fully adjusted odds ratios versus non-users."),
    ("Fig5_SHAP_importance", "Supplementary Fig. S4. Top 15 predictors of each cross-sectional outcome by mean absolute "
                             "SHAP value (XGBoost). Red bars indicate botanical features."),
    ("FigS1_calibration", "Supplementary Fig. S5. Calibration of out-of-fold predicted probabilities (XGBoost with "
                          "botanical features), with Brier scores."),
]
pdf = os.path.join(OUT, "Supplementary_Figures.pdf")
with PdfPages(pdf) as pp:
    for f, cap in supp:
        p = os.path.join(FIG, f + ".png")
        if not os.path.exists(p):
            print("  missing:", p)
            continue
        img = mpimg.imread(p)
        h, w_ = img.shape[:2]
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0.04, 0.16, 0.92, 0.80])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.05, 0.06, cap, wrap=True, fontsize=10, va="bottom")
        pp.savefig(fig, dpi=300)
        plt.close(fig)
print("Saved", pdf)
