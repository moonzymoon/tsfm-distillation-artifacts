# -*- coding: utf-8 -*-
"""从 results/m5* JSON 生成第四章 LaTeX 大表（数字程序生成防手抄错误）。"""
import json
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rows = {}
for f in glob.glob(os.path.join(RES, "m5", "D*.json")):
    d = json.load(open(f, encoding="utf-8"))
    rows[d["flow"]] = d
b = {}
for f in glob.glob(os.path.join(RES, "m5b", "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    b[(d["arm"], d["flow"])] = d["mean"]
ARMS = ["C-PM", "C-TCN", "A-PM", "A-TCN", "B1-TCN", "B1-PM", "B2-TCN", "B2-PM", "AB-PM", "AB-TCN", "LINEAR"]
flows = sorted(rows)


def get(arm, fl):
    if arm == "T0":
        return rows[fl]["teacher"]["vus_pr"]
    if arm == "PCA":
        return rows[fl]["pca"]["vus_pr"]
    if (arm, fl) in b:
        return b[(arm, fl)]
    return rows[fl][arm]["mean"]


out = ["% AUTO-GENERATED from results/m5*/ — do not edit numbers by hand",
       "\\begin{sidewaystable}",
       "\\centering",
       "\\caption{Development-set VUS-PR (mean over seeds) for eleven arms, the frozen teacher (T0), and the preregistered PCA baseline across the 20 dev flows. Best per flow in bold.}",
       "\\label{tab:dev}",
       "\\scriptsize",
       "\\setlength{\\tabcolsep}{2.4pt}",
       "\\begin{tabular}{l" + "c" * len(ARMS) + "cc}",
       "\\toprule",
       "Flow & " + " & ".join(ARMS) + " & T0 & PCA \\\\",
       "\\midrule"]
for fl in flows:
    vals = {a: get(a, fl) for a in ARMS}
    vals["T0"] = rows[fl]["teacher"]["vus_pr"]
    vals["PCA"] = rows[fl]["pca"]["vus_pr"]
    best = max(vals.values())
    cells = []
    for a in ARMS + ["T0", "PCA"]:
        v = vals[a]
        s = f"{v:.3f}"
        cells.append("\\textbf{" + s + "}" if abs(v - best) < 1e-9 else s)
    out.append(fl.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
out.append("\\midrule")
for stat, fn in [("Macro mean", np.mean), ("Median", np.median)]:
    agg = {a: float(fn([get(a, fl) for fl in flows])) for a in ARMS + ["T0", "PCA"]}
    out.append(stat + " & " + " & ".join(f"{agg[a]:.3f}" for a in ARMS + ["T0", "PCA"]) + " \\\\")
out += ["\\bottomrule", "\\end{tabular}", "\\end{sidewaystable}"]
dst = os.path.join(HERE, "..", "paper", "tables", "dev_table.tex")
open(dst, "w", encoding="utf-8").write("\n".join(out))
print(f"written {dst}, {len(flows)} flows, {len(ARMS)} arms")
