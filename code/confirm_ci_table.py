# -*- coding: utf-8 -*-
"""X4：确认集各臂宏均 bootstrap 95% CI + 中位数（含 T0/PCA 与三个新经典基线）。"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
M6 = os.path.join(RES, "m6")
CB = os.path.join(RES, "classical_baselines")

rng = np.random.default_rng(0)
base = json.load(open(os.path.join(M6, "baselines.json"), encoding="utf-8"))
flows = sorted(base.keys())

rows = {}
rows["T0"] = np.array([base[f]["teacher"] for f in flows])
rows["PCA"] = np.array([base[f]["pca"] for f in flows])
for arm in ["A-PM", "C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]:
    rows[arm] = np.array([json.load(open(os.path.join(M6, "%s_%s.json" % (arm, f)),
                                          encoding="utf-8"))["mean"] for f in flows])
for name, key in [("robustz", "Robust-Z"), ("mares", "MA-res"), ("lofwin", "LOF-win")]:
    vals = []
    for f in flows:
        p = os.path.join(CB, "%s_%s.json" % (name, f))
        vals.append(json.load(open(p, encoding="utf-8"))["vus_pr"] if os.path.exists(p) else np.nan)
    rows[key] = np.array(vals)

order = ["T0", "Robust-Z", "MA-res", "LOF-win", "PCA", "LINEAR", "B1-PM", "B1-TCN", "B2-PM", "C-PM", "A-PM"]
print("| 臂 | 宏均 | 95% CI | 中位 |")
print("|---|---|---|---|")
out = []
for k in order:
    v = rows[k]
    if np.isnan(v).any():
        out.append("| %s | (待基线) | | |" % k)
        continue
    boots = [float(np.mean(rng.choice(v, len(v)))) for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    line = "| %s | %.4f | [%.3f, %.3f] | %.4f |" % (k, v.mean(), lo, hi, np.median(v))
    print(line)
    out.append(line)
open(os.path.join(M6, "confirm_ci_table.md"), "w", encoding="utf-8").write("\n".join(out))
