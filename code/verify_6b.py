# -*- coding: utf-8 -*-
"""6-B 意见事实核验 + TOST 等价性补充计算。"""
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


def g(a, fl):
    if a == "T0":
        return rows[fl]["teacher"]["vus_pr"]
    if a == "PCA":
        return rows[fl]["pca"]["vus_pr"]
    if (a, fl) in b:
        return b[(a, fl)]
    return rows[fl][a]["mean"]


flows = sorted(rows)
print("① 指控核验（宏平均）：")
for a in ["A-TCN", "AB-TCN", "LINEAR", "B2-TCN", "T0"]:
    print(f"   {a}: {round(float(np.mean([g(a, f) for f in flows])), 3)}")
print("③ D10_GHL 各臂：", {a: g(a, "D10_GHL_id2") for a in ["T0", "PCA", "C-PM", "B1-TCN", "LINEAR"]})

rng = np.random.default_rng(0)
m6 = {}
for arm in ["C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]:
    m6[arm] = {}
    for f in glob.glob(os.path.join(RES, "m6", f"{arm}_C*.json")):
        fid = os.path.basename(f).replace(arm + "_", "").replace(".json", "")
        m6[arm][fid] = json.load(open(f, encoding="utf-8"))["mean"]
print("⑤ TOST（预注册非劣界 0.05，bootstrap CI）：")
for a, bb in [("B1-TCN", "C-PM"), ("LINEAR", "C-PM"), ("B2-PM", "B1-PM")]:
    d = np.array([m6[a][f] - m6[bb][f] for f in m6[a] if f in m6[bb]])
    bs = sorted(float(np.mean(rng.choice(d, len(d)))) for _ in range(5000))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ok = "可宣称等价" if lo >= -0.05 and hi <= 0.05 else "不可宣称等价"
    print(f"   {a} vs {bb}: CI [{lo:+.3f},{hi:+.3f}] → {ok}")
