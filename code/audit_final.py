# -*- coding: utf-8 -*-
"""终审修复·数据核实：五项争议数字用原始数据裁决。"""
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.join(HERE, "..", "results", "m6")
M5 = os.path.join(HERE, "..", "results", "m5")
CS = os.path.join(HERE, "..", "results", "capacity_sweep")

base = json.load(open(os.path.join(M6, "baselines.json"), encoding="utf-8"))
flows = sorted(base.keys())

print("=== 1) B2-PM vs B1-PM 逐流胜负（确认集 9 流）===")
d = []
for f in flows:
    b2 = json.load(open(os.path.join(M6, "B2-PM_%s.json" % f), encoding="utf-8"))["mean"]
    b1 = json.load(open(os.path.join(M6, "B1-PM_%s.json" % f), encoding="utf-8"))["mean"]
    d.append(b2 - b1)
    print("  %-20s B2 %.4f  B1 %.4f  diff %+.4f" % (f, b2, b1, b2 - b1))
d = np.array(d)
print("  wins/losses/ties:", int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()))
print("  mean diff %+.4f" % d.mean())

print()
print("=== 2) 真正的 matched-pairs rank-biserial（对 |diff| 秩）===")


def rb(x, y):
    dd = np.array(x) - np.array(y)
    r = rankdata(np.abs(dd))
    rp = r[dd > 0].sum()
    rn = r[dd < 0].sum()
    return (rp - rn) / r.sum()


pairs = [("B1-TCN", "C-PM"), ("LINEAR", "C-PM"), ("B2-PM", "B1-PM"), ("A-PM", "C-PM")]
for a, b in pairs:
    va = [json.load(open(os.path.join(M6, "%s_%s.json" % (a, f)), encoding="utf-8"))["mean"] for f in flows]
    vb = [json.load(open(os.path.join(M6, "%s_%s.json" % (b, f)), encoding="utf-8"))["mean"] for f in flows]
    print("  %-9s vs %-5s  rank-biserial %+.3f  (sign-dominance %+.3f)" % (
        a, b, rb(va, vb), (np.array(va) > np.array(vb)).sum() / 9 - (np.array(va) < np.array(vb)).sum() / 9))

print()
print("=== 3) teacher-positive 复算（规则：逐流 T0 >= B1-PM）===")
pos = [f for f in flows if base[f]["teacher"] >=
       json.load(open(os.path.join(M6, "B1-PM_%s.json" % f), encoding="utf-8"))["mean"]]
print("  五源:", pos)
for arm in ["C-PM", "B1-TCN", "B1-PM", "LINEAR"]:
    v = np.mean([json.load(open(os.path.join(M6, "%s_%s.json" % (arm, f)), encoding="utf-8"))["mean"] for f in pos])
    print("  pos 组 %-7s %.4f" % (arm, v))
cp = np.mean([json.load(open(os.path.join(M6, "C-PM_%s.json" % f), encoding="utf-8"))["mean"] for f in pos])
b1t = np.mean([json.load(open(os.path.join(M6, "B1-TCN_%s.json" % f), encoding="utf-8"))["mean"] for f in pos])
b1p = np.mean([json.load(open(os.path.join(M6, "B1-PM_%s.json" % f), encoding="utf-8"))["mean"] for f in pos])
print("  C-PM − max(B1-TCN,B1-PM) = %+.4f" % (cp - max(b1t, b1p)))

print()
print("=== 4) SWaT（dev 流）各方法数字 ===")
fid = "D1_SWaT_id1"
m5 = json.load(open(os.path.join(M5, "%s.json" % fid), encoding="utf-8"))
print("  PCA(dev):", m5.get("baselines", m5).get("PCA", "查 m5 baselines 结构"))
for k in m5:
    if isinstance(m5[k], dict) and "mean" in m5[k]:
        print("  %-10s %.4f" % (k, m5[k]["mean"]))
for name in ["robustz", "mares", "lofwin"]:
    p = os.path.join(HERE, "..", "results", "classical_baselines", "%s_%s.json" % (name, fid))
    if os.path.exists(p):
        print("  %-10s %.4f (R11)" % (name, json.load(open(p, encoding="utf-8"))["vus_pr"]))

print()
print("=== 5) 容量 1024 档逐流与宏均（裁决 0.385 vs 0.332）===")
CFLOWS = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D18_Exathlon_id1", "D12_EMG"]
for arm in ["C", "B1", "B2"]:
    vals = []
    for f in CFLOWS:
        p = os.path.join(CS, "%s_PatchMLP1024_%s.json" % (arm, f))
        v = json.load(open(p, encoding="utf-8"))["mean"]
        vals.append(v)
        print("  %s %-18s %.4f" % (arm, f, v))
    print("  %s 五流宏均 %.4f | 四流(除EMG) %.4f" % (arm, np.mean(vals), np.mean(vals[:4])))
