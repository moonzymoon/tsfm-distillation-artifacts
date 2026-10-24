# -*- coding: utf-8 -*-
"""分层宏均敏感性（P2）：teacher-positive/null 分组 + 中位数 + 剔死流口径。

数据源（只读）：results/m6/baselines.json（T0/PCA）+ results/m6/{arm}_{fid}.json。
产出：results/stratified_macro.md
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.join(HERE, "..", "results", "m6")
OUTP = os.path.join(HERE, "..", "results", "stratified_macro.md")

ARMS = ["C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]
KEYS = ["T0", "PCA", "C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]

base = json.load(open(os.path.join(M6, "baselines.json"), encoding="utf-8"))
flows = sorted(base.keys())
assert len(flows) == 9, "expected 9 confirmation flows, got %d" % len(flows)

rows = {}
for fl in flows:
    r = {"T0": base[fl]["teacher"], "PCA": base[fl]["pca"]}
    for a in ARMS:
        fp = os.path.join(M6, "%s_%s.json" % (a, fl))
        r[a] = json.load(open(fp, encoding="utf-8"))["mean"]
    rows[fl] = r

pos = [f for f in flows if rows[f]["T0"] >= rows[f]["B1-PM"]]
nul = [f for f in flows if rows[f]["T0"] < rows[f]["B1-PM"]]
dead = [f for f in flows if max(rows[f].values()) < 0.02]
live = [f for f in flows if f not in dead]


def stat(sel, key, which):
    v = [rows[f][key] for f in sel]
    return round(float(np.mean(v)), 4) if which == "mean" else round(float(np.median(v)), 4)


lines = ["# 分层宏均敏感性（确认集 9 流，只读重分析）", "",
         "- teacher-positive（T0 ≥ B1-PM）: %d 流 %s" % (len(pos), pos),
         "- teacher-null（T0 < B1-PM）: %d 流 %s" % (len(nul), nul),
         "- 死流（全方法 max<0.02）: %d 流 %s" % (len(dead), dead),
         "- 剔死流后: %d 流" % len(live), "",
         "| 组 | 口径 | " + " | ".join(KEYS) + " |",
         "|---|---|" + "---|" * len(KEYS)]
for name, sel in [("all", flows), ("pos", pos), ("null", nul), ("live", live)]:
    if not sel:
        continue
    for which in ("mean", "median"):
        vals = [stat(sel, k, which) for k in KEYS]
        lines.append("| %s | %s | " % (name, which) + " | ".join(str(v) for v in vals) + " |")

lines += ["", "排序稳定性判读：各分组内 C-PM 相对最强 teacher-free 臂（B1-TCN/B1-PM/B2-PM）"
              "的差值符号与全量口径一致即视为稳健。"]
txt = "\n".join(lines)
open(OUTP, "w", encoding="utf-8").write(txt)
print(txt)
