# -*- coding: utf-8 -*-
"""P0 汇总与判定：TimesFM 第二教师 29 流 vs Chronos 侧 + 最强 teacher-free 臂。

判定标准（R10 预注册，任一不成立 → 停机报告用户）：
  J1 宏均：C-PM(TF) 相对逐流最强 teacher-free 臂宏均无优势（差 ≥ -0.01 VUS-PR）
  J2 保真：C-PM(TF) 对 TimesFM 教师的 median Spearman ρ < 0.35（与 Chronos 侧同量级）
数据源（只读）：results/timesfm_subset/, results/m6/, results/m5/, results/m5b/,
results/m6/baselines.json, results/teacher_cache_timesfm/
产出：results/timesfm_subset/p0_summary.md
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TS = os.path.join(RES, "timesfm_subset")
TFC = os.path.join(RES, "teacher_cache_timesfm")
M6 = os.path.join(RES, "m6")
M5 = os.path.join(RES, "m5")
M5B = os.path.join(RES, "m5b")
OUTP = os.path.join(TS, "p0_summary.md")


def jload(p):
    return json.load(open(p, encoding="utf-8"))


tf_files = sorted(f for f in glob.glob(os.path.join(TS, "C-PM-TF_*.json")) if ".SKIPPED." not in f)
skipped = sorted(glob.glob(os.path.join(TS, "C-PM-TF_*.SKIPPED.json")))
if len(tf_files) < 9:
    raise SystemExit("only %d TimesFM result files - pool not done" % len(tf_files))

base = jload(os.path.join(M6, "baselines.json"))
M6_ARMS = ["B1-TCN", "B1-PM", "B2-PM", "LINEAR"]
M5B_ARMS = ["B1-TCN", "B1-PM", "B2-TCN", "B2-PM", "AB-TCN", "AB-PM", "LINEAR"]

rows = []
rhos = []
wins_tf_vs_free = 0
for fp in tf_files:
    d = jload(fp)
    fid = d["flow"]
    tf_mean = d["mean"]
    for r in d["runs"].values():
        rho = r.get("spearman")
        if rho is not None and np.isfinite(rho):
            rhos.append(rho)
    # Chronos C-PM: m6 for C flows, m5 for D flows
    if fid.startswith("C"):
        ch = jload(os.path.join(M6, "C-PM_%s.json" % fid))["mean"]
        free = {}
        for a in M6_ARMS:
            p2 = os.path.join(M6, "%s_%s.json" % (a, fid))
            if os.path.exists(p2):
                free[a] = jload(p2)["mean"]
        pca = base.get(fid, {}).get("pca")
        if pca is not None:
            free["PCA"] = pca
    else:
        p_ch = os.path.join(M5, "%s.json" % fid)
        ch = jload(p_ch)["C-PM"]["mean"] if os.path.exists(p_ch) else None
        free = {}
        for a in M5B_ARMS:
            p2 = os.path.join(M5B, "%s_%s.json" % (a, fid))
            if os.path.exists(p2):
                free[a] = jload(p2)["mean"]
    best_free = max(free.values()) if free else None
    best_free_name = (max(free, key=free.get) if free else "—")
    if best_free is not None and tf_mean >= best_free:
        wins_tf_vs_free += 1
    # TimesFM T0 (teacher zero-shot on this flow)
    tz = os.path.join(TFC, fid + ".npz")
    t0v = None
    if os.path.exists(tz):
        from data import load_flow
        from run_pilot import vus_pr
        s_te = np.load(tz)["s_te"]
        _, _, y = load_flow(fid)
        t0v = round(float(vus_pr(s_te, y)), 4)
    rows.append({"flow": fid, "tf": tf_mean, "chronos": ch, "t0": t0v,
                 "best_free": best_free, "best_free_name": best_free_name})

med_rho = round(float(np.median(rhos)), 4)
mac_tf = round(float(np.mean([r["tf"] for r in rows])), 4)
mac_ch = round(float(np.mean([r["chronos"] for r in rows if r["chronos"] is not None])), 4)
mac_free = round(float(np.mean([r["best_free"] for r in rows if r["best_free"] is not None])), 4)
mac_t0 = round(float(np.mean([r["t0"] for r in rows if r["t0"] is not None])), 4)
gap = round(mac_tf - mac_free, 4)

lines = ["# P0 汇总：TimesFM 第二教师（C-PM 配置，5 种子）", "",
         "覆盖 %d 流；跳过 %d 流（结构短于 ctx=1024，见 SKIPPED 标记）" % (len(tf_files), len(skipped)),
         "宏均（覆盖流）：C-PM(TF)=%.4f | C-PM(Chronos)=%.4f | 最强 teacher-free=%.4f | TimesFM T0=%.4f"
         % (mac_tf, mac_ch, mac_free, mac_t0),
         "median Spearman rho（C-PM-TF vs TimesFM 教师）= %.4f（%d runs）" % (med_rho, len(rhos)),
         "C-PM(TF) 逐流胜过最强 teacher-free 臂的流数：%d / %d" % (wins_tf_vs_free, len(rows)), "",
         "| 流 | C-PM(TF) | C-PM(Chronos) | TimesFM T0 | 最强free | 来自 |",
         "|---|---|---|---|---|---|"]
for r in rows:
    lines.append("| %s | %.4f | %s | %s | %s | %s |" % (
        r["flow"], r["tf"],
        "%.4f" % r["chronos"] if r["chronos"] is not None else "—",
        "%.4f" % r["t0"] if r["t0"] is not None else "—",
        "%.4f" % r["best_free"] if r["best_free"] is not None else "—",
        r["best_free_name"]))

j1 = gap >= -0.01
j2 = med_rho < 0.35
lines += ["", "## R10 判定", "",
          "- J1（宏均无优势）: gap = %+.4f → %s" % (gap, "成立" if j1 else "**不成立**"),
          "- J2（保真同量级）: median rho = %.4f → %s" % (med_rho, "成立" if j2 else "**不成立**"),
          "- 判定：%s" % ("模式复制，P0 通过" if (j1 and j2) else "**停机线触发——报告用户裁决**")]
open(OUTP, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines[-8:]))
