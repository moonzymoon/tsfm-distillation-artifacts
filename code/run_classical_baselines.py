# -*- coding: utf-8 -*-
"""R11 经典基线补跑：窗口特征 LOF + Robust-Z + MA-residual，29 流测试段，零调参。

口径：与 LINEAR 同语义（测试段逐点/逐窗打分 → VUS-PR）。
- Robust-Z：每通道 median/MAD（训练段估计），|x-med|/MAD，跨通道 max。逐点。
- MA-residual：每通道滑动均值 L=128 残差绝对值，跨通道 max。逐点（因果：只用过去窗口）。
- LOF：stride=16 窗口特征（每通道 [mean,std,last]，L=128），LocalOutlierFactor(novelty=True)
  在测试段窗口上训练+打分（无教师、无标签；与"部署时无参考正常库"语义一致披露）。
产出：results/classical_baselines/{name}_{fid}.json + report.md
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_flow
from run_pilot import vus_pr, L_WIN

OUT = os.path.join(HERE, "..", "results", "classical_baselines")
os.makedirs(OUT, exist_ok=True)
L = L_WIN
STRIDE_LOF = 16


def robust_z(xtr, xte):
    med = np.median(xtr, axis=0)
    mad = np.median(np.abs(xtr - med), axis=0)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    return np.max(np.abs(xte - med) / mad, axis=1).astype(np.float32)


def ma_residual(xtr, xte):
    T, C = xte.shape
    s = np.zeros(T, dtype=np.float32)
    # 因果滑动均值：窗口 [t-L+1, t]，前 L-1 点用可见部分
    for c in range(C):
        cs = np.cumsum(np.insert(xte[:, c].astype(np.float64), 0, 0.0))
        n = np.arange(1, T + 1)
        wlen = np.minimum(n, L)
        ma = (cs[n] - cs[np.maximum(n - wlen, 0)]) / wlen
        s = np.maximum(s, np.abs(xte[:, c] - ma).astype(np.float32))
    return s


def lof_window(xtr, xte):
    from sklearn.neighbors import LocalOutlierFactor
    T, C = xte.shape
    idx = np.arange(0, T - L + 1, STRIDE_LOF)
    feats = []
    for i in idx:
        w = xte[i:i + L]
        f = np.empty(3 * C, dtype=np.float32)
        f[0::3] = w.mean(axis=0)
        f[1::3] = w.std(axis=0)
        f[2::3] = w[-1]
        feats.append(f)
    F = np.array(feats)
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    F = (F - mu) / sd
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean")
    lof.fit(F)
    raw = -lof.score_samples(F)          # 越大越异常
    # 窗口分映射回时点：窗口 [i, i+L) 覆盖的每个点取 max（与窗口→分数语义一致）
    s = np.zeros(T, dtype=np.float32)
    for k, i in enumerate(idx):
        s[i:i + L] = np.maximum(s[i:i + L], raw[k])
    return s


def main():
    from data import FLOWS, NAB_FLOWS, TSBUAD_FLOWS
    allf = sorted(list(FLOWS) + list(NAB_FLOWS) + list(TSBUAD_FLOWS))
    methods = [("robustz", robust_z), ("mares", ma_residual), ("lofwin", lof_window)]
    report = ["# R11 经典基线（零调参，29 流测试段）", "",
              "| 流 | robust-z | ma-residual | lof-window |", "|---|---|---|---|"]
    mac = {m: [] for m, _ in methods}
    for fid in allf:
        row = [fid]
        xtr, xte, y = load_flow(fid)
        for name, fn in methods:
            outp = os.path.join(OUT, "%s_%s.json" % (name, fid))
            if os.path.exists(outp):
                m = json.load(open(outp, encoding="utf-8"))["vus_pr"]
            else:
                t0 = time.time()
                s = fn(xtr, xte)
                m = float(vus_pr(s, y))
                json.dump({"method": name, "flow": fid, "vus_pr": round(m, 4),
                           "wallclock_s": round(time.time() - t0, 1)},
                          open(outp, "w"), indent=1)
            mac[name].append(m)
            row.append("%.4f" % m)
        report.append("| " + " | ".join(row) + " |")
        print(" | ".join(row), flush=True)
    report += ["", "| 宏均 | " + " | ".join("%.4f" % float(np.mean(mac[n])) for n, _ in methods) + " |",
               "| 中位 | " + " | ".join("%.4f" % float(np.median(mac[n])) for n, _ in methods) + " |"]
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(report))
    print("\n".join(report[-2:]))


if __name__ == "__main__":
    main()
