# -*- coding: utf-8 -*-
"""P1 容量×接口扫描：PatchMLP 三档容量（base/256/512）× 三接口（C/B2/B1）× 5 dev 流 × 3 种子。

目的：排除"接口缺口是学生容量伪影"的审稿质疑（R10 预注册，pre-run）。
数据源（只读）：results/teacher_cache/{fid}.npz（Chronos 缓存，含 xhat 预测列）。
产出：results/capacity_sweep/{arm}_{bb}_{fid}.json + report.md
用法: python run_capacity_sweep.py   （GPU；P0 完成后启动）
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_flow
from run_pilot import vus_pr, fidelity, L_WIN, DEVICE
from distill import train_case
from students import PatchMLPLite, PatchMLP256, PatchMLP512, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "capacity_sweep")
os.makedirs(OUT, exist_ok=True)

FLOWS = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D18_Exathlon_id1", "D12_EMG"]
BACKBONES = ["PatchMLPLite", "PatchMLP256", "PatchMLP512"]
BB_CLS = {"PatchMLPLite": PatchMLPLite, "PatchMLP256": PatchMLP256, "PatchMLP512": PatchMLP512}
ARMS = [("C", "score-distill"), ("B2", "forecast-distill"), ("B1", "self-supervised")]
SEEDS = [0, 1, 2]


def main():
    logp = print
    lines = ["# 容量×接口扫描（R10，pre-run 注册；Chronos 教师，5 dev 流 × 3 种子）", ""]
    for fid in FLOWS:
        z = np.load(os.path.join(CACHE, fid + ".npz"))
        xtr, xte, y = load_flow(fid)
        C = xtr.shape[1]
        lines.append("## %s (C=%d, train %s)" % (fid, C, list(xtr.shape)))
        lines.append("")
        lines.append("| 臂 | 骨干 | 参数量 | mean VUS-PR | median rho |")
        lines.append("|---|---|---|---|---|")
        for bb in BACKBONES:
            n_params = count_params(BB_CLS[bb](C, L=L_WIN))
            for case, arm_desc in ARMS:
                arm = ("C" if case == "C" else case) + "-" + bb.replace("PatchMLP", "PM").replace("PatchMLPLite", "PM")
                outp = os.path.join(OUT, "%s_%s_%s.json" % (case, bb, fid))
                if os.path.exists(outp):
                    res = json.load(open(outp, encoding="utf-8"))
                else:
                    res = {"arm": case, "backbone": bb, "flow": fid, "params": n_params,
                           "device": DEVICE, "runs": {}}
                    per, rhos = [], []
                    for sd in SEEDS:
                        t1 = time.time()
                        fn, info = train_case(bb, xtr, z["s_tr"], L=L_WIN, case=case,
                                              epochs=30, bs=256, lr=1e-3, seed=sd,
                                              device=DEVICE, stride=2,
                                              teacher_Xhat=(z["xhat_tr"] if case == "B2" else None))
                        q = fn(xte)
                        m = vus_pr(q, y)
                        f = fidelity(q, z["s_te"])
                        res["runs"]["s%d" % sd] = {"vus_pr": round(m, 4), **f, **info}
                        per.append(m)
                        rhos.append(f.get("spearman", float("nan")))
                        logp("[%s-%s:%s] s%d: %.4f (%.0fs)" % (case, bb, fid, sd, m, time.time() - t1))
                    res["mean"] = round(float(np.mean(per)), 4)
                    res["median_spearman"] = round(float(np.nanmedian(rhos)), 4)
                    json.dump(res, open(outp, "w"), indent=1)
                lines.append("| %s | %s | %s | %.4f | %s |" % (
                    arm_desc, bb, "{:,}".format(res["params"]), res["mean"],
                    res.get("median_spearman", "—")))
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
