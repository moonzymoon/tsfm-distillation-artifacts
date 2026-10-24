# -*- coding: utf-8 -*-
"""M5b 辅助臂（5.3裁定3）：B1 无教师 / B2 预测蒸馏→残差 / LINEAR 线性下界 × 双学生(线性除外) × 3种子 × dev 20流。
关键读数：B1-PM vs C-PM（教师边际价值）；B2 vs 检测蒸馏（DistilTS边界）。
逐臂逐流幂等：results/m5b/{arm}_{flow}.json。用法: python run_m5b.py [--force]
"""
import os
import sys
import json
import time
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_flow
from pca_baseline import PCABaseline
from distill import train_case
from run_pilot import vus_pr, fidelity, L_WIN, DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "m5b")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
# (arm, case, backbone)
ARMS = [("B1-TCN", "B1", "TCNLite"), ("B1-PM", "B1", "PatchMLPLite"),
        ("B2-TCN", "B2", "TCNLite"), ("B2-PM", "B2", "PatchMLPLite"),
        ("AB-TCN", "AB", "TCNLite"), ("AB-PM", "AB", "PatchMLPLite"),
        ("LINEAR", "LINEAR", None)]


def dev_flow_ids():
    from data import all_flow_ids
    ids = [i for i in all_flow_ids() if i.startswith("D")]
    order = ["D1_SWaT_id1", "D11_SVDB_id2", "D3_SMD_id4"]
    return order + [i for i in ids if i not in order]


def run_flow_arm(fid, arm, case, backbone, force=False, log=print):
    outp = os.path.join(OUT, f"{arm}_{fid}.json")
    if os.path.exists(outp) and not force:
        return None
    npz = os.path.join(CACHE, f"{fid}.npz")
    if not os.path.exists(npz):
        log(f"[m5b] {fid} 缓存缺失，跳过")
        return None
    z = np.load(npz)
    xtr, xte, y = load_flow(fid)
    xhat_tr = z["xhat_tr"] if case in ("B2", "AB") else None
    res = {"arm": arm, "flow": fid, "device": DEVICE, "runs": {}}
    t0 = time.time()
    per = []
    for sd in SEEDS:
        t1 = time.time()
        if case == "AB":
            from distill import train_AB
            fn, info = train_AB(backbone, xtr, z["s_tr"], xhat_tr, L=L_WIN, lam=0.1,
                                epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE, stride=2)
        else:
            fn, info = train_case(backbone, xtr, z["s_tr"], L=L_WIN, case=case, epochs=30,
                                  bs=256, lr=1e-3, seed=sd, device=DEVICE, stride=2,
                                  teacher_Xhat=xhat_tr)
        q = fn(xte)
        m = vus_pr(q, y)
        f = fidelity(q, z["s_te"])
        res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **f, **info}
        per.append(m)
        log(f"[m5b:{arm}:{fid}] s{sd}: {m:.4f} ρ={f['spearman']} ({time.time()-t1:.0f}s)")
    res["mean"] = round(float(np.mean(per)), 4)
    res["std"] = round(float(np.std(per)), 4)
    res["wallclock_s"] = round(time.time() - t0, 1)
    json.dump(res, open(outp, "w"), indent=1)
    log(f"[m5b] {arm} {fid} 完成 mean={res['mean']} {res['wallclock_s']}s")
    return res


def main():
    force = "--force" in sys.argv
    todo = [(f, arm, case, bb)
            for f in dev_flow_ids()
            for (arm, case, bb) in ARMS
            if force or not os.path.exists(os.path.join(OUT, f"{arm}_{f}.json"))]
    logp = print
    n = len(todo)
    logp(f"[m5b] 待跑 {n} 项（臂×流）")
    for k, (f, arm, case, bb) in enumerate(todo):
        logp(f"[m5b] ({k+1}/{n}) {arm} × {f}")
        try:
            run_flow_arm(f, arm, case, bb, force=force, log=logp)
        except Exception as e:
            logp(f"[m5b] {arm} × {f} 失败（跳过继续）: {repr(e)[:140]}")
    # 汇总
    arms = sorted({a for a, _, _ in ARMS})
    lines = ["# M5b 辅助臂结果（dev 20流，3种子 mean）", "",
             "| 臂 | 宏平均 | 中位数 | 完成流数 |", "|---|---|---|---|"]
    for arm in arms:
        vals = []
        for f in dev_flow_ids():
            p = os.path.join(OUT, f"{arm}_{f}.json")
            if os.path.exists(p):
                vals.append(json.load(open(p))["mean"])
        if vals:
            lines.append(f"| {arm} | {np.mean(vals):.4f} | {np.median(vals):.4f} | {len(vals)}/20 |")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "m5b_report.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("[m5b] 报告 → results/m5b/m5b_report.md")


if __name__ == "__main__":
    main()
