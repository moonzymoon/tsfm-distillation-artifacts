# -*- coding: utf-8 -*-
"""C1 洗牌教师负对照（R6，exploratory）：A-PM × 2种子 × dev 20流，教师分数时间重排。
CPU 执行（GPU 让渡 TimesFM）。产出 results/c1_shuffled/report.md
用法: CUDA_VISIBLE_DEVICES="" python run_c1_shuffled.py
"""
import os
import sys
import json
import time
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_flow
from run_pilot import vus_pr, fidelity, L_WIN, DEVICE
from distill import train_case

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "c1_shuffled")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1]


def dev_flow_ids():
    from data import all_flow_ids
    ids = [i for i in all_flow_ids() if i.startswith("D")]
    order = ["D1_SWaT_id1", "D11_SVDB_id2", "D3_SMD_id4"]
    return order + [i for i in ids if i not in order]


def main():
    report = []
    for fid in dev_flow_ids():
        outp = os.path.join(OUT, f"A-PM-shuf_{fid}.json")
        if os.path.exists(outp):
            continue
        z = np.load(os.path.join(CACHE, f"{fid}.npz"))
        s_tr = z["s_tr"]
        xtr, xte, y = load_flow(fid)
        res = {"arm": "A-PM-shuffled", "flow": fid, "device": DEVICE, "runs": {}}
        per = []
        for sd in SEEDS:
            rng = np.random.default_rng(100 + sd)
            s_shuf = rng.permutation(s_tr)          # 时间重排：保边际分布，破时序
            t1 = time.time()
            fn, info = train_case("PatchMLPLite", xtr, s_shuf, L=L_WIN, case="A",
                                  epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE,
                                  stride=2)
            q = fn(xte)
            m = vus_pr(q, y)
            res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **info}
            per.append(m)
            print(f"[c1:{fid}] s{sd}: {m:.4f} ({time.time()-t1:.0f}s)")
        res["mean"] = round(float(np.mean(per)), 4)
        json.dump(res, open(outp, "w"), indent=1)
        report.append((fid, res["mean"]))
    # 与 M5a 真 A-PM 对比
    lines = ["# C1 洗牌教师负对照（A-PM × 2种子，exploratory R6）", "",
             "| 流 | A-PM(真教师分数) | A-PM(洗牌分数) | Δ(真-洗) |", "|---|---|---|---|"]
    for fid in dev_flow_ids():
        p_real = os.path.join(HERE, "..", "results", "m5", f"A-PM_{fid}.json")
        p_shuf = os.path.join(OUT, f"A-PM-shuf_{fid}.json")
        if os.path.exists(p_real) and os.path.exists(p_shuf):
            vr = json.load(open(p_real))["A-PM"]["mean"]
            vs = json.load(open(p_shuf))["mean"]
            lines.append(f"| {fid} | {vr:.4f} | {vs:.4f} | {vr-vs:+.4f} |")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
