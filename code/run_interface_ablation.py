# -*- coding: utf-8 -*-
"""W2 接口消融（R7, exploratory）：
消融1 等上下文：L=512（=教师上下文）下 {A-PM, B1-PM} × 5代表流 × 3种子，
     与 L=128 标准结果（M5a/M5b 已有）对比 → 上下文不对称是否改变接口结论。
复用 distill.train_case（参数化 L），无新训练逻辑。GPU。
产出 results/interface_ablation/report.md
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
OUT = os.path.join(HERE, "..", "results", "interface_ablation")
os.makedirs(OUT, exist_ok=True)
FLOWS = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D16_NAB_occ6005", "D18_Exathlon_id1"]
SEEDS = [0, 1, 2]
L_ABL = 512


def run_one(tag, fid, case, backbone, L, force=False, log=print):
    outp = os.path.join(OUT, f"{tag}_{fid}.json")
    if os.path.exists(outp) and not force:
        return None
    z = np.load(os.path.join(CACHE, f"{fid}.npz"))
    xtr, xte, y = load_flow(fid)
    xhat = z["xhat_tr"] if case == "B2" else None
    res = {"tag": tag, "flow": fid, "L": L, "case": case, "backbone": backbone,
           "device": DEVICE, "runs": {}}
    per = []
    for sd in [0, 1, 2]:
        t1 = time.time()
        fn, info = train_case(backbone, xtr, z["s_tr"], L=L, case=case, epochs=30,
                              bs=128, lr=1e-3, seed=sd, device=DEVICE, stride=2,
                              teacher_Xhat=xhat)
        q = fn(xte)
        m = vus_pr(q, y)
        res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **info}
        per.append(m)
        log(f"[abl:{tag}:{fid}] s{sd}: {m:.4f} ({time.time()-t1:.0f}s)")
    res["mean"] = round(float(np.mean(per)), 4)
    json.dump(res, open(outp, "w"), indent=1)
    log(f"[abl] {tag} {fid} 完成 mean={res['mean']}")
    return res


def main():
    force = "--force" in sys.argv
    jobs = [(f"{c}-{bb}-L{L_ABL}", c, bb, L_ABL)
            for c, bb in [("A", "PatchMLPLite"), ("B1", "PatchMLPLite")]]
    for tag, case, bb, L in jobs:
        for fid in FLOWS:
            try:
                run_one(tag, fid, case, bb, L, force=force)
            except Exception as e:
                print(f"[abl] {tag}×{fid} 失败（跳过）: {repr(e)[:120]}")
    # 汇总：与 L=128 标准对比
    lines = ["# 接口消融（R7, exploratory）：等上下文 L=512 vs 标准 L=128", "",
             "| 流 | A-PM L=512 | B1-PM L=512 | B1≥A? | A-PM L=128(M5a) | B1-PM L=128(M5b) | L128 B1≥A? |",
             "|---|---|---|---|---|---|---|"]
    for fid in FLOWS:
        p512a = os.path.join(OUT, f"A-L512_{fid}.json")
        p512b = os.path.join(OUT, f"B1-L512_{fid}.json")
        m5a = os.path.join(HERE, "..", "results", "m5", f"{fid}.json")
        m5b = os.path.join(HERE, "..", "results", "m5b", f"B1-PM_{fid}.json")
        a5, b5 = (json.load(open(m5a))["A-PM"]["mean"], json.load(open(m5b))["mean"])
        a5 = json.load(open(m5a, encoding="utf-8"))["A-PM"]["mean"]
        a512 = json.load(open(p512a))["mean"] if os.path.exists(p512a) else None
        b512 = json.load(open(p512b))["mean"] if os.path.exists(p512b) else None
        lines.append(f"| {fid} | {a512 if a512 is not None else '—'} | {b512 if b512 is not None else '—'} "
                     f"| {'是' if (b512 or 0) >= (a512 or 9) else '否'} | {a5:.4f} | {b5:.4f} "
                     f"| {'是' if b5 >= a5 else '否'} |")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
