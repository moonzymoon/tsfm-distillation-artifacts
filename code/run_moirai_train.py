# -*- coding: utf-8 -*-
"""Moirai 教师缓存 → C-PM×3 种子训练（R16-A 后半）。产出 results/moirai_subset/。"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_flow
from run_pilot import vus_pr, fidelity, L_WIN, DEVICE

OUT = os.path.join(HERE, "..", "results", "moirai_subset")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cpu"  # RTX 5060 Ti backward-pass kernels 缺失，锁 CPU
from distill import train_case

OUT = os.path.join(HERE, "..", "results", "moirai_subset")
CACH = os.path.join(HERE, "..", "results", "teacher_cache_moirai")
SUBSET = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D16_NAB_occ6005", "D18_Exathlon_id1"]


def main():
    for fid in SUBSET:
        outp = os.path.join(OUT, "C-PM-Moirai_%s.json" % fid)
        if os.path.exists(outp):
            print("[train] skip", fid, flush=True)
            continue
        cachep = os.path.join(CACH, fid + ".npz")
        if not os.path.exists(cachep):
            print("[train] cache missing, skip", fid, flush=True)
            continue
        z = np.load(cachep)
        xtr, xte, y = load_flow(fid)
        res = {"arm": "C-PM-Moirai", "flow": fid, "device": DEVICE, "runs": {}}
        per, rho = [], []
        for sd in range(3):
            t1 = time.time()
            fn, info = train_case("PatchMLPLite", xtr, z["s_tr"], L=L_WIN, case="C",
                                  epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE, stride=2)
            q = fn(xte)
            m = vus_pr(q, y)
            f = fidelity(q, z["s_te"])
            res["runs"]["s%d" % sd] = {"vus_pr": round(m, 4), **f, **info}
            per.append(m)
            rho.append(f["spearman"])
            print("[train %s s%d] %.4f rho %.3f (%.0fs)" % (fid, sd, m, f["spearman"], time.time() - t1), flush=True)
        res["mean"] = round(float(np.mean(per)), 4)
        res["median_spearman"] = round(float(np.median(rho)), 4)
        json.dump(res, open(outp, "w"), indent=1)
        print("[train] %s mean %.4f median_rho %.4f" % (fid, res["mean"], res["median_spearman"]), flush=True)
    print("MOIRAI TRAIN COMPLETE")


if __name__ == "__main__":
    main()
