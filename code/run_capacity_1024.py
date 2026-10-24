# -*- coding: utf-8 -*-
"""R13：PatchMLP1024 容量追档（45 runs，协议同 R10-P1）。产出并入 results/capacity_sweep/。"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_flow
from run_pilot import vus_pr, fidelity, L_WIN, DEVICE
from distill import train_case
from students import PatchMLPLite, count_params

CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "capacity_sweep")
os.makedirs(OUT, exist_ok=True)

FLOWS = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D18_Exathlon_id1", "D12_EMG"]
SEEDS = [0, 1, 2]


class PatchMLP1024(PatchMLPLite):
    def __init__(self, c_in, L=128, **kw):
        super().__init__(c_in, L, dim=1024)


for fid in FLOWS:
    z = np.load(os.path.join(CACHE, fid + ".npz"))
    xtr, xte, y = load_flow(fid)
    C = xtr.shape[1]
    n_params = count_params(PatchMLP1024(C, L=L_WIN))
    print("[%s] PatchMLP1024 params=%s" % (fid, "{:,}".format(n_params)), flush=True)
    for case in ["C", "B2", "B1"]:
        outp = os.path.join(OUT, "%s_PatchMLP1024_%s.json" % (case, fid))
        if os.path.exists(outp):
            continue
        res = {"arm": case, "backbone": "PatchMLP1024", "flow": fid, "params": n_params,
               "device": DEVICE, "runs": {}}
        per, rhos = [], []
        for sd in SEEDS:
            t1 = time.time()
            fn, info = train_case("PatchMLP1024", xtr, z["s_tr"], L=L_WIN, case=case,
                                  epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE,
                                  stride=2, teacher_Xhat=(z["xhat_tr"] if case == "B2" else None))
            q = fn(xte)
            m = vus_pr(q, y)
            f = fidelity(q, z["s_te"])
            res["runs"]["s%d" % sd] = {"vus_pr": round(m, 4), **f, **info}
            per.append(m)
            rhos.append(f.get("spearman", float("nan")))
            print("  %s s%d: %.4f (%.0fs)" % (case, sd, m, time.time() - t1), flush=True)
        res["mean"] = round(float(np.mean(per)), 4)
        res["median_spearman"] = round(float(np.nanmedian(rhos)), 4)
        json.dump(res, open(outp, "w"), indent=1)
print("R13 sweep done")
