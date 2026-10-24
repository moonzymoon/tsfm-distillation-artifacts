# -*- coding: utf-8 -*-
"""A-PM 确认集补充（DS v3 意见）：9 confirm 流 × 10 种子 = 90 次，CPU。
与 run_m6.py 同构，仅跑 A-PM 臂。产出并入 results/m6/。
用法: CUDA_VISIBLE_DEVICES="-1" python run_m6_apm.py
"""
import os
import sys
import json
import time
import ctypes
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from data import load_flow, all_flow_ids
from run_pilot import vus_pr, fidelity, L_WIN
from distill import train_case

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "m6")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(10))


def confirm_flow_ids():
    return sorted(i for i in all_flow_ids() if i.startswith("C"))


def main():
    def _keep_awake():
        ES_CONTINUOUS = 0x80000000; ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    threading.Thread(target=_keep_awake, daemon=True).start()
    print(f"[m6-apm] device={DEVICE}")
    for fid in confirm_flow_ids():
        outp = os.path.join(OUT, f"A-PM_{fid}.json")
        if os.path.exists(outp):
            print(f"[m6-apm] 跳过已有 {fid}")
            continue
        npz = np.load(os.path.join(CACHE, f"{fid}.npz"))
        xtr, xte, y = load_flow(fid)
        res = {"arm": "A-PM", "flow": fid, "device": DEVICE, "runs": {}}
        t0 = time.time()
        per = []
        for sd in SEEDS:
            t1 = time.time()
            fn, info = train_case("PatchMLPLite", xtr, npz["s_tr"], L=L_WIN, case="A",
                                  epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE,
                                  stride=2)
            q = fn(xte)
            m = vus_pr(q, y)
            f = fidelity(q, npz["s_te"])
            res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **f, **info}
            per.append(m)
            print(f"[m6-apm:{fid}] s{sd}: {m:.4f} ({time.time()-t1:.0f}s)", flush=True)
        res["mean"] = round(float(np.mean(per)), 4)
        res["std"] = round(float(np.std(per)), 4)
        res["wallclock_s"] = round(time.time() - t0, 1)
        json.dump(res, open(outp, "w"), indent=1)
        print(f"[m6-apm] A-PM {fid} 完成 mean={res['mean']} ({res['wallclock_s']}s)", flush=True)


if __name__ == "__main__":
    main()
