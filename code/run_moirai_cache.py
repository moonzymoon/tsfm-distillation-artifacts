# -*- coding: utf-8 -*-
"""Moirai 第三教师打分（R16 兑现 R9）：5 流逐通道一步预测残差评分。

口径（R16-A 锁定）：uni2ts 官方 MoiraiModule，context 1024（patch 32 对齐），
逐通道 one-step 预测（其他通道作为协变量静态传入=0 张量，单变量语义），
|x−x̂| 跨通道 max。权重本地 models/moirai-1.1-R-base。
产出：results/teacher_cache_moirai/{fid}.npz + report.md
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_flow

OUTC = os.path.join(HERE, "..", "results", "teacher_cache_moirai")
OUT = os.path.join(HERE, "..", "results", "moirai_subset")
os.makedirs(OUTC, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
LOCAL = os.path.join(HERE, "..", "models", "moirai-1.1-R-base")
CTX = 1024
SUBSET = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D16_NAB_occ6005", "D18_Exathlon_id1"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_cache(log=print):
    from uni2ts.model.moirai import MoiraiModule
    module = MoiraiModule.from_pretrained(LOCAL)
    module = module.to(DEVICE).eval()
    meta = {"teacher": "Salesforce/moirai-1.1-R-base (local, uni2ts MoiraiModule)",
            "context": CTX, "patch": 32, "one_step": True, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    for fid in SUBSET:
        npz = os.path.join(OUTC, fid + ".npz")
        if os.path.exists(npz):
            log("[moirai] skip %s" % fid)
            continue
        t0 = time.time()
        xtr, xte, y = load_flow(fid)
        T_tr, C = xtr.shape
        s_tr = np.zeros(T_tr, dtype=np.float32)
        xh_tr = np.zeros_like(xtr, dtype=np.float32)
        with torch.no_grad():
            for c in range(C):
                ser = xtr[:, c].astype(np.float32)
                xh = np.full(T_tr, np.nan, dtype=np.float32)
                for t in range(CTX, T_tr):
                    ctx = torch.tensor(ser[t - CTX:t][None, None, :], dtype=torch.float32, device=DEVICE)
                    ppt = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)
                    out = module(ctx, torch.tensor([[CTX]], dtype=torch.int64, device=DEVICE),
                                 ppt, torch.tensor([[0]], dtype=torch.int64, device=DEVICE))
                    xh[t] = float(out[0, 0, -1, 0])
                xh_tr[:, c] = xh
                log("[moirai:%s] train ch%d/%d %.0fs" % (fid, c + 1, C, time.time() - t0), flush=True)
        with np.errstate(invalid="ignore"):
            s_tr = np.nanmax(np.abs(xtr - xh_tr), axis=1)
        np.savez_compressed(npz, s_tr=s_tr, xhat_tr=xh_tr)
        json.dump(dict(meta, flow=fid, wallclock_s=round(time.time() - t0, 1)), open(npz.replace(".npz", ".meta.json"), "w"), indent=1)
        log("[moirai] %s cache done %.0fs" % (fid, time.time() - t0))


if __name__ == "__main__":
    build_cache()
