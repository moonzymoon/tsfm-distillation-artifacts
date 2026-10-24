# -*- coding: utf-8 -*-
"""Moirai 第三教师打分（R16 兑现 R9）：5 流逐通道残差评分，GPU 批量分块。

口径（R16-A 锁定）：uni2ts 1.1.1 MoiraiForecast（patch 32，context 1024，num_samples 20，
point = sample median），逐通道滚动一步预测，|x−x̂| 跨通道 max。
权重本地 models/moirai-1.1-R-base（离线）。
产出：results/teacher_cache_moirai/{fid}.npz（s_tr/s_te/xhat_*）+ meta。
之后 run_moirai_train.py 训练 C-PM×3 种子（s_tr）并算 fidelity（s_te）。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_flow

OUTC = os.path.join(HERE, "..", "results", "teacher_cache_moirai")
LOCAL = os.path.join(HERE, "..", "models", "moirai-1.1-R-base")
CTX = 1024
PATCH = 32
CHUNK = 1024          # 窗口/每次 predict 调用（约 8 个内部 batch）
NUM_SAMPLES = 20
import os as _os
_part = _os.environ.get("MOIRAI_PART", "A")
_ALL = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D16_NAB_occ6005", "D18_Exathlon_id1"]
SUBSET = _ALL[:3] if _part == "A" else _ALL[3:]
DEVICE = "cpu"  # RTX 5060 Ti sm_120 的 torch.distributions CUDA kernel 不兼容，锁 CPU


def build_predictor():
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained(LOCAL)
    module = module.float()   # bf16 算子在本机 GPU 崩溃，强制 fp32
    model = MoiraiForecast(
        module=module,
        prediction_length=1,
        context_length=CTX,
        patch_size=PATCH,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        num_samples=NUM_SAMPLES,
    )
    return model.create_predictor(batch_size=256, device=DEVICE)


def score_windows(predictor, ser, starts, logp, prefix):
    """对 starts 中每个 t：context=ser[t-CTX:t] 的中位数点预测。"""
    import pandas as pd
    s = np.full(int(starts.max()) + 1 if len(starts) else 0, np.nan, dtype=np.float32)
    done = 0
    base = pd.Period("2020-01-01 00:00", freq="H")
    for i in range(0, len(starts), CHUNK):
        chunk = starts[i:i + CHUNK]
        dataset = [{"item_id": "w%d" % k, "start": base,
                    "target": ser[t - CTX:t].astype(np.float32)}
                   for k, t in enumerate(chunk)]
        fcsts = list(predictor.predict(dataset, num_samples=NUM_SAMPLES))
        for k, t in enumerate(chunk):
            s[t] = float(np.asarray(fcsts[k].quantile(0.5)).reshape(-1)[0])
        done += len(chunk)
        logp("    %s %d/%d windows" % (prefix, done, len(starts)), flush=True)
    return s[starts]


def score_flow(predictor, x, logp, tag):
    T, C = x.shape
    Xhat = np.full((T, C), np.nan, dtype=np.float32)
    for c in range(C):
        ser = x[:, c].astype(np.float64)
        starts = np.arange(CTX, T)
        if len(starts) == 0:
            continue
        Xhat[starts, c] = score_windows(predictor, ser, starts, logp, "%s ch%d" % (tag, c + 1))
        logp("  %s ch%d/%d done (%.0fs)" % (tag, c + 1, C, 0), flush=True)
    with np.errstate(invalid="ignore"):
        s = np.nanmax(np.abs(x - Xhat), axis=1)
    return s.astype(np.float32), Xhat


def main():
    t00 = time.time()
    predictor = build_predictor()
    print("predictor ready %.0fs" % (time.time() - t00), flush=True)
    for fid in SUBSET:
        npz = os.path.join(OUTC, fid + ".npz")
        if os.path.exists(npz):
            print("[moirai] skip", fid, flush=True)
            continue
        t0 = time.time()
        xtr, xte, y = load_flow(fid)
        print("[moirai] %s train %s test %s" % (fid, list(xtr.shape), list(xte.shape)), flush=True)
        s_tr, xh_tr = score_flow(predictor, xtr, print, fid + ":train")
        s_te, xh_te = score_flow(predictor, xte, print, fid + ":test")
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        np.savez_compressed(npz, s_tr=s_tr, s_te=s_te, xhat_tr=xh_tr, xhat_te=xh_te)
        meta = {"teacher": "Salesforce/moirai-1.1-R-base (local, uni2ts 1.1.1 MoiraiForecast)",
                "context": CTX, "patch": PATCH, "num_samples": NUM_SAMPLES, "one_step": True,
                "flow": fid, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wallclock_s": round(time.time() - t0, 1)}
        json.dump(meta, open(npz.replace(".npz", ".meta.json"), "w"), indent=1)
        print("[moirai] %s DONE %.0fs" % (fid, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
