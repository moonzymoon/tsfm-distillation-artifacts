# -*- coding: utf-8 -*-
"""TimesFM 第二教师小子集（次要臂）：5预锁定流 × C-PM×5种子（TimesFM教师分数）
对比 C-PM（Chronos教师，M5a结果）+ TimesFM教师自身T0。
产出：results/timesfm_subset/report.md
用法: python run_timesfm_subset.py   （GPU；教师打分约1.5-2.5h + 25次训练）
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
from timesfm_scorer import TimesFMScorer

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
TF_CACHE = os.path.join(HERE, "..", "results", "teacher_cache_timesfm")
OUT = os.path.join(HERE, "..", "results", "timesfm_subset")
os.makedirs(TF_CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
SUBSET = ["C2_Genesis_id1", "C3_Daphnet_id1", "C4_MGAB", "C5_YAHOO", "C6_IOPS", "C7_MITDB_id5", "C8_NEweather", "C9_NAB_adexchange", "D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D18_Exathlon_id1", "D12_EMG"]


def build_timesfm_cache(log=print):
    sc = TimesFMScorer()
    meta = {"teacher": "google/timesfm-2.5-200m-pytorch (torch)", "context_len": sc.ctx,
            "point_forecast": "mean (模型自然点输出)", "built": datetime.datetime.now().isoformat(timespec="seconds")}
    for fid in SUBSET:
        npz = os.path.join(TF_CACHE, f"{fid}.npz")
        if os.path.exists(npz):
            log(f"[tf] 跳过已有 {fid}")
            continue
        t0 = time.time()
        xtr, xte, y = load_flow(fid)
        s_tr, xh_tr = sc.score_flow(xtr, batch_size=64)
        s_te, xh_te = sc.score_flow(xte, batch_size=64)
        np.savez_compressed(npz, s_tr=s_tr, s_te=s_te, xhat_tr=xh_tr, xhat_te=xh_te)
        meta2 = dict(meta, flow=fid, wallclock_s=round(time.time() - t0, 1),
                     train_shape=list(xtr.shape), test_shape=list(xte.shape),
                     test_AR=round(float(y.mean()), 4))
        json.dump(meta2, open(npz.replace(".npz", ".meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        log(f"[tf] {fid} 完成 {meta2['wallclock_s']}s")


def main():
    logp = print
    build_timesfm_cache(log=logp)
    results = {}
    for fid in SUBSET:
        outp = os.path.join(OUT, f"C-PM-TF_{fid}.json")
        skp = os.path.join(OUT, f"C-PM-TF_{fid}.SKIPPED.json")
        if os.path.exists(outp) or os.path.exists(skp):
            continue
        z = np.load(os.path.join(TF_CACHE, f"{fid}.npz"))
        xtr, xte, y = load_flow(fid)
        if np.isfinite(z["s_tr"]).sum() == 0 or np.isfinite(z["s_te"]).sum() == 0:
            reason = ("series shorter than locked context 1024: teacher scores all-NaN "
                      f"(s_tr finite {int(np.isfinite(z['s_tr']).sum())}/{len(z['s_tr'])}, "
                      f"s_te finite {int(np.isfinite(z['s_te']).sum())}/{len(z['s_te'])})")
            json.dump({"arm": "C-PM-TF", "flow": fid, "skipped": True, "reason": reason},
                      open(skp, "w"), indent=1)
            logp(f"[tf:{fid}] SKIPPED: {reason}")
            continue
        res = {"arm": "C-PM-TF", "flow": fid, "device": DEVICE, "runs": {}}
        per = []
        try:
            for sd in range(5):
                t1 = time.time()
                fn, info = train_case("PatchMLPLite", xtr, z["s_tr"], L=L_WIN, case="C",
                                      epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE, stride=2)
                q = fn(xte)
                m = vus_pr(q, y)
                f = fidelity(q, z["s_te"])
                res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **f, **info}
                per.append(m)
                logp(f"[tf:{fid}] C-PM-TF s{sd}: {m:.4f} ({time.time()-t1:.0f}s)")
            res["mean"] = round(float(np.mean(per)), 4)
            json.dump(res, open(outp, "w"), indent=1)
        except Exception as e:
            json.dump({"arm": "C-PM-TF", "flow": fid, "skipped": True,
                       "reason": str(e)[:300]}, open(skp, "w"), indent=1)
            logp(f"[tf:{fid}] SKIPPED on error: {str(e)[:150]}")
            continue
        results[fid] = res["mean"]
    # 汇总对比
    lines = ["# TimesFM 第二教师小子集（C-PM 配置，5种子）", "",
             "| 流 | C-PM(TimesFM教师) | C-PM(Chronos教师,M5a) | TimesFM T0 |", "|---|---|---|---|"]
    for fid in SUBSET:
        p = os.path.join(OUT, f"C-PM-TF_{fid}.json")
        tfv = json.load(open(p))["mean"] if os.path.exists(p) else None
        m5p = os.path.join(HERE, "..", "results", "m5", f"{fid}.json")
        chv = json.load(open(m5p))["C-PM"]["mean"] if os.path.exists(m5p) else None
        tz = os.path.join(TF_CACHE, f"{fid}.npz")
        t0v = None
        if os.path.exists(tz):
            s_te = np.load(tz)["s_te"]
            _, _, y = load_flow(fid)
            t0v = round(vus_pr(s_te, y), 4)
        lines.append(f"| {fid} | {tfv if tfv is not None else '—'} | {chv if chv is not None else '—'} | {t0v if t0v is not None else '—'} |")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
