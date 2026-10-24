# -*- coding: utf-8 -*-
"""M5a 核心实验（预注册 §3/§7）：dev 20 流 × 主网格4配置(C-PM/C-TCN/A-PM/A-TCN) × 5种子 = 400次。
log1p 已锁定不使用（5.3裁定2）。逐流幂等（results/m5/{flow}.json 存在即跳过，--force 重跑）。
选型不做（M5 末 5.3 判断节点）。用法: python run_m5.py [--force] [--flow FID]
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
from run_pilot import train_one, vus_pr, fidelity, fill_nan, SLIDING_WINDOW, L_WIN, DEVICE

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "m5")
os.makedirs(OUT, exist_ok=True)
CANDS = ["C-PM", "C-TCN", "A-PM", "A-TCN"]
SEEDS = [0, 1, 2, 3, 4]


def dev_flow_ids():
    from data import all_flow_ids
    ids = [i for i in all_flow_ids() if i.startswith("D")]
    order = ["D1_SWaT_id1", "D11_SVDB_id2", "D3_SMD_id4"]
    return order + [i for i in ids if i not in order]


def run_flow(fid, force=False, log=print):
    outp = os.path.join(OUT, f"{fid}.json")
    if os.path.exists(outp) and not force:
        log(f"[m5] 跳过已有 {fid}")
        return None
    npz = os.path.join(CACHE, f"{fid}.npz")
    if not os.path.exists(npz):
        log(f"[m5] {fid} 缓存缺失，跳过（P2未完成？）")
        return None
    z = np.load(npz)
    s_tr, s_te = z["s_tr"], z["s_te"]
    xtr, xte, y = load_flow(fid)
    res = {"flow": fid, "test_shape": list(xte.shape), "test_AR": round(float(y.mean()), 4),
           "device": DEVICE, "runs": {}}
    t0 = time.time()
    res["teacher"] = {"vus_pr": round(vus_pr(s_te, y), 4)}
    pca = PCABaseline().fit(xtr)
    res["pca"] = {"vus_pr": round(vus_pr(pca.score(xte), y), 4),
                  "storage_mb": round(pca.storage_mb, 3)}
    log(f"[m5:{fid}] teacher={res['teacher']['vus_pr']} pca={res['pca']['vus_pr']} "
        f"({time.time()-t0:.0f}s基线)")
    for cand in CANDS:
        per = []
        for sd in SEEDS:
            t1 = time.time()
            fn, info = train_one(cand, xtr, s_tr, sd)
            q = fn(xte)
            m = vus_pr(q, y)
            f = fidelity(q, s_te)
            res["runs"][f"{cand}-s{sd}"] = {"vus_pr": round(m, 4), **f, **info}
            per.append(m)
            log(f"[m5:{fid}] {cand} s{sd}: {m:.4f} ρ={f['spearman']} ({time.time()-t1:.0f}s)")
        res[cand] = {"mean": round(float(np.mean(per)), 4), "std": round(float(np.std(per)), 4),
                     "fid_spearman": round(float(np.mean(
                         [res['runs'][f'{cand}-s{sd}']['spearman'] for sd in SEEDS])), 4),
                     "fid_top5_overlap": round(float(np.mean(
                         [res['runs'][f'{cand}-s{sd}']['top5_overlap'] for sd in SEEDS])), 4)}
        json.dump(res, open(outp, "w"), indent=1)     # 每候选落盘一次（中断保进度）
    res["wallclock_s"] = round(time.time() - t0, 1)
    json.dump(res, open(outp, "w"), indent=1)
    log(f"[m5] {fid} 完成 {res['wallclock_s']}s")
    return res


def main():
    force = "--force" in sys.argv
    only = sys.argv[sys.argv.index("--flow") + 1] if "--flow" in sys.argv else None
    summary = {}
    for fid in dev_flow_ids():
        if only and fid != only:
            continue
        r = run_flow(fid, force=force)
        if r:
            summary[fid] = {c: r[c]["mean"] for c in CANDS} | {
                "teacher": r["teacher"]["vus_pr"], "pca": r["pca"]["vus_pr"]}
    if summary and not only:
        lines = ["# M5a 主网格结果（dev 流，5种子 mean）", "",
                 "| 流 | C-PM | C-TCN | A-PM | A-TCN | 教师 | PCA |", "|---|---|---|---|---|---|---|"]
        for f, d in summary.items():
            lines.append(f"| {f} | {d['C-PM']} | {d['C-TCN']} | {d['A-PM']} | {d['A-TCN']} "
                         f"| {d['teacher']} | {d['pca']} |")
        macro = {c: round(float(np.mean([d[c] for d in summary.values()])), 4) for c in CANDS}
        lines += ["", f"**宏平均**: {macro}", "",
                  f"（生成于 {datetime.datetime.now():%m-%d %H:%M}；选型留待 5.3 判断节点）"]
        open(os.path.join(OUT, "m5_report.md"), "w", encoding="utf-8").write("\n".join(lines))
        print("\n".join(lines[-4:]))
        print("[m5] 报告 → results/m5/m5_report.md")


if __name__ == "__main__":
    main()
