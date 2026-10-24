# -*- coding: utf-8 -*-
"""M6 确认集（5.3终裁 R4）：五臂 × 10种子 × 9 confirm 流 + 教师T0 + PCA。
CPU 模式运行（CUDA_VISIBLE_DEVICES="" 启动，让卡给博2）。
主对比对（预声明）：B1-TCN vs C-PM / LINEAR vs C-PM / B2-PM vs B1-PM。
流级(=来源级, n=9)配对 Wilcoxon + 效应量 + bootstrap 95%CI。
用法: CUDA_VISIBLE_DEVICES="" python run_m6.py [--force]
"""
import os
import sys
import json
import time
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from data import load_flow, all_flow_ids
from pca_baseline import PCABaseline
from distill import train_case
from run_pilot import vus_pr, fidelity, L_WIN

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "m6")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(10))
ARMS = {  # arm -> (case, backbone)
    "C-PM": ("C", "PatchMLPLite"),
    "B1-TCN": ("B1", "TCNLite"),
    "B1-PM": ("B1", "PatchMLPLite"),
    "B2-PM": ("B2", "PatchMLPLite"),
    "LINEAR": ("LINEAR", None),
}
PAIRS = [("B1-TCN", "C-PM"), ("LINEAR", "C-PM"), ("B2-PM", "B1-PM")]


def confirm_flow_ids():
    return sorted(i for i in all_flow_ids() if i.startswith("C"))


def run_flow_arm(fid, arm, force=False, log=print):
    outp = os.path.join(OUT, f"{arm}_{fid}.json")
    if os.path.exists(outp) and not force:
        return None
    npz = np.load(os.path.join(CACHE, f"{fid}.npz"))
    xtr, xte, y = load_flow(fid)
    if y.sum() == 0:
        log(f"[m6] {fid}: 测试段零异常，跳过所有臂")
        return None
    case, backbone = ARMS[arm]
    xhat_tr = npz["xhat_tr"] if case == "B2" else None
    res = {"arm": arm, "flow": fid, "device": DEVICE, "runs": {}}
    t0 = time.time()
    per = []
    for sd in SEEDS:
        t1 = time.time()
        fn, info = train_case(backbone, xtr, npz["s_tr"], L=L_WIN, case=case, epochs=30,
                              bs=256, lr=1e-3, seed=sd, device=DEVICE, stride=2,
                              teacher_Xhat=xhat_tr)
        q = fn(xte)
        m = vus_pr(q, y)
        f = fidelity(q, npz["s_te"])
        res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **f, **info}
        per.append(m)
        log(f"[m6:{arm}:{fid}] s{sd}: {m:.4f} ({time.time()-t1:.0f}s)")
    res["mean"] = round(float(np.mean(per)), 4)
    res["std"] = round(float(np.std(per)), 4)
    res["wallclock_s"] = round(time.time() - t0, 1)
    json.dump(res, open(outp, "w"), indent=1)
    log(f"[m6] {arm} {fid} 完成 mean={res['mean']}±{res['std']} ({res['wallclock_s']}s)")
    return res


def baselines(log=print):
    outp = os.path.join(OUT, "baselines.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for fid in confirm_flow_ids():
        if fid in res:
            continue
        npz = np.load(os.path.join(CACHE, f"{fid}.npz"))
        xtr, xte, y = load_flow(fid)
        if y.sum() == 0:
            log(f"[m6:base] {fid}: 测试段零异常 → 跳过（VUS不可算，R4记录）")
            res[fid] = {"teacher": None, "pca": None, "note": "test段零异常"}
            continue
        pca = PCABaseline().fit(xtr)
        res[fid] = {"teacher": round(vus_pr(npz["s_te"], y), 4),
                    "pca": round(vus_pr(pca.score(xte), y), 4),
                    "pca_storage_mb": round(pca.storage_mb, 3)}
        log(f"[m6:base] {fid}: teacher={res[fid]['teacher']} pca={res[fid]['pca']}")
        json.dump(res, open(outp, "w"), indent=1)      # 逐流落盘（断点安全）
    json.dump(res, open(outp, "w"), indent=1)
    return res


def stats(flow_means, base):
    """流级(=来源级)配对检验：Wilcoxon + 配对秩二列相关 r + bootstrap 95%CI。"""
    from scipy.stats import wilcoxon
    fids = sorted(flow_means.get(PAIRS[0][0], {}))
    lines = ["## 预声明主对比（流级配对，n=9）", ""]
    for a, b in PAIRS:
        d = np.array([flow_means[a][f] - flow_means[b][f] for f in fids])
        w = wilcoxon(d)
        # 配对秩二列相关（matched-pairs rank-biserial）
        n = len(d)
        r = 1 - 2 * (np.sum([1 for x in d if x < 0]) + 0.5 * np.sum([x == 0 for x in d])) / n
        rng = np.random.default_rng(0)
        bs = [float(np.mean(rng.choice(d, n))) for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"- **{a} vs {b}**: 流级均值差={d.mean():+.4f} "
                     f"(95%CI [{lo:+.4f}, {hi:+.4f}]), Wilcoxon p={w.pvalue:.4f}, "
                     f"秩二列相关 r={r:+.2f}, 胜/负/平="
                     f"{int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}")
    return "\n".join(lines)


def main():
    force = "--force" in sys.argv
    logp = print
    logp(f"[m6] device={DEVICE}，confirm流={confirm_flow_ids()}")
    base = baselines(log=logp)
    todo = [(arm, f) for f in confirm_flow_ids() for arm in ARMS
            if force or not os.path.exists(os.path.join(OUT, f"{arm}_{f}.json"))]
    logp(f"[m6] 待跑 {len(todo)} 项（臂×流）")
    for k, (arm, f) in enumerate(todo):
        logp(f"[m6] ({k+1}/{len(todo)}) {arm} × {f}")
        try:
            run_flow_arm(f, arm, force=force, log=logp)
        except Exception as e:
            logp(f"[m6] {arm} × {f} 失败（跳过继续）: {repr(e)[:140]}")
    # 汇总
    flow_means = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for f in confirm_flow_ids():
            p = os.path.join(OUT, f"{arm}_{f}.json")
            if os.path.exists(p):
                flow_means[arm][f] = json.load(open(p))["mean"]
    lines = [f"# M6 确认集结果（10种子，{datetime.datetime.now():%m-%d %H:%M}，device={DEVICE}）", "",
             "| 流 | " + " | ".join(ARMS) + " | 教师 | PCA |", "|" + "---|" * (len(ARMS) + 3)]
    for f in confirm_flow_ids():
        cells = [f"{flow_means[a].get(f, float('nan')):.3f}" if f in flow_means[a] else "—" for a in ARMS]
        lines.append(f"| {f} | " + " | ".join(cells) +
                     f" | {base[f]['teacher']:.3f} | {base[f]['pca']:.3f} |")
    mac = {a: float(np.mean(list(flow_means[a].values()))) for a in ARMS if flow_means[a]}
    lines.append("| **宏平均** | " + " | ".join(f"{mac.get(a, float('nan')):.4f}" for a in ARMS) + " | — | — |")
    txt = "\n".join(lines)
    st = stats(flow_means, base) if all(len(flow_means[a]) == 9 for a, _ in PAIRS) else "\n(统计待全量后补)"
    open(os.path.join(OUT, "m6_report.md"), "w", encoding="utf-8").write(txt + "\n\n" + st)
    print(txt); print(); print(st)


if __name__ == "__main__":
    import ctypes, threading
    def _keep_awake():
        ES_CONTINUOUS = 0x80000000; ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    threading.Thread(target=_keep_awake, daemon=True).start()
    main()
