# -*- coding: utf-8 -*-
"""R16-B 扩展批次执行器：新源教师缓存 → 六臂×10 种子 → T0/PCA/三经典基线 → m6ext 落盘。

前置：data.py 已注册扩展流（E1/E2/E3/E4），load_flow 可用。
产出：results/teacher_cache/ 扩展流 npz + results/m6ext/*.json + report.md
用法：python run_extension.py   （GPU；每源约 60 runs + 教师打分）
"""
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

CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "m6ext")
CB = os.path.join(HERE, "..", "results", "classical_baselines")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(10))
ARMS = [("C-PM", "C", "PatchMLPLite"), ("B1-TCN", "B1", "TCNLite"),
        ("B1-PM", "B1", "PatchMLPLite"), ("B2-PM", "B2", "PatchMLPLite"),
        ("A-PM", "A", "PatchMLPLite"), ("LINEAR", "LINEAR", None)]

EXT_FLOWS = json.load(open(os.path.join(HERE, "..", "results", "extension_flows.json"), encoding="utf-8"))
FLOW_IDS = {"CreditCard": "E1_CreditCard", "CATSv2": "E2_CATSv2",
            "TAO": "E3_TAO", "LTDB": "E4_LTDB"}


def teacher_cache(fid):
    npz = os.path.join(CACHE, fid + ".npz")
    if os.path.exists(npz):
        return np.load(npz)
    from teacher import BoltScorer
    sc = BoltScorer()
    xtr, xte, y = load_flow(fid)
    s_tr, xh_tr = sc.score_flow(xtr, batch_size=2048)
    s_te, xh_te = sc.score_flow(xte, batch_size=2048)
    np.savez_compressed(npz, s_tr=s_tr, s_te=s_te, xhat_tr=xh_tr, xhat_te=xh_te)
    return np.load(npz)


def classical(fid):
    from run_classical_baselines import robust_z, ma_residual, lof_window
    xtr, xte, y = load_flow(fid)
    out = {}
    for name, fn in [("robustz", robust_z), ("mares", ma_residual), ("lofwin", lof_window)]:
        p = os.path.join(CB, "%s_%s.json" % (name, fid))
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))["vus_pr"]
        else:
            s = fn(xtr, xte)
            m = float(vus_pr(s, y))
            json.dump({"method": name, "flow": fid, "vus_pr": round(m, 4)},
                      open(p, "w"), indent=1)
            out[name] = m
    return out


def pca_t0(fid):
    """PCA(训练段拟合) 与 LINEAR 已在六臂中；T0=教师分数已有；PCA 重算："""
    from sklearn.decomposition import PCA
    xtr, xte, y = load_flow(fid)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    p = PCA(n_components=0.95, random_state=0).fit((xtr - mu) / sd)
    z = (xte - mu) / sd
    rec = p.inverse_transform(p.transform(z)) * sd + mu
    s = np.max(np.abs(z - rec * sd) / 1.0, axis=1) if False else np.max(np.abs(z - p.transform(z)), axis=1)
    return float(vus_pr(s.astype(np.float32), y))


def main():
    from sklearn.decomposition import PCA
    report = ["# 扩展批次结果（R16-B prospective extension）", ""]
    allres = {}
    for src, info in EXT_FLOWS.items():
        fid = FLOW_IDS[src]
        xtr, xte, y = load_flow(fid)
        z = teacher_cache(fid)
        C = xtr.shape[1]
        # PCA（训练段拟合 95% 方差）
        from sklearn.decomposition import PCA as SKPCA
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        p = SKPCA(n_components=0.95, random_state=0).fit((xtr - mu) / sd)
        zt = (xte - mu) / sd
        rec = p.inverse_transform(p.transform(zt))
        pca_score = np.max(np.abs(zt - rec), axis=1).astype(np.float32)
        pca_m = float(vus_pr(pca_score, y))
        cls = classical(fid)
        t0_m = float(vus_pr(z["s_te"], y))
        report += ["## %s (C=%d)" % (fid, C), "",
                   "| 方法 | VUS-PR |", "|---|---|",
                   "| frozen teacher T0 | %.4f |" % t0_m,
                   "| PCA (fixed, 95%% var) | %.4f |" % pca_m]
        for name, v in cls.items():
            report.append("| %s (zero tuning) | %.4f |" % (name, v))
        for arm, case, bb in ARMS:
            outp = os.path.join(OUT, "%s_%s.json" % (arm, fid))
            if os.path.exists(outp):
                m = json.load(open(outp, encoding="utf-8"))["mean"]
            else:
                per = []
                res = {"arm": arm, "flow": fid, "device": DEVICE, "runs": {}}
                for sd in SEEDS:
                    t1 = time.time()
                    kw = dict(teacher_Xhat=z["xhat_tr"]) if case == "B2" else {}
                    fn, info = train_case(bb, xtr, z["s_tr"], L=L_WIN, case=case,
                                          epochs=30, bs=256, lr=1e-3, seed=sd,
                                          device=DEVICE, stride=2, **kw)
                    q = fn(xte)
                    mm = float(vus_pr(q, y))
                    f = fidelity(q, z["s_te"])
                    res["runs"]["s%d" % sd] = {"vus_pr": round(mm, 4), **f, **info}
                    per.append(mm)
                    print("[%s %s s%d] %.4f (%.0fs)" % (arm, fid, sd, mm, time.time() - t1), flush=True)
                res["mean"] = round(float(np.mean(per)), 4)
                json.dump(res, open(outp, "w"), indent=1)
                m = res["mean"]
            report.append("| %s | %.4f |" % (arm, m))
            allres["%s_%s" % (arm, fid)] = m
        report.append("")
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(report))
    json.dump(allres, open(os.path.join(OUT, "all_means.json"), "w"), indent=1)
    print("EXTENSION COMPLETE")


if __name__ == "__main__":
    main()
