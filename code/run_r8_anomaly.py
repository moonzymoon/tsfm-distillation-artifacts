# -*- coding: utf-8 -*-
"""R8 教师-异常蒸馏对照（exploratory）：
阶段1：合成异常注入 → 教师对增广段打分（GPU）→ results/teacher_cache_aug/
阶段2：A-PM-aug / B1-PM-aug × 5流 × 3种子 → results/r8_anomaly/
阶段3：与原 A-PM/B1-PM 对比报告
用法: python run_r8_anomaly.py [--stage 1|2|3|all]
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
from data import load_flow
from teacher import BoltScorer
from run_pilot import vus_pr, fidelity, L_WIN
from distill import train_case

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
CACHE_AUG = os.path.join(HERE, "..", "results", "teacher_cache_aug")
OUT = os.path.join(HERE, "..", "results", "r8_anomaly")
os.makedirs(CACHE_AUG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
FLOWS = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D16_NAB_occ6005", "D18_Exathlon_id1"]
SEEDS = [0, 1, 2]
INJECT_SEED = 999


def augment(xtr):
    """R8 锁定：瞬态尖峰+持续电平移各约1%，k~{3,5,8}σ，seed=999。"""
    rng = np.random.default_rng(INJECT_SEED)
    x = xtr.copy()
    T, C = x.shape
    sig = x.std(axis=0) + 1e-8
    n_inj = max(1, int(T * 0.01))
    labels = np.zeros(T, dtype=bool)
    for _ in range(n_inj):  # 尖峰
        t = rng.integers(0, T)
        c = rng.integers(0, C)
        k = rng.choice([3.0, 5.0, 8.0])
        x[t, c] += k * sig[c]
        labels[t] = True
    for _ in range(n_inj):  # 电平移（100点）
        t = rng.integers(0, max(1, T - 100))
        c = rng.integers(0, C)
        k = rng.choice([3.0, 5.0, 8.0])
        x[t:t + 100, c] += k * sig[c]
        labels[t:t + 100] = True
    return x, labels


def stage1():
    sc = BoltScorer("amazon/chronos-bolt-base")
    for fid in FLOWS:
        npz = os.path.join(CACHE_AUG, f"{fid}.npz")
        if os.path.exists(npz):
            print(f"[r8-s1] 跳过 {fid}")
            continue
        xtr, _, _ = load_flow(fid)
        xaug, inj_labels = augment(xtr)
        t0 = time.time()
        s_aug, xh_aug = sc.score_flow(xaug, batch_size=2048)
        np.savez_compressed(npz, x_aug=xaug, s_aug=s_aug, xhat_aug=xh_aug,
                            inj_labels=inj_labels)
        meta = {"flow": fid, "teacher": "amazon/chronos-bolt-base",
                "injection": "1% spikes + 1% level-shifts(100pts), k in {3,5,8}sigma, seed=999",
                "built": datetime.datetime.now().isoformat(timespec="seconds"),
                "wallclock_s": round(time.time() - t0, 1)}
        json.dump(meta, open(npz.replace(".npz", ".meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        # 教师对注入异常的反应强度（诊断）
        v = ~np.isnan(s_aug)
        react = float(np.nanmean(s_aug[inj_labels & v])) if (inj_labels & v).any() else float("nan")
        base = float(np.nanmean(s_aug[~inj_labels & v]))
        print(f"[r8-s1] {fid}: 教师分@注入={react:.3f} vs @正常={base:.3f} ({meta['wallclock_s']}s)")


def stage2():
    for fid in FLOWS:
        z_aug = np.load(os.path.join(CACHE_AUG, f"{fid}.npz"))
        xaug = z_aug["x_aug"]
        s_aug = z_aug["s_aug"]
        _, xte, y = load_flow(fid)
        for arm, case in [("A-PM-aug", "A"), ("B1-PM-aug", "B1")]:
            outp = os.path.join(OUT, f"{arm}_{fid}.json")
            if os.path.exists(outp):
                continue
            res = {"arm": arm, "flow": fid, "device": DEVICE, "runs": {}}
            per = []
            for sd in SEEDS:
                t1 = time.time()
                fn, info = train_case("PatchMLPLite", xaug, s_aug, L=L_WIN, case=case,
                                      epochs=30, bs=256, lr=1e-3, seed=sd, device=DEVICE,
                                      stride=2)
                q = fn(xte)
                m = vus_pr(q, y)
                z_orig = np.load(os.path.join(CACHE, f"{fid}.npz"))
                f = fidelity(q, z_orig["s_te"])
                res["runs"][f"s{sd}"] = {"vus_pr": round(m, 4), **f, **info}
                per.append(m)
                print(f"[r8-s2:{arm}:{fid}] s{sd}: {m:.4f} ({time.time()-t1:.0f}s)", flush=True)
            res["mean"] = round(float(np.mean(per)), 4)
            json.dump(res, open(outp, "w"), indent=1)
            print(f"[r8-s2] {arm} {fid} mean={res['mean']}", flush=True)


def stage3():
    import glob
    lines = ["# R8 教师-异常蒸馏对照（exploratory）", "",
             "| 流 | A-PM-aug | A-PM(原) | Δ(A) | B1-PM-aug | B1-PM(原) | Δ(B1) |",
             "|---|---|---|---|---|---|---|"]
    dA, dB = [], []
    for fid in FLOWS:
        a_aug = json.load(open(os.path.join(OUT, f"A-PM-aug_{fid}.json")))["mean"] \
            if os.path.exists(os.path.join(OUT, f"A-PM-aug_{fid}.json")) else None
        b_aug = json.load(open(os.path.join(OUT, f"B1-PM-aug_{fid}.json")))["mean"] \
            if os.path.exists(os.path.join(OUT, f"B1-PM-aug_{fid}.json")) else None
        a_orig = json.load(open(os.path.join(CACHE_AUG, "..", "m5", f"{fid}.json")))["A-PM"]["mean"]
        b_orig = json.load(open(os.path.join(CACHE_AUG, "..", "m5b", f"B1-PM_{fid}.json")))["mean"]
        if a_aug is not None:
            dA.append(a_aug - a_orig)
        if b_aug is not None:
            dB.append(b_aug - b_orig)
        dA_s = f"{a_aug - a_orig:+.4f}" if a_aug is not None else "—"
        dB_s = f"{b_aug - b_orig:+.4f}" if b_aug is not None else "—"
        a_s = f"{a_aug:.4f}" if a_aug is not None else "—"
        b_s = f"{b_aug:.4f}" if b_aug is not None else "—"
        lines.append(f"| {fid} | {a_s} | {a_orig:.4f} | {dA_s} | "
                     f"{b_s} | {b_orig:.4f} | {dB_s} |")
    if dA:
        lines.append("")
        lines.append(f"**Δ(A) 均值 {np.mean(dA):+.4f}（{len(dA)}流）；Δ(B1) 均值 {np.mean(dB):+.4f}**")
        lines.append("读法（R8 预锁定）：Δ(A)≈0 → 接口主张强化；Δ(A)>0 显著 → 约束在数据，叙事转向'正常段条件下'。")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"
    if stage in ("1", "all"):
        stage1()
    if stage in ("2", "all"):
        stage2()
    if stage in ("3", "all"):
        stage3()
