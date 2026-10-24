# -*- coding: utf-8 -*-
"""P4 试点：3任务×4主候选×5种子 + 案A log1p 消比 + PCA强基线 + 教师T0。
预注册 v1.0 §5 判据 → results/pilot/pilot_report.md + pilot_results.json
运行: python run_pilot.py [--flow D1_SWaT_id1]  （单流模式=P1试跑）
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
from students import TCNLite, PatchMLPLite, count_params
from distill import train_case
from export_time import export_onnx, time_onnx

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
OUT = os.path.join(HERE, "..", "results", "pilot")
os.makedirs(OUT, exist_ok=True)
PILOT_FLOWS = ["D1_SWaT_id1", "D11_SVDB_id2", "D3_SMD_id4"]
SLIDING_WINDOW = 100
SEEDS_MAIN, SEEDS_LOG1P = [0, 1, 2, 3, 4], [0, 1, 2]
L_WIN = 128
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def fill_nan(s):
    s = s.copy()
    v = s[~np.isnan(s)]
    if len(v) == 0:
        return np.zeros_like(s)
    s[np.isnan(s)] = v.min()
    return s


def vus_pr(scores, labels):
    from vus.metrics import get_metrics
    r = get_metrics(fill_nan(scores).astype(np.float64), labels.astype(int),
                    metric="vus", slidingWindow=SLIDING_WINDOW)
    return float(r["VUS_PR"])


def fidelity(s_stu, s_tea, q=0.05):
    v = ~(np.isnan(s_stu) | np.isnan(s_tea))
    a, b = s_stu[v], s_tea[v]
    from scipy.stats import spearmanr
    rho = float(spearmanr(a, b).statistic)
    if not np.isfinite(rho):                     # 常数分数（教师完全失效的流）→ 定义为0
        rho = 0.0
    nq = max(10, int(len(b) * q))
    top_t = set(np.argsort(-b)[:nq].tolist())
    top_s = set(np.argsort(-a)[:nq].tolist())
    top_overlap = len(top_t & top_s) / nq
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(b), 4000); j = rng.integers(0, len(b), 4000)
    m = (i != j) & (np.abs(b[i] - b[j]) > 1e-6)
    agree = float((np.sign(a[i[m]] - a[j[m]]) == np.sign(b[i[m]] - b[j[m]])).mean())
    return {"spearman": round(rho, 4), "pair_agree": round(agree, 4),
            f"top{int(q*100)}_overlap": round(top_overlap, 4)}


def train_one(cand, Xtr, s_tr, seed, xhat_tr=None, epochs=30):
    case = "A" if cand.startswith("A") else "C"
    arch = "TCNLite" if cand.endswith("TCN") else "PatchMLPLite"
    use_log1p = "log1p" in cand
    t0 = time.time()
    fn, info = train_case(arch, Xtr, s_tr, L=L_WIN, case=case, use_log1p=use_log1p,
                          epochs=epochs, bs=256, lr=1e-3, seed=seed, device=DEVICE,
                          stride=2, teacher_Xhat=xhat_tr)
    return fn, {**info, "train_s": round(time.time() - t0, 1)}


def run_flow(fid, quick=False, log=print):
    xtr, xte, y = load_flow(fid)
    z = np.load(os.path.join(CACHE, f"{fid}.npz"))
    s_tr, s_te = z["s_tr"], z["s_te"]
    seeds = [0] if quick else SEEDS_MAIN
    cands = {"A-TCN": [], "A-PM": [], "C-TCN": [], "C-PM": []}
    res = {"flow": fid, "test_AR": round(float(y.mean()), 4), "device": DEVICE, "runs": {}}

    t0 = time.time()
    pca = PCABaseline().fit(xtr)
    q_pca = pca.score(xte)
    res["pca"] = {"vus_pr": round(vus_pr(q_pca, y), 4), "storage_mb": round(pca.storage_mb, 3)}
    res["teacher"] = {"vus_pr": round(vus_pr(s_te, y), 4)}
    log(f"[pilot:{fid}] teacher VUS-PR={res['teacher']['vus_pr']}  PCA={res['pca']['vus_pr']}")

    for cand in cands:
        qs, fids = [], []
        for sd in seeds:
            fn, info = train_one(cand, xtr, s_tr, sd)
            q = fn(xte)
            m = vus_pr(q, y)
            qs.append(m)
            fids.append(fidelity(q, s_te))          # 预注册§5：尾部保真描述性证据
            res["runs"][f"{cand}-s{sd}"] = {"vus_pr": round(m, 4),
                                            **{k: v for k, v in fids[-1].items()}, **info}
            log(f"[pilot:{fid}] {cand} s{sd}: VUS-PR={m:.4f} ρ={fids[-1]['spearman']} "
                f"top5%={fids[-1]['top5_overlap']} ({info['train_s']}s)")
        res[cand] = {"mean": round(float(np.mean(qs)), 4), "std": round(float(np.std(qs)), 4),
                     "fid_spearman": round(float(np.mean([f["spearman"] for f in fids])), 4),
                     "fid_top5_overlap": round(float(np.mean([f["top5_overlap"] for f in fids])), 4),
                     "fid_pair_agree": round(float(np.mean([f["pair_agree"] for f in fids])), 4)}
    res["wallclock_s"] = round(time.time() - t0, 1)

    if not quick:  # 案A log1p 消比（3种子，仅 dev 决策用）
        for cand in ["A-log1p-TCN", "A-log1p-PM"]:
            qs = []
            for sd in SEEDS_LOG1P:
                fn, info = train_one(cand, xtr, s_tr, sd)
                qs.append(vus_pr(fn(xte), y))
            res[cand] = {"mean": round(float(np.mean(qs)), 4)}
    return res


def gate(res_by_flow):
    """预注册 §5：四候选宏平均选一；ΔST≥-0.02, ΔSB≥+0.02, ≥2/3任务ΔSB>0, min ΔSB≥-0.05。"""
    main = ["A-TCN", "A-PM", "C-TCN", "C-PM"]
    macro = {c: float(np.mean([res_by_flow[f][c]["mean"] for f in res_by_flow])) for c in main}
    best = max(macro, key=macro.get)
    rows = []
    for f, r in res_by_flow.items():
        q_best, q_t, q_b = r[best]["mean"], r["teacher"]["vus_pr"], r["pca"]["vus_pr"]
        rows.append({"flow": f, "Q_S": q_best, "Q_T": q_t, "Q_B": q_b,
                     "dST": round(q_best - q_t, 4), "dSB": round(q_best - q_b, 4)})
    dst = float(np.mean([r["dST"] for r in rows])); dsb = float(np.mean([r["dSB"] for r in rows]))
    gate = {"best_candidate": best, "macro_avg": {k: round(v, 4) for k, v in macro.items()},
            "per_flow": rows, "mean_dST": round(dst, 4), "mean_dSB": round(dsb, 4),
            "pass_mean_dST": dst >= -0.02, "pass_mean_dSB": dsb >= 0.02,
            "pass_2of3_dSB_pos": sum(r["dSB"] > 0 for r in rows) >= 2,
            "pass_min_dSB": min(r["dSB"] for r in rows) >= -0.05}
    gate["GO"] = all(gate[k] for k in ["pass_mean_dST", "pass_mean_dSB", "pass_2of3_dSB_pos", "pass_min_dSB"])
    return gate


def main():
    only = None
    if "--flow" in sys.argv:
        only = sys.argv[sys.argv.index("--flow") + 1]
    flows = [only] if only else PILOT_FLOWS
    by_flow = {}
    for fid in flows:
        quick = only is not None          # 单流模式 = P1 试跑（1种子）
        by_flow[fid] = run_flow(fid, quick=quick)
    if only:
        json.dump(by_flow, open(os.path.join(OUT, f"smoketest_{only}.json"), "w"), indent=1)
        print("[pilot] 单流试跑完成 →", f"results/pilot/smoketest_{only}.json")
        return
    g = gate(by_flow)
    best = g["best_candidate"]
    json.dump({"by_flow": by_flow, "gate": g},
              open(os.path.join(OUT, "pilot_results.json"), "w"), indent=1)
    lines = [f"# 试点报告（{datetime.datetime.now():%Y-%m-%d %H:%M}）",
             f"最优候选: **{g['best_candidate']}**  宏平均: {g['macro_avg']}",
             f"平均 ΔST={g['mean_dST']} (≥-0.02? {g['pass_mean_dST']})  平均 ΔSB={g['mean_dSB']} (≥+0.02? {g['pass_mean_dSB']})",
             f"≥2/3任务ΔSB>0: {g['pass_2of3_dSB_pos']}  min ΔSB≥-0.05: {g['pass_min_dSB']}",
             f"**GO = {g['GO']}**", ""]
    for r in g["per_flow"]:
        lines.append(f"- {r['flow']}: 学生={r['Q_S']} 教师={r['Q_T']} PCA={r['Q_B']} ΔST={r['dST']} ΔSB={r['dSB']}")
    bf = by_flow[flows[0]][best]
    lines.append(f"\n**最优候选保真度（{flows[0]}）**: Spearman ρ={bf.get('fid_spearman')} | "
                 f"top-5%重合率={bf.get('fid_top5_overlap')} | 非并列一致率={bf.get('fid_pair_agree')}")
    open(os.path.join(OUT, "pilot_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
