# -*- coding: utf-8 -*-
"""n=10 三层分析：primary(9, 冻结复述) / extension(E2) / pooled(层级 secondary) + MDE 实测。"""
import json
import os

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.join(HERE, "..", "results", "m6")
M6E = os.path.join(HERE, "..", "results", "m6ext")
CB = os.path.join(HERE, "..", "results", "classical_baselines")

base = json.load(open(os.path.join(M6, "baselines.json"), encoding="utf-8"))
flows9 = sorted(base.keys())
E2 = "E2_CATSv2"

ARMS = ["C-PM", "A-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]


def m6_mean(arm, f):
    return json.load(open(os.path.join(M6, "%s_%s.json" % (arm, f)), encoding="utf-8"))["mean"]


def m6e_mean(arm, f):
    return json.load(open(os.path.join(M6E, "%s_%s.json" % (arm, f)), encoding="utf-8"))["mean"]


def rb(x, y):
    dd = np.asarray(x) - np.asarray(y)
    r = rankdata(np.abs(dd))
    return (r[dd > 0].sum() - r[dd < 0].sum()) / r.sum()


def boot_ci(diff, rng, n=5000):
    boots = [float(np.mean(rng.choice(diff, len(diff)))) for _ in range(n)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def cluster_boot(treat_by_source, ctrl_by_source, sources, rng, n=5000):
    """source-cluster bootstrap：按源整簇重采样差的均值。"""
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(sources), len(sources))
        stats.append(float(np.mean([treat_by_source[s] - ctrl_by_source[s] for s in pick])))
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


out = ["# n=10 三层分析（primary 9 冻结 / extension E2 独立 / pooled 层级 secondary）", ""]

# ---- primary 9（原样复述，不自算新数）----
out += ["## Primary（原 9 源，冻结）", "",
        "| 臂 | 宏均 |", "|---|---|"]
for arm in ARMS:
    out.append("| %s | %.4f |" % (arm, np.mean([m6_mean(arm, f) for f in flows9])))

# ---- extension E2 ----
ext = {}
for arm in ARMS:
    try:
        ext[arm] = m6e_mean(arm, E2)
    except FileNotFoundError:
        ext[arm] = None
out += ["", "## Extension（E2_CATSv2，独立新源）", "",
        "| 臂 | 宏均 |", "|---|---|"]
for arm in ARMS:
    out.append("| %s | %s |" % (arm, "%.4f" % ext[arm] if ext[arm] else "—"))

# 经典基线（E2）
cls = {}
for name in ["robustz", "mares", "lofwin"]:
    p = os.path.join(CB, "%s_%s.json" % (name, E2))
    if os.path.exists(p):
        cls[name] = json.load(open(p, encoding="utf-8"))["vus_pr"]
for k, v in cls.items():
    out.append("| %s (E2) | %.4f |" % (k, v))

# T0/PCA on E2（教师缓存里有 s_te → VUS-PR；PCA 见 run_extension report）
z = np.load(os.path.join(HERE, "..", "results", "teacher_cache", E2 + ".npz"))
import sys
sys.path.insert(0, HERE)
from run_pilot import vus_pr, load_flow
_, _, y = load_flow(E2)
t0_e2 = float(vus_pr(z["s_te"], y))
out.append("| T0 (E2) | %.4f |" % t0_e2)

# ---- pooled secondary（层级）----
out += ["", "## Pooled secondary（层级 bootstrap，流嵌套于源）", ""]
if all(ext[a] is not None for a in ARMS):
    for arm in ARMS:
        pooled = float(np.mean([m6_mean(arm, f) for f in flows9] + [ext[arm]]))
        out.append("| %s pooled(10源) | %.4f |" % (arm, pooled))
    # C-PM vs B1-PM 差（原9源 + E2）
    d9 = [m6_mean("C-PM", f) - m6_mean("B1-PM", f) for f in flows9]
    d_ext = ext["C-PM"] - ext["B1-PM"]
    out.append("- C-PM−B1-PM 差：原9源 %+.4f；E2 %+.4f；pooled %+.4f" % (
        np.mean(d9), d_ext, (np.mean(d9) * 9 + d_ext) / 10))

# ---- MDE 实测（基于原9源配对 SD 与 E2 加入后的口径）----
out += ["", "## MDE", ""]
for a, b in [("C-PM", "B1-PM")]:
    d9 = np.array([m6_mean(a, f) - m6_mean(b, f) for f in flows9])
    sd = d9.std(ddof=1)
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    m9 = [float(np.mean(rng.choice(d9, 9))) for _ in range(3000)]
    ci9 = [round(float(np.percentile(m9, 2.5)), 4), round(float(np.percentile(m9, 97.5)), 4)]
    out.append("- %s vs %s：paired SD %.4f → n=9 MDE ≈ %.3f；n=13（同 SD 假设）≈ %.3f"
               % (a, b, sd, 2.94 * sd / 3, 2.18 * sd / 3.606))

open(os.path.join(HERE, "..", "results", "ext_summary.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
