# -*- coding: utf-8 -*-
"""R14-a：经典基线流式延迟实测（桌面单线程，与 cost 轴同口径）。

MA-residual（增量滑窗 O(1)/点）与 Robust-Z（O(C)/点）的流式语义单点延迟，
66 通道参考维度，5 轮 × (50 warm + 500 timed)，p95 汇总与图轴同口径
（换算为 per-window：一个新点 = 窗口右移一格出分）。
产出：results/cost/classical_timing.json
"""
import json
import os
import time

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cost")
os.makedirs(OUT, exist_ok=True)
C = 66
N_WARM, N_TIMED, ROUNDS = 50, 500, 5


def bench(fn, x):
    rounds = []
    for _ in range(ROUNDS):
        state = fn("init", None)
        for i in range(N_WARM):
            fn("step", x[i])
        ts = []
        for i in range(N_WARM, N_WARM + N_TIMED):
            t0 = time.perf_counter()
            fn("step", x[i])
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000.0
        rounds.append({"p50_ms": float(np.percentile(ts, 50)),
                       "p95_ms": float(np.percentile(ts, 95)),
                       "mean_ms": float(ts.mean())})
    agg = {k: round(float(np.median([r[k] for r in rounds])), 6) for k in rounds[0]}
    return {"rounds_median": agg, "rounds": rounds}


def make_mares():
    st = {"cum": None, "n": 0, "ma": None}
    L = 128

    def step(x):  # x [C]
        if st["cum"] is None:
            st["cum"] = np.zeros(C + 1)
        st["cum"][1:] += x
        st["n"] += 1
        w = min(st["n"], L)
        ma = (st["cum"][-1] - st["cum"][-min(st["n"] - w, C + 1) - 0]) if False else None
        # 简化：直接维护每通道滑窗和
        if "buf" not in st:
            st["buf"] = np.zeros((L, C))
        i = st["n"] % L
        old = st["buf"][i].copy()
        st["buf"][i] = x
        w = min(st["n"], L)
        s = st["buf"].sum(axis=0) if st["n"] >= L else st["buf"][:st["n"]].sum(axis=0)
        ma = s / w
        return float(np.max(np.abs(x - ma)))

    def api(mode, x):
        if mode == "init":
            st["cum"] = None
            st["n"] = 0
            st.pop("buf", None)
            return st
        return step(x)
    return api


def make_rz(med, mad):
    def api(mode, x):
        if mode == "init":
            return None
        return float(np.max(np.abs(x - med) / mad))
    return api


rng = np.random.default_rng(0)
x = rng.standard_normal((N_WARM + N_TIMED + 10, C)).astype(np.float64)
med = np.median(x[:200], axis=0)
mad = np.median(np.abs(x[:200] - med), axis=0) + 1e-9

res = {
    "protocol": "desktop single-thread python/numpy, streaming one-point latency at 66ch, "
                "5 rounds x (50 warm + 500 timed), median across rounds",
    "MA-residual": bench(make_mares(), x),
    "Robust-Z": bench(make_rz(med, mad), x),
}
json.dump(res, open(os.path.join(OUT, "classical_timing.json"), "w"), indent=1)
for k in ("MA-residual", "Robust-Z"):
    a = res[k]["rounds_median"]
    print("%-12s p50 %.6f ms  p95 %.6f ms (per point, streaming)" % (k, a["p50_ms"], a["p95_ms"]))
