# -*- coding: utf-8 -*-
"""冒烟测试（CPU）：合成流上验证 学生前向/反向、参数量、案A/C/B1/B2/LINEAR 训练、ONNX导出+计时；
真实流验证 data.py 两个小加载器（J-D1 service0、NAB）。教师全链路待夜晚窗口（bolt-mini 下载）。
运行：python smoke_test.py
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from students import TCNLite, PatchMLPLite, count_params, make_windows
from distill import train_case
from export_time import export_onnx, time_onnx

PASS = []
def ok(name, cond, extra=""):
    PASS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

def synth(T=3000, C=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = np.stack([np.sin(t / 23 + c) + 0.3 * rng.standard_normal(T) for c in range(C)], 1).astype(np.float32)
    sT = np.abs(rng.standard_normal(T)).astype(np.float32) + (t % 500 < 5) * 5.0   # 假教师分
    x[300:305] += 6.0                                                              # 注入异常
    return x, sT

def main():
    X, sT = synth()
    L = 128

    # 1) 学生前向/参数量
    for name, cls in [("TCNLite", TCNLite), ("PatchMLPLite", PatchMLPLite)]:
        m = cls(c_in=X.shape[1], L=L)
        z = m(torch.randn(4, X.shape[1], L))
        ok(f"{name} forward", z.shape == (4,) and torch.isfinite(z).all(), f"params={count_params(m)/1e6:.2f}M")

    # 2) 四种训练案（少 epoch，CPU）
    for case, kw in [("A", {}), ("C", {}), ("B1", {}), ("B2", {"teacher_Xhat": (sT[:, None] * np.ones((1, X.shape[1]))).astype(np.float32)}), ("LINEAR", {})]:
        t0 = time.time()
        try:
            Xtr, Xte = X[:2000], X[2000:]
            sT_tr = np.concatenate([sT[:2000], np.full(len(Xte), np.nan)])[ :2000]
            sT_tr = sT[:2000]
            score_fn, info = train_case("TCNLite", Xtr, sT_tr, L=L, case=case, epochs=2, bs=128,
                                        device="cpu", stride=2, teacher_Xhat=(info_xhat(Xtr) if case == "B2" else None))
            s = score_fn(Xte)
            ok(f"case {case} train+score", np.isfinite(s[~np.isnan(s)]).all(), f"{time.time()-t0:.1f}s params={info['params']}")
        except Exception as e:
            ok(f"case {case} train+score", False, repr(e)[:120])

    # 3) ONNX 导出 + 计时
    try:
        net = TCNLite(X.shape[1], L).eval()
        p, mb = export_onnx(net, X.shape[1], L, "smoke_tcn.onnx")
        r = time_onnx(p, X.shape[1], L, n_iter=50)
        ok("ONNX export+timing", r["p95_ms"] > 0, f"file={mb:.2f}MB p50={r['p50_ms']:.2f}ms p95={r['p95_ms']:.2f}ms")
    except Exception as e:
        ok("ONNX export+timing", False, repr(e)[:120])

    # 4) 真实小流加载器
    try:
        from data import load_flow
        xtr, xte, y = load_flow("D13_JD1_svc0")
        ok("J-D1 svc0 loader", len(xte) == len(y) and xtr.shape[1] == xte.shape[1], f"train={xtr.shape} test={xte.shape} AR={100*y.mean():.2f}%")
    except Exception as e:
        ok("J-D1 svc0 loader", False, repr(e)[:120])
    try:
        from data import load_flow
        xtr, xte, y = load_flow("D14_NAB_cpu24ae8d")
        ok("NAB cpu loader", len(xte) == len(y), f"train={xtr.shape} test={xte.shape} AR={100*y.mean():.2f}%")
    except Exception as e:
        ok("NAB cpu loader", False, repr(e)[:120])
    try:
        from data import load_flow
        xtr, xte, y = load_flow("C4_MGAB")
        ok("MGAB loader", len(xte) == len(y), f"train={xtr.shape} test={xte.shape} AR={100*y.mean():.2f}%")
    except Exception as e:
        ok("MGAB loader", False, repr(e)[:120])

    n_fail = sum(1 for _, c in PASS if not c)
    print(f"\n==== 冒烟结果: {len(PASS)-n_fail}/{len(PASS)} PASS ====")
    return n_fail == 0

def info_xhat(X):
    """B2 冒烟用的假教师预测（真实场景用缓存）。"""
    return X + np.abs(np.random.default_rng(1).standard_normal(X.shape)).astype(np.float32) * 0.1

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
