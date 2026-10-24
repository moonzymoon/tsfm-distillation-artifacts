# -*- coding: utf-8 -*-
"""开销轴计时矩阵 v2（DS33/GPT16 修复）：训练后工件 + 多轮重复 + INT8 质量漂移。

与旧 timing_matrix.py 的区别（响应外审）：
1. 工件为**真实训练后的学生**（案C 排名蒸馏，真教师缓存，M6 协议 epochs=30/
   bs=256/lr=1e-3/stride=2/seed=0），不再是随机权重；
2. 每配置 5 轮重复（50 warm + 500 iter），报轮间 median/min/max（空载机器）；
3. 每档补 FP32 vs INT8 制品的测试集 VUS-PR（量化质量漂移），支撑 Fig 7 口径。
维度→真实流：1=SVDB单通道, 9=Daphnet, 38=SMD, 66=SWaT, 116=WADI, 248=OPP。
用法: python timing_matrix_trained.py
"""
import os
import sys
import json
import time
import shutil
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from students import TCNLite, PatchMLPLite, make_windows
from distill import train_case
from data import load_flow
from run_pilot import vus_pr, L_WIN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "cost")
TR = os.path.join(HERE, "..", "results", "teacher_cache")
os.makedirs(OUT, exist_ok=True)
L = L_WIN
DIMS = {1: ("D11_SVDB_id2", True), 9: ("C3_Daphnet_id1", False),
        38: ("D3_SMD_id4", False), 66: ("D1_SWaT_id1", False),
        116: ("C1_WADI", False), 248: ("D19_OPP_id3", False)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_ROUNDS, N_WARM, N_ITER = 5, 50, 500


class LinearScorer(torch.nn.Module):
    """LINEAR 臂产物：L-1 输入 → 线性 → 标量分数（窗口→分数口径）。"""
    def __init__(self, c_in):
        super().__init__()
        self.lin = torch.nn.Linear(c_in * (L - 1), 1)

    def forward(self, x):
        return self.lin(x[:, :, :-1].flatten(1)).squeeze(-1)


def time_onnx_rounds(onnx_path, c_in, n_rounds=N_ROUNDS, n_warm=N_WARM, n_iter=N_ITER):
    """多轮重复计时：每轮独立 warm-up + 计时，报各轮 p50/p95 与轮间汇总。"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, c_in, L).astype(np.float32)
    inp = {sess.get_inputs()[0].name: x}
    rounds = []
    for _ in range(n_rounds):
        for _ in range(n_warm):
            sess.run(None, inp)
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            sess.run(None, inp)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000.0
        rounds.append({"p50_ms": float(np.percentile(ts, 50)),
                       "p95_ms": float(np.percentile(ts, 95)),
                       "mean_ms": float(ts.mean())})
    agg = {k: {"median": round(float(np.median([r[k] for r in rounds])), 4),
               "min": round(float(np.min([r[k] for r in rounds])), 4),
               "max": round(float(np.max([r[k] for r in rounds])), 4)}
           for k in ("p50_ms", "p95_ms", "mean_ms")}
    return {"rounds": rounds, "agg": agg}


def onnx_score(onnx_path, X, c_in):
    """ONNX 制品逐窗打分（与 distill.score_fn 相同时点口径）。"""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    W, tidx = make_windows(X, L, 1)
    W = W[:, :, :-1] if c_in == 0 else W          # LINEAR 在 wrapper 内自切
    outs = []
    for i in range(0, len(W), 512):
        outs.append(sess.run(None, {sess.get_inputs()[0].name:
                                    np.ascontiguousarray(W[i:i + 512])})[0])
    s = np.full(len(X), np.nan, dtype=np.float32)
    s[tidx] = np.concatenate(outs).astype(np.float32)
    return s


def quantize(src, tag):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    with tempfile.TemporaryDirectory() as td:
        s2, d2 = os.path.join(td, "a.onnx"), os.path.join(td, "b.onnx")
        shutil.copy(src, s2)
        quantize_dynamic(s2, d2, weight_type=QuantType.QInt8)
        dst = src.replace(".onnx", f"_{tag}.onnx")
        shutil.move(d2, dst)
    return dst


def export(net, c_in, path):
    net = net.cpu().eval()
    dummy = torch.randn(1, c_in, L)
    try:
        torch.onnx.export(net, dummy, path, opset_version=17, dynamo=False,
                          input_names=["window"], output_names=["score"],
                          dynamic_axes={"window": {0: "batch"}, "score": {0: "batch"}})
    except TypeError:                               # 旧版 torch 无 dynamo 参数
        torch.onnx.export(net, dummy, path, opset_version=17,
                          input_names=["window"], output_names=["score"],
                          dynamic_axes={"window": {0: "batch"}, "score": {0: "batch"}})
    return path


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    rows, drift = [], []
    for C, (fid, single_ch) in sorted(DIMS.items()):
        xtr, xte, y = load_flow(fid)
        if single_ch:
            xtr, xte = xtr[:, :1], xte[:, :1]
        npz = np.load(os.path.join(TR, f"{fid}.npz"))
        s_tr = npz["s_tr"]
        print(f"[{C}ch] {fid}: train {xtr.shape}", flush=True)
        # 训练三个真实学生（案C / LINEAR；权重进入导出工件）
        nets = {}
        for name, backbone in (("TCN", TCNLite), ("PatchMLP", PatchMLPLite)):
            fn, info = train_case(backbone.__name__, xtr, s_tr, L=L, case="C",
                                  epochs=30, bs=256, lr=1e-3, seed=0,
                                  device=DEV, stride=2)
            nets[name] = fn.__defaults__[0]          # case A/C 的 score_fn(_net=net,...)
        fn_lin, _ = train_case("LINEAR", xtr, s_tr, L=L, case="LINEAR",
                               epochs=30, bs=256, lr=1e-3, seed=0,
                               device=DEV, stride=2)
        lin_net = fn_lin.__defaults__[0]             # score_fn(_lin=lin,...)
        scorer = LinearScorer(C)
        scorer.lin.load_state_dict(lin_net.state_dict())
        nets["LINEAR"] = scorer

        for name, net in nets.items():
            base = os.path.join(OUT, f"TR_{name}_C{C}.onnx")
            p_fp = export(net, C, base)
            rows.append({"dim": C, "flow": fid, "model": name, "prec": "FP32",
                         "file_MB": round(os.path.getsize(p_fp) / 1e6, 3),
                         **time_onnx_rounds(p_fp, C)})
            p_q = quantize(p_fp, "int8")
            rows.append({"dim": C, "flow": fid, "model": name, "prec": "INT8",
                         "file_MB": round(os.path.getsize(p_q) / 1e6, 3),
                         **time_onnx_rounds(p_q, C)})
            # 量化质量漂移（有标签测试集）
            try:
                q_fp = onnx_score(p_fp, xte, C)
                q_q = onnx_score(p_q, xte, C)
                drift.append({"dim": C, "flow": fid, "model": name,
                              "vus_fp32": round(float(vus_pr(q_fp, y)), 4),
                              "vus_int8": round(float(vus_pr(q_q, y)), 4)})
            except Exception as e:
                drift.append({"dim": C, "flow": fid, "model": name, "error": str(e)[:200]})
            print(f"  {name} C{C} done", flush=True)

    json.dump({"rows": rows, "drift": drift,
               "protocol": {"rounds": N_ROUNDS, "warm": N_WARM, "iter": N_ITER,
                            "threads": 1, "L": L, "device_train": DEV,
                            "date": time.strftime("%Y-%m-%d %H:%M")}},
              open(os.path.join(OUT, "timing_matrix_TRAINED.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    # 摘要表
    lines = ["# 计时矩阵 v2（训练后工件，空载桌面 CPU，batch=1，L=128，5轮×(50warm+500iter)）",
             f"训练: {time.strftime('%Y-%m-%d')} | 训练协议=M6(案C, epochs30/stride2/seed0) | 计时: 单线程 ORT", ""]
    lines.append("| 维度 | 模型 | 精度 | p50中位 | p95中位 | p95轮间min–max | MB | VUS FP32 | VUS INT8 | Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    dr = {(d["dim"], d["model"]): d for d in drift if "vus_fp32" in d}
    for r in rows:
        if r["prec"] != "FP32":
            continue
        q = next(x for x in rows if x["dim"] == r["dim"] and x["model"] == r["model"]
                 and x["prec"] == "INT8")
        d = dr.get((r["dim"], r["model"]), {})
        dv = (d.get("vus_int8", 0) - d.get("vus_fp32", 0))
        lines.append(f"| {r['dim']} | {r['model']} | FP32 | {r['agg']['p50_ms']['median']} "
                     f"| {r['agg']['p95_ms']['median']} | {r['agg']['p95_ms']['min']}–{r['agg']['p95_ms']['max']} "
                     f"| {r['file_MB']} | {d.get('vus_fp32','—')} | {d.get('vus_int8','—')} | "
                     f"{round(dv,4) if d else '—'} |")
        lines.append(f"| {r['dim']} | {r['model']} | INT8 | {q['agg']['p50_ms']['median']} "
                     f"| {q['agg']['p95_ms']['median']} | {q['agg']['p95_ms']['min']}–{q['agg']['p95_ms']['max']} "
                     f"| {q['file_MB']} | | | |")
    open(os.path.join(OUT, "timing_matrix_TRAINED.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("\n→ results/cost/timing_matrix_TRAINED.{json,md}")


if __name__ == "__main__":
    main()
