# -*- coding: utf-8 -*-
"""开销轴计时矩阵（预注册 §6，CPU）：代表维度 {1,9,38,66,116,248} × 臂产物
（TCN/PatchMLP/LINEAR 的 ONNX FP32 与 INT8）端到端 p50/p95 + 文件大小。
随机权重（推理延迟与权重值无关；正文披露）。B2/B1 的打分路径与 A 相同（残差 max
在 numpy 侧，不计入模型延迟——与正文口径一致：模型=窗口→分数头）。
用法: python timing_matrix.py
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from students import TCNLite, PatchMLPLite
from export_time import export_onnx, time_onnx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "cost")
os.makedirs(OUT, exist_ok=True)
L = 128
DIMS = [1, 9, 38, 66, 116, 248]


class LinearScorer(torch.nn.Module):
    """LINEAR 臂产物：L-1 输入 → 线性 → 标量分数（窗口→分数口径）。"""
    def __init__(self, c_in):
        super().__init__()
        self.lin = torch.nn.Linear(c_in * (L - 1), 1)

    def forward(self, x):
        return self.lin(x[:, :, :-1].flatten(1)).squeeze(-1)


def main():
    rows = []
    for C in DIMS:
        models = {
            "TCN": TCNLite(C, L).eval(),
            "PatchMLP": PatchMLPLite(C, L).eval(),
            "LINEAR": LinearScorer(C).eval(),
        }
        for name, net in models.items():
            base = os.path.join(OUT, f"{name}_C{C}.onnx")
            p_fp, mb_fp = export_onnx(net, C, L, base, quantize_int8=False)
            r_fp = time_onnx(p_fp, C, L, n_iter=200)
            rows.append({"dim": C, "model": name, "prec": "FP32",
                         "file_MB": round(mb_fp, 3), **{k: round(v, 4) if isinstance(v, float) else v
                                                        for k, v in r_fp.items() if k != "rss_mb"}})
            try:
                import tempfile, shutil
                from onnxruntime.quantization import quantize_dynamic, QuantType
                # onnx 形状推断的中间文件在非 ASCII 路径下静默写入失败 → 全程在 ASCII 临时目录量化
                with tempfile.TemporaryDirectory() as td:
                    src = os.path.join(td, "fp32.onnx")
                    dst = os.path.join(td, "int8.onnx")
                    shutil.copy(p_fp, src)
                    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
                    p_q = base.replace(".onnx", "_int8.onnx")
                    shutil.move(dst, p_q)
                mb_q = os.path.getsize(p_q) / 1e6
                r_q = time_onnx(p_q, C, L, n_iter=200)
                rows.append({"dim": C, "model": name, "prec": "INT8",
                             "file_MB": round(mb_q, 3), **{k: round(v, 4) if isinstance(v, float) else v
                                                           for k, v in r_q.items() if k != "rss_mb"}})
            except Exception as e:
                import traceback
                rows.append({"dim": C, "model": name, "prec": "INT8",
                             "error": traceback.format_exc()[-500:]})
    # 汇总表
    import collections
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["dim"], r["model"])][r["prec"]] = r
    lines = ["# 计时矩阵（桌面 CPU，batch=1，L=128，随机权重；2026-10-04）", "",
             "| 维度 | 模型 | FP32 p95(ms) | INT8 p95(ms) | FP32 MB | INT8 MB | E1(≤1ms,≤1MB) | E2 | E3 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (C, name), d in sorted(by.items()):
        fp = d.get("FP32", {}); q = d.get("INT8", {})
        def tier(r):
            if not r or "p95_ms" not in r:
                return "—"
            p, m = r["p95_ms"], r["file_MB"]
            if p <= 1 and m <= 1: return "E1"
            if p <= 5 and m <= 5: return "E2"
            if p <= 20 and m <= 20: return "E3"
            return "超档"
        lines.append(f"| {C} | {name} | {fp.get('p95_ms','—')} | {q.get('p95_ms','—')} "
                     f"| {fp.get('file_MB','—')} | {q.get('file_MB','—')} | {tier(fp)} | {tier(q)} | {tier(q)} |")
    open(os.path.join(OUT, "timing_matrix.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(rows, open(os.path.join(OUT, "timing_matrix.json"), "w"), indent=1)
    print("\n".join(lines))
    print("\n→ results/cost/timing_matrix.md")


if __name__ == "__main__":
    main()
