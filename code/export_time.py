# -*- coding: utf-8 -*-
"""导出与计时（预注册 §6）：ONNX 导出（opset 17 + 可选 INT8 动态量化），
batch=1 端到端 p50/p95 延迟、模型文件大小、峰值 RSS（psutil 可得则记）。
"""
import os
import time
import numpy as np
import torch


def export_onnx(net, c_in, L, path, quantize_int8=False):
    net = net.cpu().eval()
    dummy = torch.randn(1, c_in, L)
    try:
        torch.onnx.export(net, dummy, path, opset_version=17, dynamo=False,
                          input_names=["window"], output_names=["score"],
                          dynamic_axes={"window": {0: "batch"}, "score": {0: "batch"}})
    except TypeError:                                 # 旧版 torch 无 dynamo 参数
        torch.onnx.export(net, dummy, path, opset_version=17,
                          input_names=["window"], output_names=["score"],
                          dynamic_axes={"window": {0: "batch"}, "score": {0: "batch"}})
    size_mb = os.path.getsize(path) / 1e6
    if quantize_int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(path, q, weight_type=QuantType.QInt8)
        path, size_mb = q, os.path.getsize(q) / 1e6
    return path, size_mb


def time_onnx(onnx_path, c_in, L, n_warm=20, n_iter=200):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                      # 边缘口径：单窗口单线程（也抗后台负载噪声）
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, c_in, L).astype(np.float32)
    inp = {sess.get_inputs()[0].name: x}
    for _ in range(n_warm):
        sess.run(None, inp)
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, inp)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000.0
    rss = None
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "file_mb": os.path.getsize(onnx_path) / 1e6, "rss_mb": rss}
