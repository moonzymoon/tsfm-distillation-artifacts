# -*- coding: utf-8 -*-
"""R15：Pi 4B 持续吞吐实测（10 分钟连续推理，30 秒采样吞吐/温度/频率）。

目的：验证无热节流下的持续性能（威胁(7) 的 sustained-throughput 项）。
对象：TR_PatchMLP_C66_int8.onnx（E2 档代表工件）+ TR_LINEAR_C66.onnx（E1 档对照）。
产出：~/rpi_timing/sustained.json
"""
import json
import os
import subprocess
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DUR_S = 600
SAMPLE_S = 30
CONFIGS = [("PatchMLP-INT8", "TR_PatchMLP_C66_int8.onnx"),
           ("LINEAR", "TR_LINEAR_C66.onnx")]


def read_temp():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return round(int(f.read().strip()) / 1000.0, 1)


def read_freq_mhz():
    out = subprocess.run(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) // 1000


out = {"protocol": "Raspberry Pi 4B Rev 1.4, 10-minute continuous inference, single-thread "
                   "ORT 1.31, batch 1, L=128 C=66, governor=performance; sampled every 30s",
       "configs": {}}
rng = np.random.default_rng(0)
for label, fn in CONFIGS:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(HERE, fn), so, providers=["CPUExecutionProvider"])
    x = rng.standard_normal((1, 66, 128)).astype(np.float32)
    inp = {sess.get_inputs()[0].name: x}
    for _ in range(100):
        sess.run(None, inp)                     # warm-up
    samples = []
    t_end = time.time() + DUR_S
    while time.time() < t_end:
        t0 = time.time()
        n = 0
        while time.time() - t0 < SAMPLE_S:
            sess.run(None, inp)
            n += 1
        dt = time.time() - t0
        samples.append({"t_s": round(time.time() - (t_end - DUR_S), 1),
                        "win_per_s": round(n / dt, 1),
                        "p50_us": round(dt / n * 1e6, 1),
                        "temp_c": read_temp(),
                        "freq_mhz": read_freq_mhz()})
        print(label, samples[-1], flush=True)
    first, last = samples[0], samples[-1]
    wps = [s["win_per_s"] for s in samples]
    out["configs"][label] = {
        "samples": samples,
        "first_win_per_s": first["win_per_s"], "last_win_per_s": last["win_per_s"],
        "min_win_per_s": min(wps), "max_win_per_s": max(wps),
        "degradation_pct": round(100 * (first["win_per_s"] - last["win_per_s"]) / first["win_per_s"], 2),
        "temp_start_c": first["temp_c"], "temp_max_c": max(s["temp_c"] for s in samples),
        "freq_min_mhz": min(s["freq_mhz"] for s in samples),
    }
    print(label, "SUMMARY:", {k: v for k, v in out["configs"][label].items() if k != "samples"},
          flush=True)
json.dump(out, open(os.path.join(HERE, "sustained.json"), "w"), indent=1)
print("wrote sustained.json")
