# -*- coding: utf-8 -*-
"""Raspberry Pi 4B on-device timing — mirrors code/timing_matrix_trained.py protocol.

Single-thread ORT, batch 1, L=128, 5 rounds x (50 warm + 500 timed), fixed random
window (latency is data-independent). Each config runs in a fresh subprocess so
ru_maxrss peak RSS is per-config. Output: ~/rpi_timing/rpi_timing_raw.json

Usage on Pi: python3 rpi_timing_run.py
"""
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L, C = 128, 66
N_ROUNDS, N_WARM, N_ITER = 5, 50, 500
CONFIGS = [
    ("LINEAR", "TR_LINEAR_C66.onnx", "FP32"),
    ("LINEAR", "TR_LINEAR_C66_int8.onnx", "INT8"),
    ("PatchMLP", "TR_PatchMLP_C66.onnx", "FP32"),
    ("PatchMLP", "TR_PatchMLP_C66_int8.onnx", "INT8"),
    ("TCN", "TR_TCN_C66.onnx", "FP32"),
    ("TCN", "TR_TCN_C66_int8.onnx", "INT8"),
]


def cpu_temp():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return round(int(f.read().strip()) / 1000.0, 1)


def run_one(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, C, L)).astype(np.float32)
    inp = {sess.get_inputs()[0].name: x}
    rounds = []
    for _ in range(N_ROUNDS):
        for _ in range(N_WARM):
            sess.run(None, inp)
        ts = []
        for _ in range(N_ITER):
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
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"rounds": rounds, "agg": agg,
            "file_MB": round(os.path.getsize(path) / 1e6, 3),
            "rss_mb": round(rss_mb, 1), "ort_version": ort.__version__}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print(json.dumps(run_one(sys.argv[2])))
        return
    rows, t0 = [], time.time()
    temps = [cpu_temp()]
    for name, fn, prec in CONFIGS:
        p = os.path.join(HERE, fn)
        assert os.path.exists(p), "missing artifact: " + p
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", p],
                             capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append({"model": name, "prec": prec, "dim": C, **r})
        temps.append(cpu_temp())
        print("%s %s done (p95 med %.4f ms, rss %.1f MB)"
              % (name, prec, r["agg"]["p95_ms"]["median"], r["rss_mb"]), flush=True)
    meta = {"protocol": "Raspberry Pi 4B Rev 1.4 (4 GB), Raspberry Pi OS 64-bit "
                        "(Debian 12 bookworm), ONNX Runtime %s, single thread, batch 1, "
                        "L=%d; 5 rounds x (%d warm + %d timed); CPU governor=performance"
                        % (rows[0]["ort_version"], L, N_WARM, N_ITER),
            "temp_c_start": temps[0], "temp_c_end": temps[-1],
            "temp_c_series": temps,
            "n_procs": os.cpu_count(), "date": time.strftime("%Y-%m-%d %H:%M"),
            "wall_s": round(time.time() - t0, 1)}
    json.dump({"rows": rows, "meta": meta},
              open(os.path.join(HERE, "rpi_timing_raw.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("wrote rpi_timing_raw.json | temp %s -> %s C"
          % (temps[0], temps[-1]))


if __name__ == "__main__":
    main()
