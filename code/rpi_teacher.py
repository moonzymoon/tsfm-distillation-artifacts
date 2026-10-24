# -*- coding: utf-8 -*-
"""R15b：Chronos-Bolt-base 教师在 Pi 4B 上的部署侧延迟实测。

与桌面教师计时同口径（batch 1，512 点上下文，端到端含归一化）。
产出：~/rpi_timing/teacher_pi.json
"""
import json
import time

import numpy as np
import torch

from chronos import BaseChronosPipeline

pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cpu",
    torch_dtype=torch.float32,
)
rng = np.random.default_rng(0)

ctx = np.ones((1, 512), dtype=np.float32)
ctx += rng.standard_normal((1, 512)).astype(np.float32) * 0.1

# warm-up（首次含模型加载后的内核编译，不计）
for _ in range(3):
    pipe.predict(context=torch.tensor(ctx), prediction_length=1)

ts = []
N = 30
for _ in range(N):
    t0 = time.perf_counter()
    pipe.predict(context=torch.tensor(ctx), prediction_length=1)
    ts.append(time.perf_counter() - t0)
ts = np.array(ts) * 1000.0
res = {
    "teacher": "amazon/chronos-bolt-base",
    "device": "Raspberry Pi 4B Rev 1.4 (cpu, float32)",
    "protocol": "batch 1, context 512, horizon 1, end-to-end predict call, "
                "3 warm-up + 30 timed",
    "p50_ms": round(float(np.percentile(ts, 50)), 1),
    "p95_ms": round(float(np.percentile(ts, 95)), 1),
    "mean_ms": round(float(ts.mean()), 1),
    "per_point_ms": round(float(np.mean(ts)), 3),
}
with open("/home/pi/rpi_timing/teacher_pi.json", "w") as f:
    json.dump(res, f, indent=1)
print(res)
