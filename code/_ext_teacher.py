# -*- coding: utf-8 -*-
"""E2_CATSv2 教师缓存独立执行（可见逐通道进度）。产出 results/teacher_cache/E2_CATSv2.npz。"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_flow
from teacher import BoltScorer

npz = os.path.join(HERE, "..", "results", "teacher_cache", "E2_CATSv2.npz")
t0 = time.time()
sc = BoltScorer()
print("bolt loaded %.0fs" % (time.time() - t0), flush=True)
xtr, xte, y = load_flow("E2_CATSv2")
print("E2 train %s test %s" % (list(xtr.shape), list(xte.shape)), flush=True)
s_tr, xh_tr = sc.score_flow(xtr, batch_size=2048)
print("train scored %.0fs finite %d/%d" % (time.time() - t0, int(np.isfinite(s_tr).sum()), len(s_tr)), flush=True)
s_te, xh_te = sc.score_flow(xte, batch_size=2048)
print("test scored %.0fs" % (time.time() - t0), flush=True)
np.savez_compressed(npz, s_tr=s_tr, s_te=s_te, xhat_tr=xh_tr, xhat_te=xh_te)
import json
json.dump({"flow": "E2_CATSv2", "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "wallclock_s": round(time.time() - t0, 1)},
          open(npz.replace(".npz", ".meta.json"), "w"), indent=1)
print("E2 CACHE DONE")
