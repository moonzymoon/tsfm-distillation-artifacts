# -*- coding: utf-8 -*-
"""R16 前置审计：29 流训练段异常率与切分回退核查（泄漏审计第 0 步）。

_only _tsbadm 系存在"首异常<20% 时切 60%"回退分支 → 训练段可能含异常。
其余 loader（_hai/_nab/_tsbuad/_wadi 等）使用官方 train/test 文件切分，无此分支。
产出：results/train_leak_audit.md
"""
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D

OUTP = os.path.join(HERE, "..", "results", "train_leak_audit.md")

# 从 data.py 源码提取 _tsbadm 流 → idx_name 映射
src = open(os.path.join(HERE, "data.py"), encoding="utf-8").read()
tsb = dict(re.findall(r'"(D\d+_[A-Za-z0-9_]+|C\d+_[A-Za-z0-9_]+)":\s*lambda: _tsbadm\("([^"]+)"\)', src))

results = []   # (flow, T, first_anom_pct, fallback, cut, train_anom, train_anom_rate)
for fid, idx in sorted(tsb.items()):
    path = D.glob_one(idx)
    import pandas as pd
    df = pd.read_csv(path)
    label = df["Label"].to_numpy().astype(np.int8)
    T = len(label)
    first = int(np.argmax(label > 0)) if label.any() else T
    fallback = first < 0.2 * T
    cut = first if first >= 0.2 * T else int(0.6 * T)
    tr_anom = int((label[:cut] > 0).sum())
    tr_rate = tr_anom / cut if cut else 0.0
    results.append((fid, T, 100.0 * first / T, fallback, cut, tr_anom, 100.0 * tr_rate))

affected = [r for r in results if r[5] > 0]
lines = ["# 训练段泄漏审计（R16 前置，29 流 _tsbadm 系 " + str(len(results)) + " 流 + 其他 loader 静态核查）", "",
         "| 流 | 全长 | 首异常位置% | 回退触发 | 切分点 | 训练段异常点 | 训练段异常率 |",
         "|---|---|---|---|---|---|---|"]
for fid, T, fpct, fb, cut, ta, tr in results:
    lines.append("| %s | %d | %.1f%% | %s | %d | %d | %.2f%% |" % (
        fid, T, fpct, "YES" if fb else "no", cut, ta, tr))
n_fb = sum(1 for r in results if r[3])
lines += ["", "- 回退触发流数: %d / %d" % (n_fb, len(results)),
          "- 训练段含异常流数: %d / %d" % (len(affected), len(results))]
if affected:
    lines.append("- 受影响流明细: " + ", ".join("%s(%.2f%%)" % (r[0], r[6]) for r in affected))
# 其他 loader：官方 train/test 文件，切分与标签无关——静态说明
lines += ["", "- 非 _tsbadm 流（HAI/NAB/其他官方切片）使用官方 train/test 文件，切分不依赖标签；",
          "  其训练段纯度由数据集官方协议保证，不在本审计范围。"]
open(OUTP, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
print("\nAFFECTED:", len(affected))
