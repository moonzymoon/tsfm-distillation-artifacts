# -*- coding: utf-8 -*-
"""TSB-AD-M.zip 盲态选流 v2：CreditCard/CATSv2/TAO/LTDB 各选 1 流。

盲态 = 仅检标签/长度/切分位置，不接触任何模型指标。
标准（R16-B + 纯净训练段）：逐点二值标签、首异常 ≥20% 文件长（无回退→训练段严格
normal-only）、测试段长度 ≥2000、测试段含异常。
每源取第一个合格流（按文件名序），全部候选与排除原因记录。
产出：results/extension_selection.md + results/extension_flows.json
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "TSB-AD-M.zip")
OUT_MD = os.path.join(HERE, "..", "results", "extension_selection.md")
OUT_JS = os.path.join(HERE, "..", "results", "extension_flows.json")
SOURCES = ["CreditCard", "CATSv2", "TAO", "LTDB"]


def main():
    zf = zipfile.ZipFile(ZIP)
    names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
    print("csv files in zip:", len(names))
    picked, log = {}, []
    for src in SOURCES:
        cands = [n for n in names if src.lower() in os.path.basename(n).lower()]
        if not cands:
            log.append("%s: NO FILES in TSB-AD-M" % src)
            continue
        chosen = None
        for n in cands:
            try:
                df = pd.read_csv(zf.open(n))
            except Exception as e:
                log.append("%s | %s READ FAIL %s" % (src, os.path.basename(n), e))
                continue
            if "Label" not in df.columns:
                log.append("%s | %s no Label column" % (src, os.path.basename(n)))
                continue
            lab = df["Label"].to_numpy().astype(np.int8)
            T = len(lab)
            if T < 4000:
                log.append("%s | %s too short T=%d (<4000)" % (src, os.path.basename(n), T))
                continue
            first = int(np.argmax(lab > 0)) if lab.any() else T
            first_pct = 100.0 * first / T
            if first < 0.2 * T:
                log.append("%s | %s fallback-split would trigger (first %.1f%%) — excluded for clean-train requirement"
                           % (src, os.path.basename(n), first_pct))
                continue
            cut = first
            test_len = T - cut
            test_anom = int((lab[cut:] > 0).sum())
            if test_len < 2000 or test_anom == 0:
                log.append("%s | %s test segment inadequate (len %d, anom %d)" % (
                    src, os.path.basename(n), test_len, test_anom))
                continue
            tr_anom = int((lab[:cut] > 0).sum())
            if tr_anom > 0:
                log.append("%s | %s train has anomalies (%d) — excluded" % (
                    src, os.path.basename(n), tr_anom))
                continue
            log.append("%s | %s QUALIFIED (T=%d, cut@%.1f%%, test len %d, test anom %d, train clean)"
                       % (src, os.path.basename(n), T, first_pct, test_len, test_anom))
            chosen = (n, T, cut, test_anom)
            break
        if chosen:
            picked[src] = chosen
        else:
            log.append("%s: NO qualified stream" % src)

    md = ["# 扩源盲态选流 v2（R17 候选；盲态=仅标签/长度/切分位置核验，无模型指标）", ""]
    md += ["- " + l for l in log]
    md += ["", "## 选定流", ""]
    for src, (n, T, cut, ta) in picked.items():
        md.append("- **%s**: `%s`（T=%d，切分@%d，测试段 %d 点含 %d 异常）"
                  % (src, os.path.basename(n), T, cut, T - cut, ta))
    md += ["", "选定数: %d / 4。不足 4 则按 R16-B 精神如实报告并缩减扩展规模（不凑数）。" % len(picked)]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    json.dump({k: {"file": v[0], "T": v[1], "cut": v[2], "test_anom": v[3]}
               for k, v in picked.items()}, open(OUT_JS, "w"), indent=1)
    print("\n".join(md[-8:]))
    print("picked:", len(picked))


if __name__ == "__main__":
    main()
