# -*- coding: utf-8 -*-
"""P2/P3 教师分数缓存（幂等：已有 npz 的流跳过）。预注册 v1.0 §2/§9。
输出 results/teacher_cache/{flow_id}.npz + .meta.json
  s_tr/s_te: 教师聚合分（NaN=未定义段）  xhat_tr/xhat_te: 逐通道中位数预测
meta: 教师名/上下文/时间戳/耗时/形状/AR —— 预注册§8 出处模板。
P3: TimesFM-200m 小子集（未安装则记录跳过，属次要臂不阻塞）。
"""
import os
import json
import time
import hashlib
import datetime
import numpy as np

from data import load_flow, all_flow_ids
from teacher import BoltScorer, CTX_LEN

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")
os.makedirs(CACHE, exist_ok=True)

PILOT_FIRST = ["D1_SWaT_id1", "D11_SVDB_id2", "D3_SMD_id4"]
TIMESFM_SUBSET = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D18_Exathlon_id1", "D12_EMG"]


def order():
    """试点先行 → dev 流（M5 核心实验需要）→ confirm 流殿后（仅 GO 后 M6 需要）。"""
    ids = all_flow_ids()
    dev = [i for i in ids if i.startswith("D") and i not in PILOT_FIRST]
    conf = [i for i in ids if i.startswith("C")]
    return PILOT_FIRST + dev + conf


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_primary(only=None, log=print):
    sc = BoltScorer("amazon/chronos-bolt-base")
    done, failed = [], []
    for fid in order():
        if only and fid not in only:
            continue
        npz = os.path.join(CACHE, f"{fid}.npz")
        if os.path.exists(npz):
            log(f"[cache] 跳过已有 {fid}")
            continue
        t0 = time.time()
        try:
            xtr, xte, y = load_flow(fid)
            s_tr, xh_tr = sc.score_flow(xtr, batch_size=2048)
            s_te, xh_te = sc.score_flow(xte, batch_size=2048)
        except Exception as e:
            failed.append(fid)
            log(f"[cache] {fid} 失败（跳过继续）: {repr(e)[:120]}")
            continue
        np.savez_compressed(npz, s_tr=s_tr, s_te=s_te, xhat_tr=xh_tr, xhat_te=xh_te)
        meta = {
            "flow": fid, "teacher": "amazon/chronos-bolt-base", "context_len": CTX_LEN,
            "built": datetime.datetime.now().isoformat(timespec="seconds"),
            "wallclock_s": round(time.time() - t0, 1),
            "train_shape": list(xtr.shape), "test_shape": list(xte.shape),
            "test_AR": round(float(y.mean()), 4),
            "train_cut_rule": "TSB-AD-M=首个异常点前; 其余=前50/60% (见data.py注释)",
        }
        json.dump(meta, open(npz.replace(".npz", ".meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        done.append(fid)
        log(f"[cache] {fid} 完成 {meta['wallclock_s']}s train={xtr.shape} test={xte.shape}")
    if failed:
        log(f"[cache] 本轮失败 {len(failed)} 流: {failed}（修复后重跑即可，幂等）")
    return done


def build_timesfm_subset(log=print):
    try:
        import timesfm  # noqa
    except ImportError:
        log("[timesfm] 未安装，尝试 pip 安装（预算15分钟）...")
        import subprocess
        r = subprocess.run([ "pip", "install", "timesfm", "-q"], timeout=900)
        try:
            import timesfm  # noqa
        except ImportError:
            log("[timesfm] 安装/导入失败 → P3 跳过（次要臂，不阻塞主包），留待后续夜窗")
            return []
    # TimesFM 接口在 P3 实装前先记录占位：本轮仅验证环境，真正小子集打分在下次夜窗
    log("[timesfm] 环境可用；打分实装留待独立夜窗（P3 本轮只做环境验证）")
    return TIMESFM_SUBSET


if __name__ == "__main__":
    import sys
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    done = build_primary(only)
    print(f"[cache] 本轮新建 {len(done)} 流: {done}")
