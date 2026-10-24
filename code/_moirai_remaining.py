# -*- coding: utf-8 -*-
"""Moirai D3+D11 打分 + 全部训练 + 报告。独立运行，不被 Bash 工具回收。"""
import json
import os
import sys
import time

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
LOG = open(os.path.join(HERE, "..", "results", "moirai_remaining.log"), "a", encoding="utf-8")


def logp(m):
    msg = "[%s] %s\n" % (time.strftime("%m-%d %H:%M:%S"), m)
    LOG.write(msg)
    LOG.flush()
    print(msg, end="")


def remaining_flows():
    """返回未完成打分的流列表。"""
    need = []
    for fid in ["D3_SMD_id4", "D11_SVDB_id2"]:
        npz = os.path.join(HERE, "..", "results", "teacher_cache_moirai", fid + ".npz")
        if not os.path.exists(npz):
            need.append(fid)
    return need


def score_all():
    from run_moirai_score import main as score_main
    need = remaining_flows()
    if not need:
        logp("all caches done, skip scoring")
        return
    logp("scoring: %s" % need)
    # 临时改 SUBSET 只跑缺的
    import run_moirai_score as rms
    rms.SUBSET = need
    rms.main()


def train_all():
    from run_moirai_train import main as train_main
    train_main()


if __name__ == "__main__":
    logp("pipeline start")
    try:
        score_all()
        train_all()
        logp("ALL COMPLETE")
    except Exception as e:
        logp("ERROR: %s" % str(e)[:300])
    logp("pipeline end")
