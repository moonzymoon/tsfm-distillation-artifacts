# -*- coding: utf-8 -*-
"""GPU 排班守卫：nvidia-smi 探测空闲（避开博4），连续两次满足才放行；带硬截止。"""
import subprocess
import time
import datetime


def gpu_state():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().split("\n")[0]
    util, used, total = [int(x) for x in out.split(",")]
    return util, used, total


def wait_free(max_util=30, min_free_mb=12000, patience=2, interval=1800,
              deadline=None, log=print):
    """连续 patience 次满足 (util<=max_util 且 空闲显存>=min_free_mb) → True；
    到 deadline 仍未放行 → False。"""
    good, n = 0, 0
    while True:
        if deadline and datetime.datetime.now() > deadline:
            log(f"[guard] {datetime.datetime.now():%H:%M:%S} 硬截止，放弃等待（已查 {n} 次）")
            return False
        try:
            util, used, total = gpu_state()
            free = total - used
            ok = util <= max_util and free >= min_free_mb
        except Exception as e:
            ok = False
            util = used = free = -1
            log(f"[guard] nvidia-smi 异常 {repr(e)[:60]}")
        good = good + 1 if ok else 0
        n += 1
        log(f"[guard] #{n} {datetime.datetime.now():%H:%M:%S} util={util}% free={free}MB "
            f"-> {'FREE' if ok else 'BUSY'} ({good}/{patience})")
        if good >= patience:
            return True
        time.sleep(interval)
