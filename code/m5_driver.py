# -*- coding: utf-8 -*-
"""M5 接续驱动：等 P2 缓存完成 → 试点保真复算（GPU ~10分钟）→ M5a 主网格（GPU+CPU）。
M5a 前用守卫避开博4（30分钟轮询，用户设定）。截止默认次日 07:30。
用法: python m5_driver.py [HH:MM 截止，默认07:30]
"""
import os
import sys
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_guard import wait_free
from data import all_flow_ids

LOG = os.path.join(HERE, "..", "results", "m5_driver_log.md")
CACHE = os.path.join(HERE, "..", "results", "teacher_cache")


def log(msg):
    line = f"- `{datetime.datetime.now():%m-%d %H:%M:%S}` {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dev_cache_done():
    need = [i for i in all_flow_ids() if i.startswith("D")
            and not os.path.exists(os.path.join(CACHE, i + ".npz"))]
    return need


def run_step(name, cmd, deadline):
    log(f"### {name} 开始: `{cmd}`")
    if deadline:
        timeout = max(600, int((deadline - datetime.datetime.now()).total_seconds()))
    else:
        timeout = None                                   # 无截止：跑完为止
    r = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True,
                       timeout=timeout)
    open(os.path.join(HERE, "..", "results", f"m5_{name}.log"), "w", encoding="utf-8").write(
        r.stdout[-20000:] + "\n===STDERR===\n" + r.stderr[-8000:])
    ok = r.returncode == 0
    log(f"### {name} {'OK' if ok else 'FAIL(code %d)' % r.returncode}，日志 results/m5_{name}.log")
    return ok


def main():
    hhmm = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "none"
    noguard = "--noguard" in sys.argv
    if hhmm == "none":
        deadline = None                                  # 用户指令：无截止，跑完为止
    else:
        d = datetime.date.today()
        deadline = datetime.datetime.combine(d, datetime.time(*map(int, hhmm.split(":"))))
        if datetime.datetime.now() > deadline:
            deadline += datetime.timedelta(days=1)
    log(f"\n# M5驱动启动 {datetime.datetime.now():%Y-%m-%d %H:%M}，截止 "
        + ("无（跑完为止）" if deadline is None else f"{deadline:%H:%M}"))

    # 阶段0：等 P2 完成（纯轮询不碰GPU）
    import time
    while True:
        need = dev_cache_done()
        if not need:
            log("# P2 dev缓存全部就绪")
            break
        if deadline and datetime.datetime.now() > deadline:
            log(f"# 截止：dev缓存仍缺 {len(need)} 流 → 退出（幂等续跑）")
            return
        log(f"# 等P2：dev缓存缺 {len(need)} 流 {need[:3]}...（10分钟后再查）")
        time.sleep(600)

    # 阶段1：试点保真复算（GPU ~10分钟）
    if (deadline is None or datetime.datetime.now() < deadline) and wait_free(deadline=deadline, log=log):
        run_step("pilot_fid", "python run_pilot.py", deadline)

    # 阶段2：M5a 主网格（守卫避开博4）
    if (deadline is None or datetime.datetime.now() < deadline) and wait_free(deadline=deadline, log=log):
        run_step("m5a", "python run_m5.py", deadline)
    rep = os.path.join(HERE, "..", "results", "m5", "m5_report.md")
    if os.path.exists(rep):
        log("# M5a 完成，报告 results/m5/m5_report.md（选型留待 5.3）")
    log(f"# M5驱动结束 {datetime.datetime.now():%H:%M:%S}")


if __name__ == "__main__":
    main()
