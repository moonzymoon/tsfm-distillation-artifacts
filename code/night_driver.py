# -*- coding: utf-8 -*-
"""夜晚驱动：P1 试跑 → P2 全量缓存 → P3 二教师 → P4 试点网格。
守卫=gpu_guard（避开博4），硬截止默认 07:30。全步骤幂等，中断可重跑。
用法: python night_driver.py [截止时刻 HH:MM，默认 07:30]
"""
import os
import sys
import json
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_guard import wait_free
from build_cache import PILOT_FIRST, build_primary, build_timesfm_subset

LOG = os.path.join(HERE, "..", "results", "night_log.md")


def log(msg):
    line = f"- `{datetime.datetime.now():%m-%d %H:%M:%S}` {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, cmd, deadline):
    log(f"### {name} 开始: `{cmd}`")
    r = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True,
                       timeout=max(60, int((deadline - datetime.datetime.now()).total_seconds())))
    (open(os.path.join(HERE, "..", "results", f"night_{name}.log"), "w", encoding="utf-8")
     .write(r.stdout[-20000:] + "\n===STDERR===\n" + r.stderr[-8000:]))
    ok = r.returncode == 0
    log(f"### {name} {'OK' if ok else 'FAIL(code %d)' % r.returncode}，日志 results/night_{name}.log")
    return ok


def main():
    hhmm = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "none"
    noguard = "--noguard" in sys.argv
    if hhmm == "none":
        deadline = None                                  # 无截止：跑完为止
    else:
        d = datetime.date.today()
        deadline = datetime.datetime.combine(d, datetime.time(*map(int, hhmm.split(":"))))
        if datetime.datetime.now() > deadline:
            deadline += datetime.timedelta(days=1)

    global wait_free
    if noguard:
        def wait_free(**kw):
            try:
                from gpu_guard import gpu_state
                u, used, t = gpu_state()
                log(f"[guard] --noguard 单次核查 util={u}% free={t-used}MB → 直接放行")
            except Exception as e:
                log(f"[guard] --noguard（状态查询失败 {repr(e)[:50]}）→ 直接放行")
            return True

    log(f"\n# 夜晚驱动启动 {datetime.datetime.now():%Y-%m-%d %H:%M}，截止 "
        + ("无（跑完为止）" if deadline is None else f"{deadline:%H:%M}")
        + f"，noguard={noguard}")

    # P0 守卫放行
    if not wait_free(deadline=deadline, log=log):  # deadline=None 时永不超时
        log("# 守卫未放行 → 全包退出（幂等，可今晚重跑）")
        return
    build_primary(only=PILOT_FIRST, log=log)   # 试点流缓存先行（供 P1）

    # P1 端到端试跑：SWaT_id_1 单流单种子（幂等：已完成则跳过）
    j = os.path.join(HERE, "..", "results", "pilot", f"smoketest_{PILOT_FIRST[0]}.json")
    if os.path.exists(j):
        log("# P1 已有试跑结果，跳过")
    elif (deadline is None or datetime.datetime.now() < deadline) and wait_free(deadline=deadline, log=log):
        p1 = run_step("P1_smoketest", f'python run_pilot.py --flow {PILOT_FIRST[0]}', deadline)
        if p1 and os.path.exists(j):
            d1 = json.load(open(j))[PILOT_FIRST[0]]
            log(f"# P1 试跑结果: teacher={d1['teacher']['vus_pr']} pca={d1['pca']['vus_pr']} "
                f"A-TCN={d1['A-TCN']['mean']} 训练用时样例={list(d1['runs'].values())[0]['train_s']}s")
        elif p1:
            log("# P1 JSON 缺失（异常），继续 P2/P3，跳过 P4")

    # P2 全量缓存（幂等跳过已有）
    if (deadline is None or datetime.datetime.now() < deadline) and wait_free(deadline=deadline, log=log):
        run_step("P2_cache", "python build_cache.py", deadline)

    # P3 TimesFM 环境验证（打分实装留独立夜窗）
    if deadline is None or datetime.datetime.now() < deadline:
        try:
            build_timesfm_subset(log=log)
        except Exception as e:
            log(f"# P3 异常跳过: {repr(e)[:80]}")

    # P4 试点全网格（依赖 P2 的3条试点流缓存；幂等：已有报告则跳过）
    rep = os.path.join(HERE, "..", "results", "pilot", "pilot_report.md")
    if os.path.exists(rep):
        log("# P4 试点报告已存在，跳过（判定见 pilot_report.md）")
        log(f"# 夜晚驱动结束 {datetime.datetime.now():%H:%M:%S}")
        return
    need = [f for f in PILOT_FIRST if not os.path.exists(
        os.path.join(HERE, "..", "results", "teacher_cache", f + ".npz"))]
    if (deadline is None or datetime.datetime.now() < deadline) and not need and wait_free(deadline=deadline, log=log):
        run_step("P4_pilot", "python run_pilot.py", deadline)
        if os.path.exists(rep):
            log("# P4 完成，试点报告 results/pilot/pilot_report.md（GO 判定见报告）")
    elif need:
        log(f"# P4 跳过：试点流缓存缺失 {need}")

    log(f"# 夜晚驱动结束 {datetime.datetime.now():%H:%M:%S}")


if __name__ == "__main__":
    main()
