# -*- coding: utf-8 -*-
"""链式启动：Moirai 打分（GPU）→ C-PM 训练 → 报告。单实例运行。"""
import os
import subprocess
import sys
import time

V = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
import os

log = open(os.path.join(HERE, "..", "results", "moirai_chain.log"), "a", encoding="utf-8")
log.write("[%s] chain start\n" % time.strftime("%m-%d %H:%M:%S"))
r1 = subprocess.run([V, "-u", "run_moirai_score.py"], cwd=HERE, capture_output=True, text=True)
log.write("score exit=%d\n%s\n" % (r1.returncode, r1.stdout[-500:]))
r2 = subprocess.run([V, "-u", "run_moirai_train.py"], cwd=HERE, capture_output=True, text=True)
log.write("train exit=%d\n%s\n" % (r2.returncode, r2.stdout[-500:]))
log.write("chain done\n")
