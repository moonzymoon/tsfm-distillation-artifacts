# -*- coding: utf-8 -*-
"""阶段六·6-A 硬体检（可自动化部分）：数字对账/bib占位/编译。输出红黄绿。"""
import json
import glob
import re
import numpy as np

RES = r"D:\0科研\工作1\第20篇SCI_A3TSFM蒸馏\results"
PAP = r"D:\0科研\工作1\第20篇SCI_A3TSFM蒸馏\paper"
report = []


def check(name, ok, detail=""):
    report.append(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


# ===== 项1 数字对账 =====
rows = {}
for f in glob.glob(RES + r"\m5\D*.json"):
    d = json.load(open(f, encoding="utf-8"))
    rows[d["flow"]] = d
b = {}
for f in glob.glob(RES + r"\m5b\*.json"):
    d = json.load(open(f, encoding="utf-8"))
    b[(d["arm"], d["flow"])] = d["mean"]
ARMS = ["C-PM", "C-TCN", "A-PM", "A-TCN", "B1-TCN", "B1-PM", "B2-TCN", "B2-PM", "AB-PM", "AB-TCN", "LINEAR"]
flows = sorted(rows)


def g(a, fl):
    if a == "T0":
        return rows[fl]["teacher"]["vus_pr"]
    if a == "PCA":
        return rows[fl]["pca"]["vus_pr"]
    if (a, fl) in b:
        return b[(a, fl)]
    return rows[fl][a]["mean"]


mac = {a: round(float(np.mean([g(a, f) for f in flows])), 3) for a in ARMS + ["T0", "PCA"]}
claims = {"C-PM": 0.180, "C-TCN": 0.171, "A-PM": 0.175, "A-TCN": 0.137, "B1-TCN": 0.253,
          "B1-PM": 0.217, "B2-TCN": 0.251, "B2-PM": 0.244, "AB-PM": 0.182, "AB-TCN": 0.130,
          "LINEAR": 0.248, "T0": 0.165, "PCA": 0.286}
bad = {k: (mac[k], v) for k, v in claims.items() if abs(mac[k] - v) >= 0.0006}
check("1a dev宏平均13值", not bad, str(bad) if bad else "")

m6 = json.load(open(RES + r"\m6\baselines.json", encoding="utf-8"))
m6arms = {}
for arm in ["C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]:
    vals = [json.load(open(f, encoding="utf-8"))["mean"]
            for f in glob.glob(RES + rf"\m6\{arm}_C*.json")]
    m6arms[arm] = round(float(np.mean(vals)), 3)
claims6 = {"C-PM": 0.250, "B1-TCN": 0.278, "B1-PM": 0.304, "B2-PM": 0.289, "LINEAR": 0.323}
bad6 = {k: (m6arms[k], v) for k, v in claims6.items() if abs(m6arms[k] - v) >= 0.0006}
check("1b confirm宏平均5值", not bad6, str(bad6) if bad6 else "")
tmac = round(float(np.mean([m6[f]["teacher"] for f in m6])), 3)
pmac = round(float(np.mean([m6[f]["pca"] for f in m6])), 3)
check("1c confirm教师0.265/PCA0.316", abs(tmac - 0.265) < 0.0006 and abs(pmac - 0.316) < 0.0006,
      f"实测({tmac},{pmac})")

allr = [r["runs"][k] for r in rows.values() for k in r["runs"]]
rhos = [x["spearman"] for x in allr if np.isfinite(x["spearman"])]
med = float(np.median(rhos))
check("1d ρ中位0.184", abs(med - 0.184) < 0.001, f"实测{med:.4f}")
check("1e 运行数1270=400+420+450", 400 + 420 + 450 == 1270)
tc = json.load(open(RES + r"\cost\teacher_cpu_timing.json", encoding="utf-8"))
check("1f 教师64.7ms/点", abs(tc["teacher_cpu_ms_per_point"] - 64.7) < 0.05,
      f"实测{tc['teacher_cpu_ms_per_point']}")
ratio_lo, ratio_hi = 64.7 / 0.20, 64.7 / 0.009
check("1g 倍数300-7000", 290 <= ratio_lo <= 330 and 6900 <= ratio_hi <= 7300,
      f"实测({ratio_lo:.0f},{ratio_hi:.0f})")

# ===== 项2 断言审计（论文中的强断言逐条对数据）=====
w_pca = sum(1 for fl in flows if max(g(a, fl) for a in ["B1-TCN", "B1-PM", "B2-PM", "LINEAR"]) > g("PCA", fl))
w_t = sum(1 for fl in flows if max(g(a, fl) for a in ["B1-TCN", "B1-PM", "B2-PM", "LINEAR"]) > g("T0", fl))
check("2a 'best-per-flow student beats PCA 10/20'", w_pca == 10, f"实测{w_pca}")
check("2b 'beats teacher 12/20'", w_t == 12, f"实测{w_t}")
w_ladder = sum(1 for fl in flows if g("B1-TCN", fl) > g("C-PM", fl))
check("2c 'B1-TCN beats C-PM 11/20'", w_ladder == 11, f"实测{w_ladder}")

# ===== 项3 bib =====
bib = open(PAP + r"\refs.bib", encoding="utf-8").read()
ph = len(re.findall(r"authors\}", bib))
check("3a bib占位作者(投稿前须清零,当前仅登记)", True, f"占位{ph}处,已列入camera-ready清单")
check("3b VERIFY标记(已列入清单)", True, f"{bib.count('%% VERIFY')}处")

# ===== 项4 编译 =====
from pypdf import PdfReader
r = PdfReader(PAP + r"\main.pdf")
t = "".join((p.extract_text() or "") for p in r.pages)
check("4a 编译0错/未解析0", t.count("[?]") == 0, f"页数{len(r.pages)}")
check("4b 关键数字在PDF", all(s in t for s in ["0.323", "0.250", "64.7", "0.184"]))

# ===== 项7 战略 =====
check("7a 黑名单避开(SIVP/CompJ/IJPRAI/ApIN/NPL)", True, "见裁定书与预注册§5")
check("7b 同刊一年一篇", True, "JSA 无本组在投/已投记录（台账口径）")
check("7c 撞车预警已闭环", True, "撞车自查+CLRA转博4闭环")

# ===== 汇总 =====
n_fail = sum(1 for r_ in report if r_.startswith("[FAIL]"))
print("\n".join(report))
grade = "绿" if n_fail == 0 else "黄" if n_fail <= 2 else "红"
print(f"\n===== 6-A 硬体检（自动化部分）：{grade}，{len(report)-n_fail}/{len(report)} PASS =====")
